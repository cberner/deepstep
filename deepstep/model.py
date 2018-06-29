'''
Copyright 2016 the original author or authors.
See the NOTICE file distributed with this work for additional
information regarding copyright ownership.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from typing import List, Set, Tuple, Sequence, Any

import random
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from deepstep.midi import Sound, Track
from hyperflow import Hyperparameters, NeuralLayerType
from libs.dcgan import DCGAN


Dense = tf.keras.layers.Dense
Reshape = tf.keras.layers.Reshape
Dropout = tf.keras.layers.Dropout
LSTM = tf.keras.layers.LSTM
Sequential = tf.keras.models.Sequential


class Model(ABC):
    @abstractmethod
    def train(self, tracks: List[Track], epochs: int) -> None:
        pass

    @abstractmethod
    def evaluate(self, tracks: List[Track]) -> float:
        pass

    @abstractmethod
    def generate(self, seed_track: Track, measures: int) -> Track:
        pass


def next_power_of_2(value: int) -> int:
    for i in range(8):
        if 2**i >= value:
            return 2**i
    raise Exception()


class GAN(Model):
    def __init__(self,
                 notes: Set[int],
                 sound_volume: int) -> None:
        self.max_size = 32
        self.sound_volume = sound_volume
        # TOOD: support variable length note prediction
        self.sound_duration = 1
        self.id_to_note = dict((i, note) for (i, note) in enumerate(sorted(notes)))
        self.note_to_id = dict((note, i) for (i, note) in enumerate(sorted(notes)))

        # TODO: close this
        self.session = tf.Session(graph=tf.Graph())

        with self.session.graph.as_default():
            self.model = DCGAN(self.session,
                               output_height=self.max_size,
                               output_width=next_power_of_2(len(notes)),
                               c_dim=3)

    def train(self, tracks: List[Track], epochs: int) -> None:
        examples = self.__scores_to_matrices(tracks)
        with self.session.graph.as_default():
            self.model.train(learning_rate=0.0002, beta1=0.5, epochs=epochs, data=examples)

    def evaluate(self, tracks: List[Track]) -> float:
        raise Exception()

    def __scores_to_matrices(self, tracks: Sequence[Track]) -> np.ndarray:
        # XXX: hack because DCGAN needs powers of 2
        num_ids = next_power_of_2(len(self.note_to_id))

        num_examples = 0
        for track in tracks:
            assert track.ticks_per_beat == 4, "Track must be in sixteenth notes"
            if track.duration > self.max_size:
                num_examples += track.duration - self.max_size
        assert num_examples > 0

        print("Allocating {}x{}x{}x{} matrix of floats".format(num_examples, self.max_size, num_ids, 3))
        examples = np.full((num_examples, self.max_size, num_ids, 3), -1, dtype=np.float)
        randomized_example_indices = list(range(num_examples))
        random.shuffle(randomized_example_indices)
        for track in tracks:
            for i in range(track.duration - self.max_size):
                example_num = randomized_example_indices.pop()
                # Copy the seed section of the score
                for j in range(self.max_size):
                    for start, sound in track[i + j]:
                        examples[example_num, j, self.note_to_id[sound.note], 0] = 1
                        if start == i + j:
                            examples[example_num, j, self.note_to_id[sound.note], 1] = 1

        return examples

    def generate(self, seed_track: Track, measures: int) -> Track:
        # XXX: hackz. Should only pass seed_track once here,
        # but need to pass it twice to have enough data for the batch size
        seed_matrix = self.__scores_to_matrices([seed_track, seed_track])

        with self.session.graph.as_default():
            generated_matrix = self.model.generate(seed_matrix)[0][0]
        generated = []
        for note_id in range(self.max_size):
            if note_id not in self.id_to_note:
                continue

            note = self.id_to_note[note_id]
            start = -1
            for i in range(generated_matrix.shape[0]):
                present = generated_matrix[i, note_id][0] > 0
                starting = generated_matrix[i, note_id][1] > 0
                if not present:
                    if starting:
                        print("Bad prediction: starting a note, but it's not present")
                    if start == -1:
                        # not in a note, and not starting one
                        continue
                    else:
                        # end the note
                        sound = Sound(note=note,
                                      volume=self.sound_volume,
                                      duration=(i - start) * self.sound_duration)
                        generated.append((start * self.sound_duration, sound))
                else:
                    if start == -1:
                        if not starting:
                            print("Bad prediction: note became present without explicitly starting")
                        # start a new note
                        start = i
                    else:
                        if starting:
                            # end current note and start another one
                            sound = Sound(note=note,
                                          volume=self.sound_volume,
                                          duration=(i - start) * self.sound_duration)
                            generated.append((start * self.sound_duration, sound))
                            start = i

        return Track(Track(generated, seed_track.ticks_per_beat)[:measures * 4], seed_track.ticks_per_beat)


class DNN(Model):
    def __init__(self,
                 hyperparameters: Hyperparameters,
                 notes: Set[int],
                 look_back: int,
                 sound_volume: int) -> None:
        self.look_back = look_back
        self.sound_volume = sound_volume
        # TOOD: support variable length note prediction
        self.sound_duration = 1
        self.id_to_note = dict((i, note) for (i, note) in enumerate(sorted(notes)))
        self.note_to_id = dict((note, i) for (i, note) in enumerate(sorted(notes)))

        model = Sequential()
        # Dense layers require that the input be reshaped to remove the temporal dimension
        if hyperparameters.layers[0].layer_type == NeuralLayerType.DENSE:
            model.add(Reshape((self.look_back * len(notes),), input_shape=(self.look_back, len(notes))))

        first_layer = True
        for i, layer in enumerate(hyperparameters.layers):
            if not first_layer:
                model.add(Dropout(0.2))
            if layer.layer_type == NeuralLayerType.DENSE:
                model.add(Dense(layer.neurons, activation='relu'))
            elif layer.layer_type == NeuralLayerType.LSTM:
                parameters = {} # type: dict[str, Any]
                if first_layer:
                    parameters['input_shape'] = (self.look_back, len(notes))
                if i + 1 < len(hyperparameters.layers) and \
                    hyperparameters.layers[i + 1].layer_type == NeuralLayerType.LSTM:
                    parameters['return_sequences'] = True
                model.add(LSTM(layer.neurons, **parameters))
            else:
                raise Exception("Unsupported layer type: " + str(layer.layer_type))
            first_layer = False

        model.add(Dense(len(notes), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model = model

    def train(self, tracks: List[Track], epochs: int) -> None:
        examples, labels = self.__scores_to_matrices(tracks)
        self.model.fit(examples, labels, epochs=epochs)

    def evaluate(self, tracks: List[Track]) -> float:
        examples, labels = self.__scores_to_matrices(tracks)
        return self.model.evaluate(examples, labels, verbose=False)

    def __scores_to_matrices(self, tracks: Sequence[Track]) -> Tuple[np.ndarray, np.ndarray]:
        num_ids = len(self.note_to_id)
        num_examples = 0
        for track in tracks:
            assert track.ticks_per_beat == 4, "Track must be in sixteenth notes"
            if track.duration > self.look_back:
                num_examples += track.duration - self.look_back
        assert num_examples > 0

        examples = np.zeros((num_examples, self.look_back, num_ids), dtype=np.bool)
        labels = np.zeros((num_examples, num_ids), dtype=np.bool)
        example_num = 0
        for track in tracks:
            for i in range(track.duration - self.look_back):
                # Copy the seed section of the score
                for j in range(self.look_back):
                    for _, sound in track[i + j]:
                        examples[example_num, j, self.note_to_id[sound.note]] = 1
                # Copy the expected next note of the score
                for _, sound in track[i + self.look_back]:
                    labels[example_num, self.note_to_id[sound.note]] = 1
                example_num += 1

        return (examples, labels)

    def generate(self, seed_track: Track, measures: int) -> Track:
        seed = [] # type: List[Tuple[int, Sound]]
        for start, sound in seed_track[:self.look_back + 1]:
            note = sound.note
            if note not in self.note_to_id:
                note = random.choice(list(self.note_to_id.keys()))
            seed.append((start, Sound(note=note, volume=sound.volume, duration=self.sound_duration)))

        # Treat measures as 4/4 time
        length = 0
        generated = []
        while length < measures * 4:
            track = Track(seed, seed_track.ticks_per_beat, duration=self.look_back + 1)
            seed_ids, _ = self.__scores_to_matrices([track])

            # TODO: seed_ids could have more than one example, if the last notes have a duration > 1
            # just take the results of the first one, but should remove them from seed_ids
            predictions = self.model.predict(seed_ids, verbose=False)[0]
            predicted_note_ids = []
            for i, prediction in enumerate(predictions):
                if prediction > random.random():
                    predicted_note_ids.append(i)

            for note_id in predicted_note_ids:
                note = self.id_to_note[note_id]

                sound = Sound(note=note,
                              volume=self.sound_volume,
                              duration=self.sound_duration)
                seed.append((length, sound))
                generated.append((length, sound))
            seed = [(start - self.sound_duration, sound) for start, sound in seed if start >= self.sound_duration]
            length += self.sound_duration

        return Track(generated, seed_track.ticks_per_beat, duration=length)


class NormalizedTime(Model):
    def __init__(self, delegate: Model) -> None:
        self.__delegate = delegate

    def train(self, tracks: List[Track], epochs: int) -> None:
        self.__delegate.train([self.__normalize(track) for track in tracks], epochs)

    def evaluate(self, tracks: List[Track]) -> float:
        return self.__delegate.evaluate([self.__normalize(track) for track in tracks])

    def generate(self, seed_track: Track, measures: int) -> Track:
        generated = self.__delegate.generate(self.__normalize(seed_track), measures)
        return self.__unnormalize(generated)

    @staticmethod
    def __normalize(track: Track) -> Track:
        events = []
        for start, sound in track:
            #TODO: For now just normalize everything to 16th notes
            sixteenth_notes = sound.duration * 4 // track.ticks_per_beat
            if sixteenth_notes == 0:
                sixteenth_notes = 1
            sound = Sound(volume=sound.volume, note=sound.note, duration=sixteenth_notes)
            events.append((start * 4 // track.ticks_per_beat, sound))
        return Track(events, ticks_per_beat=4)

    @staticmethod
    def __unnormalize(track: Track) -> Track:
        events = []
        for start, sound in track:
            ticks = sound.duration * track.ticks_per_beat // 4
            if ticks == 0:
                ticks = 1
            sound = Sound(volume=sound.volume, note=sound.note, duration=ticks)
            events.append((start * track.ticks_per_beat // 4, sound))
        return Track(events, ticks_per_beat=4)
