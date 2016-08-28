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
from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Sequence
import random

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Reshape, Dropout

from hyperflow import Hyperparameters

from deepstep.midi import Sound, Track


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
        model.add(Reshape((self.look_back * len(notes),), input_shape=(self.look_back, len(notes))))

        first_layer = True
        for neurons in hyperparameters.layers:
            if not first_layer:
                model.add(Dropout(0.2))
            model.add(Dense(neurons, activation='relu'))
            first_layer = False

        model.add(Dense(len(notes), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model = model

    def train(self, tracks: List[Track], epochs: int) -> None:
        examples, labels = self.__scores_to_matrices(tracks)
        self.model.fit(examples, labels, nb_epoch=epochs)

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
