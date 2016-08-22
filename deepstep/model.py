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

from typing import List, Set, Tuple, Sequence
from collections import deque
import random

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Reshape, Dropout

from hyperflow import Hyperparameters

from deepstep.sound import Sound


def expand_rest_notes(score: Sequence[Sound], duration: float) -> List[Sound]:
    result = []
    for sound in score:
        if not sound.is_rest():
            result.append(sound)
            continue
        rest_duration = sound.duration
        while rest_duration > 0:
            result.append(Sound(volume=0, notes=[], duration=duration))
            rest_duration -= duration
    return result


class Model:
    def __init__(self,
                 hyperparameters: Hyperparameters,
                 notes: Set[int],
                 look_back: int,
                 sound_volume: int,
                 sound_duration: float) -> None:
        self.look_back = look_back
        self.sound_volume = sound_volume
        self.sound_duration = sound_duration
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

    def train(self, scores: List[List[Sound]], epochs: int) -> None:
        examples, labels = self.__scores_to_matrices(scores)
        self.model.fit(examples, labels, nb_epoch=epochs)

    def evaluate(self, scores: List[List[Sound]]) -> float:
        examples, labels = self.__scores_to_matrices(scores)
        return self.model.evaluate(examples, labels, verbose=False)

    def __scores_to_matrices(self, scores: Sequence[Sequence[Sound]]) -> Tuple[np.ndarray, np.ndarray]:
        expanded_scores = []
        for score in scores:
            expanded = expand_rest_notes(score, self.sound_duration)
            if len(expanded) > self.look_back:
                expanded_scores.append(expanded)

        num_ids = len(self.note_to_id)
        num_examples = 0
        for score in expanded_scores:
            num_examples += len(score) - self.look_back
        assert num_examples > 0

        examples = np.zeros((num_examples, self.look_back, num_ids), dtype=np.bool)
        labels = np.zeros((num_examples, num_ids), dtype=np.bool)
        example_num = 0
        for score in expanded_scores:
            for i in range(len(score) - self.look_back):
                # Copy the seed section of the score
                for j in range(self.look_back):
                    for note in score[i + j].notes:
                        examples[example_num, j, self.note_to_id[note]] = 1
                # Copy the expected next note of the score
                for note in score[i + self.look_back].notes:
                    labels[example_num, self.note_to_id[note]] = 1
                example_num += 1

        return (examples, labels)

    def generate(self, seed_score: List[Sound], measures: int) -> List[Sound]:
        generated = []
        seed = deque() # type: deque[Sound]
        for sound in seed_score[:self.look_back + 1]:
            notes = [] # type: List[int]
            for note in sound.notes:
                if note not in self.note_to_id:
                    notes.append(random.choice(list(self.note_to_id.keys())))
                else:
                    notes.append(note)
            seed.append(Sound(notes=notes, volume=sound.volume, duration=sound.duration))

        # Treat measures as 4/4 time
        length = 0.0
        while length <= measures * 4:
            seed_ids, _ = self.__scores_to_matrices([seed])

            predictions = self.model.predict(seed_ids, verbose=False)[0]
            predicted_note_ids = []
            for i, prediction in enumerate(predictions):
                if prediction > random.random():
                    predicted_note_ids.append(i)

            notes = [self.id_to_note[note_id] for note_id in predicted_note_ids]
            sound = Sound(notes=notes,
                          volume=self.sound_volume if notes else 0,
                          duration=self.sound_duration)
            seed.popleft()
            seed.append(sound)
            generated.append(sound)
            length += sound.duration

        return generated
