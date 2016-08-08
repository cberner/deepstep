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

from typing import List
from collections import deque
import random

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Reshape, Dropout

from deepstep.sound import Sound


def expand_rest_notes(score: List[Sound], duration):
    result = []
    for sound in score:
        if not sound.is_rest():
            result.append(sound)
            continue
        rest_duration = sound.duration
        while rest_duration > 0:
            result.append(Sound(volume=0, notes=[], duration=duration))
            rest_duration -= duration
    return tuple(result)


class Model:
    def __init__(self, look_back):
        self.look_back = look_back
        self.id_to_note = {}
        self.note_to_id = {}
        self.model = None
        self.sound_volume = 0
        self.sound_duration = 0

    def train(self, scores: List[List[Sound]], epochs: int):
        # save the mapping from ids to sounds to recover the sounds
        sounds = set() # type: Set[Sound]
        all_notes = set() # type: Set[int]
        for score in scores:
            sounds = sounds.union(set(score))
        for sound in sounds:
            all_notes = all_notes.union(set(sound.notes))
        for i, note in enumerate(sorted(all_notes)):
            self.note_to_id[note] = i
            self.id_to_note[i] = note

        self.sound_volume = np.median([sound.volume for sound in sounds if sound.volume])
        # Treat all notes as the same duration
        self.sound_duration = np.median([sound.duration for sound in sounds if not sound.is_rest()])

        expanded_scores = []
        for score in scores:
            expanded_scores.append(expand_rest_notes(score, self.sound_duration))

        num_ids = len(self.note_to_id)
        num_examples = 0
        for score in expanded_scores:
            num_examples += len(score) - self.look_back
        assert num_examples > 0

        examples = np.zeros((num_examples, self.look_back, num_ids), dtype=np.bool)
        targets = np.zeros((num_examples, num_ids), dtype=np.bool)
        example_id = 0
        for score in expanded_scores:
            for i in range(len(score) - self.look_back):
                # Copy the seed section of the score
                for j in range(self.look_back):
                    for note in score[i + j].notes:
                        examples[example_id, j, self.note_to_id[note]] = 1
                # Copy the expected next note of the score
                for note in score[i + self.look_back].notes:
                    targets[example_id, self.note_to_id[note]] = 1
                example_id += 1

        model = Sequential()
        model.add(Reshape((self.look_back * num_ids,), input_shape=(self.look_back, num_ids)))
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(num_ids, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.fit(examples, targets, nb_epoch=epochs)
        self.model = model

    def generate(self, seed_score: List[Sound], measures: int) -> List[Sound]:
        generated = []
        seed = deque() # type: deque[List[int]]
        for sound in seed_score[:self.look_back]:
            note_ids = []
            for note in sound.notes:
                if note not in self.note_to_id:
                    note_ids.append(random.choice(range(len(self.note_to_id))))
                else:
                    note_ids.append(self.note_to_id[note])
            seed.append(note_ids)

        # Treat measures as 4/4 time
        length = 0
        while length <= measures * 4:
            seed_ids = np.zeros((1, self.look_back, len(self.id_to_note)))
            for i, note_ids in enumerate(seed):
                for note_id in note_ids:
                    seed_ids[0, i, note_id] = 1

            predictions = self.model.predict(seed_ids, verbose=False)[0]
            predicted_note_ids = []
            for i, prediction in enumerate(predictions):
                if prediction > random.random():
                    predicted_note_ids.append(i)
            seed.popleft()
            seed.append(predicted_note_ids)

            notes = [self.id_to_note[note_id] for note_id in predicted_note_ids]
            sound = Sound(notes=notes,
                          volume=self.sound_volume if notes else 0,
                          duration=self.sound_duration)
            generated.append(sound)
            length += sound.duration

        return generated
