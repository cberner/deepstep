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

import argparse
import os
import os.path
from typing import List, Callable, Set

import numpy as np

from hyperflow import Hyperparameters, HyperparameterSpace, RandomWalk

from deepstep.midi import Sound, midi_to_score
from deepstep.model import Model


def loss(training_scores: List[List[Sound]],
         validation_scores: List[List[Sound]],
         hyperparameters: Hyperparameters,
         notes: Set[int],
         volume: int,
         duration: int) -> float:
    model = Model(hyperparameters, notes, hyperparameters.look_back, sound_volume=volume, sound_duration=duration)
    model.train(training_scores, hyperparameters.epochs)
    return model.evaluate(validation_scores)

def create_objective(training_scores: List[List[Sound]],
                     validation_scores: List[List[Sound]],
                     notes: Set[int],
                     volume: int,
                     duration: int) -> Callable[[Hyperparameters], float]:
    return lambda parameters: loss(training_scores, validation_scores, parameters, notes, volume, duration)

def main() -> None:
    parser = argparse.ArgumentParser(description="DNN to generate music")
    parser.add_argument('files', type=str, help="Directory of training data")
    parser.add_argument('--validation_percent', type=int, default=10, help="Percentage of data to use for validation")
    parser.add_argument('--budget', type=int, default=60, help="Time budget in seconds")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Verbosity")

    args = parser.parse_args()

    expanded_name = os.path.expanduser(args.files)
    paths = []
    assert os.path.isdir(expanded_name)
    for filename in os.listdir(expanded_name):
        paths.append(os.path.join(expanded_name, filename))

    all_notes = set() # type: set[int]
    sounds = set() # type: Set[Sound]
    scores = []
    for path in paths:
        score = midi_to_score(path, verbose=(args.verbose > 1))
        scores.append(score)
        sounds = sounds.union(set(score))
        for sound in score:
            all_notes = all_notes.union(set(sound.notes))

    sound_volume = np.median([sound.volume for sound in sounds if sound.volume])
    # Treat all notes as the same duration
    sound_duration = np.median([sound.duration for sound in sounds if not sound.is_rest()])

    split = len(scores) // 10
    validation_scores = scores[:split]
    training_scores = scores[split:]

    space = HyperparameterSpace([1, 1, 1, 1], [500, 500, 500, 500], 1, 50, 1, 20)

    optimizer = RandomWalk(space)
    objective = create_objective(training_scores,
                                 validation_scores,
                                 all_notes,
                                 sound_volume,
                                 sound_duration)
    best_loss, best_parameters = optimizer.minimize(objective, args.budget)
    print("Loss: " + str(best_loss))
    print("Parameters: " + str(best_parameters))

if __name__ == '__main__':
    main()
