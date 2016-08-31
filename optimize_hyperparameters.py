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

from deepstep.midi import midi_to_track, Track
from deepstep.model import NormalizedTime, DNN


def model_loss(training_tracks: List[Track],
               validation_tracks: List[Track],
               hyperparameters: Hyperparameters,
               notes: Set[int],
               volume: int) -> float:
    model = NormalizedTime(DNN(hyperparameters, notes, hyperparameters.look_back, sound_volume=volume))
    model.train(training_tracks, hyperparameters.epochs)
    return model.evaluate(validation_tracks)

def create_objective(training_tracks: List[Track],
                     validation_tracks: List[Track],
                     notes: Set[int],
                     volume: int) -> Callable[[Hyperparameters], float]:
    return lambda parameters: model_loss(training_tracks, validation_tracks, parameters, notes, volume)

def main() -> None:
    parser = argparse.ArgumentParser(description="DNN to generate music")
    parser.add_argument('files', type=str, help="Directory of training data")
    parser.add_argument('--validation_percent', type=int, default=50, help="Percentage of data to use for validation")
    parser.add_argument('--budget', type=int, default=60, help="Time budget in seconds")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Verbosity")

    args = parser.parse_args()

    expanded_name = os.path.expanduser(args.files)
    paths = []
    assert os.path.isdir(expanded_name)
    for filename in os.listdir(expanded_name):
        paths.append(os.path.join(expanded_name, filename))

    all_notes = set() # type: Set[int]
    tracks = []
    volumes = []
    durations = []
    for path in paths:
        track = midi_to_track(path, verbose=(args.verbose > 1))
        tracks.append(track)
        for _, sound in track:
            all_notes.add(sound.note)
            volumes.append(sound.volume)
            durations.append(sound.duration)

    sound_volume = int(np.median(volumes))

    split = len(tracks) * args.validation_percent // 100
    validation_scores = tracks[:split]
    training_scores = tracks[split:]

    space = HyperparameterSpace([1, 1, 1, 1], [500, 500, 500, 500], 1, 20, 1, 20)

    optimizer = RandomWalk(space)
    objective = create_objective(training_scores,
                                 validation_scores,
                                 all_notes,
                                 sound_volume)
    for loss, parameters in optimizer.minimize(objective, args.budget):
        print("Loss: " + str(loss))
        print("Parameters: " + str(parameters))
        print()

if __name__ == '__main__':
    main()
