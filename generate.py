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
from typing import Set

import os
import os.path

import numpy as np

from hyperflow import Hyperparameters, NeuralLayer

from deepstep.midi import midi_to_track, bpm_of_midi, write_track_as_midi
from deepstep.model import NormalizedTime, DNN


def main() -> None:
    parser = argparse.ArgumentParser(description="DNN to generate music")
    parser.add_argument('training_files', type=str, help="File or directory of training data")
    parser.add_argument('seed_file', type=str, help="File to use as seed data for generation")
    parser.add_argument('--epochs', type=int, default=1, help="Training epochs")
    parser.add_argument('--output_file', type=str, default="out.midi", help="Output file")
    parser.add_argument('--measures', type=int, default=100, help="Measures of output to generate")
    parser.add_argument('--look_back', type=int, default=20, help="Look back distance during training")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Verbosity")

    args = parser.parse_args()
    hyperparameters = Hyperparameters([NeuralLayer.lstm(250),
                                       NeuralLayer.lstm(100),
                                       NeuralLayer.dense(50),
                                       NeuralLayer.dense(25)],
                                      epochs=args.epochs, look_back=args.look_back)

    expanded_name = os.path.expanduser(args.training_files)
    paths = []
    if os.path.isdir(expanded_name):
        for filename in os.listdir(expanded_name):
            paths.append(os.path.join(expanded_name, filename))
    else:
        paths.append(expanded_name)

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

    split = len(tracks) // 2
    validation_scores = tracks[:split]
    training_scores = tracks[split:]
    model = NormalizedTime(DNN(hyperparameters, all_notes, args.look_back, sound_volume=sound_volume))
    model.train(training_scores, args.epochs)
    print("Validation loss: " + str(model.evaluate(validation_scores)))

    # re-train on all scores
    model = NormalizedTime(DNN(hyperparameters, all_notes, args.look_back, sound_volume=sound_volume))
    model.train(tracks, args.epochs)

    seed_score = midi_to_track(os.path.expanduser(args.seed_file), verbose=(args.verbose > 0))
    bpm = bpm_of_midi(os.path.expanduser(args.seed_file))
    write_track_as_midi(model.generate(seed_score, args.measures), bpm, args.output_file)
    write_track_as_midi(seed_score, bpm, "diagnostic.midi")

if __name__ == '__main__':
    main()
