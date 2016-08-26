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

import numpy as np

from hyperflow import Hyperparameters

# Suppression needed until PyLint 2.0 is out
from deepstep.midi import Sound # pylint: disable=unused-import
from deepstep.midi import midi_to_score, bpm_of_midi, write_score_as_midi
from deepstep.model import Model


def main() -> None:
    parser = argparse.ArgumentParser(description="DNN to generate music")
    parser.add_argument('training_files', type=str, help="File or directory of training data")
    parser.add_argument('seed_file', type=str, help="File to use as seed data for generation")
    parser.add_argument('--epochs', type=int, default=1, help="Training epochs")
    parser.add_argument('--output_file', type=str, default="out.midi", help="Output file")
    parser.add_argument('--measures', type=int, default=20, help="Measures of output to generate")
    parser.add_argument('--look_back', type=int, default=20, help="Look back distance during training")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Verbosity")

    args = parser.parse_args()
    hyperparameters = Hyperparameters([250, 100, 50, 25], epochs=args.epochs, look_back=args.look_back)

    expanded_name = os.path.expanduser(args.training_files)
    paths = []
    if os.path.isdir(expanded_name):
        for filename in os.listdir(expanded_name):
            paths.append(os.path.join(expanded_name, filename))
    else:
        paths.append(expanded_name)

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
    model = Model(hyperparameters, all_notes, args.look_back, sound_volume=sound_volume, sound_duration=sound_duration)
    model.train(training_scores, args.epochs)
    print("Validation loss: " + str(model.evaluate(validation_scores)))

    # re-train on all scores
    model = Model(hyperparameters, all_notes, args.look_back, sound_volume=sound_volume, sound_duration=sound_duration)
    model.train(scores, args.epochs)

    seed_score = midi_to_score(os.path.expanduser(args.seed_file), verbose=(args.verbose > 0))
    bpm = bpm_of_midi(os.path.expanduser(args.seed_file))
    write_score_as_midi(model.generate(seed_score, args.measures), bpm, args.output_file)
    write_score_as_midi(seed_score, bpm, "diagnostic.midi")

if __name__ == '__main__':
    main()
