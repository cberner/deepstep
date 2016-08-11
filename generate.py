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
from typing import List, Tuple

from music21 import converter
from music21.midi.translate import streamToMidiFile
from music21.tempo import MetronomeMark
from music21.stream import Stream
from music21.note import Rest, Note
from music21.chord import Chord

from deepstep.sound import Sound
from deepstep.model import Model


def midi_to_score(filename: str, verbose: bool=False) -> Tuple[int, List[Sound]]:
    midi = converter.parse(filename)
    if verbose:
        print("MIDI:")
        print("Channels: ", len(midi))
        print("Notes per channel: ", [len(channel.flat) for channel in midi])
    score = []
    bpm = 140
    # TODO: For now only use the first channel
    for note in midi[0].flat:
        if verbose and not (isinstance(note, Note) or isinstance(note, Rest) or
                            isinstance(note, Chord) or isinstance(note, MetronomeMark)):
            print(note)
            continue
        if isinstance(note, MetronomeMark):
            bpm = note.number
            continue
        sound = Sound.from_midi_note(note)
        if sound.duration == 0.0:
            continue
        score.append(sound)

    return (bpm, score)


def write_score_as_midi(score: List[Sound], bpm: int, filename: str) -> None:
    midi_stream = Stream()
    offset = 0.0
    for sound in score:
        midi_stream.insert(offset + sound.duration, sound.to_midi_note())
        offset += sound.duration

    midi_stream.insert(0, MetronomeMark(number=bpm))
    midi_file = streamToMidiFile(midi_stream)
    # TODO: Support other instrument types. Channel 10 forces playback as drumset
    midi_file.tracks[0].setChannel(10)
    midi_file.open(filename, 'wb')
    midi_file.write()
    midi_file.close()


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

    expanded_name = os.path.expanduser(args.training_files)
    paths = []
    if os.path.isdir(expanded_name):
        for filename in os.listdir(expanded_name):
            paths.append(os.path.join(expanded_name, filename))
    else:
        paths.append(expanded_name)

    all_notes = set() # type: set[int]
    scores = []
    for path in paths:
        _, score = midi_to_score(path, verbose=(args.verbose > 1))
        scores.append(score)
        for sound in score:
            all_notes = all_notes.union(set(sound.notes))

    model = Model(all_notes, args.look_back)
    model.train(scores, args.epochs)

    bpm, seed_score = midi_to_score(os.path.expanduser(args.seed_file), verbose=(args.verbose > 0))
    write_score_as_midi(model.generate(seed_score, args.measures), bpm, args.output_file)
    write_score_as_midi(seed_score, bpm, "diagnostic.midi")

if __name__ == '__main__':
    main()
