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

from music21 import converter
from music21.midi.translate import streamToMidiFile
from music21.tempo import MetronomeMark
from music21.stream import Stream
from music21.note import Rest, Note
from music21.chord import Chord

from deepstep.sound import Sound


def midi_to_score(filename: str, verbose: bool=False) -> List[Sound]:
    midi = converter.parse(filename)
    if verbose:
        print("MIDI:")
        print("Channels: ", len(midi))
        print("Notes per channel: ", [len(channel.flat) for channel in midi])
    score = []
    # TODO: For now only use the first channel
    for note in midi[0].flat:
        if verbose and not (isinstance(note, Note) or isinstance(note, Rest) or
                            isinstance(note, Chord) or isinstance(note, MetronomeMark)):
            print(note)
            continue
        sound = Sound.from_midi_note(note)
        if sound.duration == 0.0:
            continue
        score.append(sound)

    return score


def bpm_of_midi(filename: str) -> int:
    midi = converter.parse(filename)
    for track in midi:
        for note in track.flat:
            if isinstance(note, MetronomeMark):
                return note.number

    return 120


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
