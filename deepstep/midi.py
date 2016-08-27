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

from typing import List, SupportsFloat, Any, Sequence

from music21 import converter
from music21.midi import MidiFile
from music21.midi.translate import streamToMidiFile
from music21.tempo import MetronomeMark
from music21.stream import Stream
from music21.note import Rest, Note, GeneralNote
from music21.chord import Chord


class Sound:
    def __init__(self, volume: int, notes: List[int], duration: SupportsFloat) -> None:
        self.__volume = volume
        self.__notes = tuple(notes)
        self.__duration = float(duration)

    def is_rest(self) -> bool:
        return not self.notes

    @property
    def volume(self) -> int:
        return self.__volume

    @property
    def notes(self) -> Sequence[int]:
        return self.__notes

    @property
    def duration(self) -> float:
        return self.__duration

    def to_midi_note(self) -> GeneralNote:
        if self.is_rest():
            return Rest(quarterLength=self.duration)
        if len(self.__notes) == 1:
            note = Note(quarterLength=self.duration)
            note.pitch.midi = self.__notes[0]
            note.volume = self.volume
            return note
        else:
            chord = Chord(self.notes, quarterLength=self.duration)
            chord.volume = self.volume
            return chord

    @staticmethod
    def from_midi_note(note: GeneralNote) -> 'Sound':
        notes = []
        volume = None
        if isinstance(note, Note):
            notes.append(note.pitch.midi)
            volume = note.volume.velocity
        elif isinstance(note, Chord):
            notes = [pitch.midi for pitch in note.pitches]
            volume = note.volume.velocity
        return Sound(volume, notes, note.quarterLength)

    def __repr__(self) -> str:
        return "volume={},notes={},duration={}".format(self.volume, self.notes, self.duration)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Sound):
            return False
        return (self.volume == other.volume and
                self.notes == other.notes and self.duration == other.duration)

    def __hash__(self) -> int:
        return hash((self.volume, self.notes, self.duration))


class ScoreMetadata:
    def __init__(self, instrument: str, notes: int) -> None:
        self.__instrument = instrument
        self.__notes = notes

    @property
    def instrument(self) -> str:
        return self.__instrument

    @property
    def notes(self) -> int:
        return self.__notes


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
        midi_stream.insert(offset, sound.to_midi_note())
        offset += sound.duration

    midi_stream.insert(0, MetronomeMark(number=bpm))
    midi_file = streamToMidiFile(midi_stream)
    # TODO: Support other instrument types. Channel 10 forces playback as drumset
    midi_file.tracks[0].setChannel(10)
    midi_file.open(filename, 'wb')
    midi_file.write()
    midi_file.close()


def midi_to_metadata(filename: str) -> List[ScoreMetadata]:
    midi_file = MidiFile()
    midi_file.open(filename)
    midi_file.read()
    scores = []
    for track in midi_file.tracks:
        instruments = {} # type: dict[str, int]
        for note in track.events:
            if note.type == 'NOTE_ON' and note.velocity > 0:
                instrument = 'Other'
                # TODO: Handle other types of instruments
                if note.channel == 10:
                    instrument = 'Drumset'
                instruments[instrument] = 1 + instruments.get(instrument, 0)
        for instrument, notes in instruments.items():
            scores.append(ScoreMetadata(instrument, notes))
    midi_file.close()
    return scores