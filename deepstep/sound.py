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

from music21.note import Rest, Note
from music21.chord import Chord


class Sound:
    def __init__(self, volume: int, notes: List[str], duration) -> None:
        self.__volume = volume
        self.__notes = tuple(notes)
        self.__duration = float(duration)

    def is_rest(self):
        return not self.notes

    @property
    def volume(self):
        return self.__volume

    @property
    def notes(self):
        return self.__notes

    @property
    def duration(self):
        return self.__duration

    def to_midi_note(self):
        if self.is_rest():
            return Rest(quarterLength=self.duration)
        if len(self.notes) == 1:
            note = Note(self.notes[0], quarterLength=self.duration)
            note.volume = self.volume
            return note
        else:
            chord = Chord(self.notes, quarterLength=self.duration)
            chord.volume = self.volume
            return chord

    @staticmethod
    def from_midi_note(note):
        notes = []
        volume = None
        if isinstance(note, Note):
            notes.append(note.nameWithOctave)
            volume = note.volume.velocity
        elif isinstance(note, Chord):
            notes = [pitch.nameWithOctave for pitch in note.pitches]
            volume = note.volume.velocity
        return Sound(volume, notes, note.quarterLength)

    def __repr__(self):
        return "volume={},notes={},duration={}".format(self.volume, self.notes, self.duration)

    def __eq__(self, other):
        if not isinstance(other, Sound):
            return False
        return (self.volume == other.volume and
                self.notes == other.notes and self.duration == other.duration)

    def __hash__(self):
        return hash((self.volume, self.notes, self.duration))
