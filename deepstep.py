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
import random
import os
import os.path
from enum import Enum
from typing import List
from collections import deque

import numpy as np
from music21 import converter
from music21.midi.translate import streamToMidiFile
from music21.tempo import MetronomeMark
from music21.stream import Stream
from music21.note import Rest, Note
from music21.chord import Chord

from keras.models import Sequential
from keras.layers.core import Dense, Reshape, Dropout


class SoundType(Enum):
    REST = 1
    NOTE = 2

class Sound:
    def __init__(self, sound_type: SoundType, volume: int, notes: List[str], duration) -> None:
        self.__type = sound_type
        self.__volume = volume
        self.__notes = tuple(notes)
        self.__duration = float(duration)

    @staticmethod
    def rest(duration):
        return Sound(sound_type=SoundType.REST, volume=0, notes=[], duration=duration)

    @property
    def type(self):
        return self.__type

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
        if self.__type == SoundType.REST:
            return Rest(quarterLength=self.__duration)
        if len(self.__notes) == 1:
            note = Note(self.__notes[0], quarterLength=self.__duration)
            note.volume = self.__volume
            return note
        else:
            chord = Chord(self.__notes, quarterLength=self.__duration)
            chord.volume = self.__volume
            return chord

    @staticmethod
    def from_midi_note(note):
        sound_type = SoundType.REST if isinstance(note, Rest) else SoundType.NOTE
        notes = []
        volume = None
        if isinstance(note, Note):
            notes.append(note.nameWithOctave)
            volume = note.volume.velocity
        elif isinstance(note, Chord):
            notes = [pitch.nameWithOctave for pitch in note.pitches]
            volume = note.volume.velocity
        return Sound(sound_type, volume, notes, note.quarterLength)

    def __repr__(self):
        return "type={},volume={},notes={},duration={}".format(
            self.__type, self.__volume, self.__notes, self.__duration)

    def __eq__(self, other):
        if not isinstance(other, Sound):
            return False
        return (self.type == other.type and self.volume == other.volume and
                self.notes == other.notes and self.duration == other.duration)

    def __hash__(self):
        return hash((self.type, self.volume, self.notes, self.duration))


class Model:
    def __init__(self, look_back):
        self.look_back = look_back
        self.id_to_note = {}
        self.note_to_id = {}
        self.model = None
        self.sound_volume = 0
        self.sound_duration = 0

    @staticmethod
    def expand_rest_notes(score: List[Sound], duration):
        result = []
        for sound in score:
            if sound.type != SoundType.REST:
                result.append(sound)
                continue
            rest_duration = sound.duration
            while rest_duration > 0:
                result.append(Sound.rest(duration))
                rest_duration -= duration
        return tuple(result)

    def train(self, scores: List[List[Sound]], epochs: int):
        # save the mapping from ids to sounds to recover the sounds
        sounds = set() # type: Set[Sound]
        all_notes = set() # type: Set[str]
        for score in scores:
            sounds = sounds.union(set(score))
        for sound in sounds:
            all_notes = all_notes.union(set(sound.notes))
        for i, note in enumerate(all_notes):
            self.note_to_id[note] = i
            self.id_to_note[i] = note

        self.sound_volume = np.median([sound.volume for sound in sounds if sound.volume])
        # Treat all notes as the same duration
        self.sound_duration = np.median([sound.duration for sound in sounds if sound.type == SoundType.NOTE])

        expanded_scores = []
        for score in scores:
            expanded_scores.append(self.expand_rest_notes(score, self.sound_duration))

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
            sound = Sound(sound_type=SoundType.NOTE if notes else SoundType.REST,
                          notes=notes,
                          volume=self.sound_volume if notes else 0,
                          duration=self.sound_duration)
            generated.append(sound)
            length += sound.duration

        return generated


def midi_to_score(filename, verbose=False):
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


def write_score_as_midi(score: List[Sound], bpm: int, filename: str):
    midi_stream = Stream()
    offset = 0
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



def main():
    parser = argparse.ArgumentParser(description="DNN to generate music")
    parser.add_argument('training_files', type=str, help="File or directory of training data")
    parser.add_argument('seed_file', type=str, help="File to use as seed data for generation")
    parser.add_argument('--epochs', type=int, default=1, help="Training epochs")
    parser.add_argument('--output_file', type=str, default="out.midi", help="Output file")
    parser.add_argument('--measures', type=float, default=20, help="Measures of output to generate")
    parser.add_argument('--look_back', type=float, default=20, help="Look back distance during training")

    args = parser.parse_args()

    scores = []
    expanded_name = os.path.expanduser(args.training_files)
    if os.path.isdir(expanded_name):
        for filename in os.listdir(expanded_name):
            _, score = midi_to_score(os.path.join(expanded_name, filename))
            scores.append(score)
    else:
        _, score = midi_to_score(expanded_name)
        scores.append(score)

    model = Model(args.look_back)
    model.train(scores, args.epochs)

    bpm, seed_score = midi_to_score(os.path.expanduser(args.seed_file))
    write_score_as_midi(model.generate(seed_score, args.measures), bpm, args.output_file)
    write_score_as_midi(seed_score, bpm, "diagnostic.midi")

if __name__ == '__main__':
    main()
