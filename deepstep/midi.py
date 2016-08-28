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
from typing import Union, Iterator
# Suppression needed until PyLint 2.0 is out
from typing import MutableMapping # pylint: disable=unused-import
from typing import List, Any, Sequence, Tuple

from collections import deque

import collections
from bisect import bisect_left, bisect_right
from mido import tempo2bpm
from mido import MetaMessage
from mido import bpm2tempo
from mido import Message, MidiFile, MidiTrack


class Sound:
    def __init__(self, volume: int, note: int, duration: int) -> None:
        assert duration > 0
        assert volume >= 0
        assert isinstance(duration, int), "Expected int, got {}".format(type(duration))
        assert isinstance(volume, int), "Expected int, got {}".format(type(volume))
        assert isinstance(note, int)
        self.__volume = volume
        self.__note = note
        self.__duration = duration

    @property
    def volume(self) -> int:
        return self.__volume

    @property
    def note(self) -> int:
        return self.__note

    @property
    def duration(self) -> int:
        return self.__duration

    def __repr__(self) -> str:
        return "volume={},notes={},duration={}".format(self.volume, self.note, self.duration)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Sound):
            return False
        return (self.volume == other.volume and
                self.note == other.note and self.duration == other.duration)

    def __hash__(self) -> int:
        return hash((self.volume, self.note, self.duration))


class Track(collections.Iterable):
    def __init__(self, sounds: Sequence[Tuple[int, Sound]], ticks_per_beat: int, duration: int=None) -> None:
        self.__sounds = sorted(sounds, key=lambda x: x[0])
        self.__starts = [start for start, sound in self.__sounds]
        self.__max_sound_duration = max([sound.duration for start, sound in sounds], default=0)
        if duration is None:
            self.__duration = max([start + sound.duration for start, sound in sounds], default=0)
        else:
            self.__duration = duration
        self.__ticks_per_beat = ticks_per_beat

    def __getitem__(self, key: Union[int, slice]) -> Sequence[Tuple[int, Sound]]:
        if isinstance(key, int):
            if key >= self.__duration:
                raise IndexError("Index out of bounds")
            lower_index = key
            upper_index = key + 1
        elif isinstance(key, slice):
            if key.step:
                raise ValueError("Slicing with step size not supported")
            lower_index = key.start or 0
            upper_index = key.stop or self.__duration
            if upper_index > self.__duration:
                upper_index = self.__duration
        else:
            raise TypeError("Invalid argument type")
        lower_bound = bisect_right(self.__starts, lower_index - self.__max_sound_duration)
        upper_bound = bisect_left(self.__starts, upper_index)
        result = []
        for start, sound in self.__sounds[lower_bound:upper_bound]:
            if start + sound.duration - 1 >= lower_index:
                result.append((start, sound))
        return result

    @property
    def ticks_per_beat(self) -> int:
        return self.__ticks_per_beat

    @property
    def duration(self) -> int:
        return self.__duration

    def __iter__(self) -> Iterator[Tuple[int, Sound]]:
        return iter(self.__sounds)


class TrackMetadata:
    def __init__(self, instrument: str, notes: int) -> None:
        self.__instrument = instrument
        self.__notes = notes

    @property
    def instrument(self) -> str:
        return self.__instrument

    @property
    def notes(self) -> int:
        return self.__notes


def midi_to_track(filename: str, verbose: bool=False) -> Track:
    with MidiFile(filename) as midi_file:
        if verbose:
            print("MIDI:")
            print("Tracks: ", len(midi_file.tracks))
            track_lengths = []
            for track in midi_file.tracks:
                count = 0
                for message in track:
                    if isinstance(message, Message) and message.type == 'note_on':
                        count += 1
                track_lengths.append(count)
            print("Notes per track: ", track_lengths)

        # TODO: For now only use the first channel
        note_start_time = {} # type: MutableMapping[int, int]
        note_volume = {} # type: MutableMapping[int, int]
        sounds = [] # type: List[Tuple[int, Sound]]
        now = 0
        for message in midi_file.tracks[0]:
            now += message.time
            if isinstance(message, MetaMessage):
                if message.type == 'set_tempo':
                    pass
                elif verbose:
                    print(message)
            elif isinstance(message, Message):
                if message.type == 'note_on':
                    if message.note in note_start_time:
                        start = note_start_time[message.note]
                        volume = note_volume[message.note]
                        del note_start_time[message.note]
                        del note_volume[message.note]
                        duration = now - start
                        if duration > 0:
                            sounds.append((start, Sound(volume=volume, note=message.note, duration=duration)))
                    # Some midi files use 'note_on', velocity=0 instead of 'note_off'
                    if message.velocity > 0:
                        note_start_time[message.note] = now
                        note_volume[message.note] = message.velocity
                elif message.type == 'note_off':
                    if message.note in note_start_time:
                        start = note_start_time[message.note]
                        volume = note_volume[message.note]
                        del note_start_time[message.note]
                        del note_volume[message.note]
                        duration = now - start
                        if duration > 0:
                            sounds.append((start, Sound(volume=volume, note=message.note, duration=duration)))
                    elif verbose > 1:
                        print("Note ({}) not in progress at time {}".format(message.note, message.time))
                elif verbose:
                    print(message)
            else:
                raise TypeError("Unknown message type: " + str(type(message)))

        # Process notes that weren't explicitly ended
        max_duration = max([sound.duration for _, sound in sounds], default=0)
        for note, start in note_start_time.items():
            volume = note_volume[note]
            duration = now - start
            # Skip anything longer than the longest note
            if duration > max_duration or duration == 0:
                continue
            sounds.append((start, Sound(volume=volume, note=note, duration=duration)))

        return Track(sounds, midi_file.ticks_per_beat)


def bpm_of_midi(filename: str) -> int:
    with MidiFile(filename) as midi_file:
        for track in midi_file.tracks:
            for note in track:
                if note.type == 'set_tempo':
                    return round(tempo2bpm(note.tempo))

    return 120


def write_track_as_midi(track: Track, bpm: int, filename: str) -> None:
    with MidiFile() as midi_file:
        midi_file.ticks_per_beat = track.ticks_per_beat
        midi_track = MidiTrack()
        midi_file.tracks.append(midi_track)
        now = 0
        ending_times = deque(sorted([(start + sound.duration, sound) for start, sound in track], key=lambda x: x[0]))
        midi_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))
        # TODO: Support other instrument types. Channel 10 (9 in Mido) forces playback as drumset
        for start, starting_sound in track:
            while ending_times[0][0] <= start:
                end, ending_sound = ending_times.popleft()
                midi_track.append(Message('note_off', note=ending_sound.note, time=end - now, channel=9))
                now = end
            midi_track.append(Message('note_on',
                                      note=starting_sound.note,
                                      time=start - now,
                                      velocity=starting_sound.volume,
                                      channel=9))
            now = start
        while ending_times:
            end, ending_sound = ending_times.popleft()
            midi_track.append(Message('note_off', note=ending_sound.note, time=end - now, channel=9))
            now = end
        midi_file.save(filename)


def midi_to_metadata(filename: str) -> List[TrackMetadata]:
    with MidiFile(filename) as midi_file:
        scores = []
        for track in midi_file.tracks:
            instruments = {} # type: MutableMapping[str, int]
            for message in track:
                if message.type == 'note_on' and message.velocity > 0:
                    instrument = 'Other'
                    # TODO: Handle other types of instruments
                    if message.channel == 9:
                        instrument = 'Drumset'
                    instruments[instrument] = 1 + instruments.get(instrument, 0)
            for instrument, notes in instruments.items():
                scores.append(TrackMetadata(instrument, notes))
        return scores
