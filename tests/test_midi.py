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

import unittest
import tempfile
import pkgutil

from deepstep.midi import Sound, Track
from deepstep.midi import midi_to_track, bpm_of_midi, write_track_as_midi, midi_to_metadata

class TestMidi(unittest.TestCase):
    def setUp(self) -> None:
        self.midi_file = tempfile.NamedTemporaryFile(suffix='.midi')
        self.midi_file.write(pkgutil.get_data('test_resources', 'test_score.midi'))
        self.midi_file.flush()

    def tearDown(self) -> None:
        self.midi_file.close()

    def test_bpm_parsing(self) -> None:
        self.assertEqual(bpm_of_midi(self.midi_file.name), 140)

    def test_metadata_parsing(self) -> None:
        metadata = midi_to_metadata(self.midi_file.name)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0].instrument, 'Drumset')
        self.assertEqual(metadata[0].notes, 2)

    def test_read_write(self) -> None:
        midi_file = tempfile.NamedTemporaryFile(suffix='.midi')
        track = Track([(0, Sound(volume=50, note=65, duration=1))], ticks_per_beat=1)
        try:
            write_track_as_midi(track, 110, midi_file.name)
            midi_file.flush()
            read_track = midi_to_track(midi_file.name)
        finally:
            midi_file.close()
        self.assertEqual(read_track.duration, track.duration)
        self.assertEqual(read_track.ticks_per_beat, track.ticks_per_beat)
        self.assertEqual(list(read_track), list(track))


class TestTrack(unittest.TestCase):
    def test_indexing(self) -> None:
        first = Sound(volume=1, note=65, duration=10)
        second = Sound(volume=1, note=65, duration=5)
        track = Track([(10, first), (11, second)], ticks_per_beat=1)
        self.assertEqual(track[0], [])
        self.assertEqual(track[9], [])
        self.assertEqual(track[10], [(10, first)])
        self.assertEqual(track[11], [(10, first), (11, second)])
        self.assertEqual(track[15], [(10, first), (11, second)])
        self.assertEqual(track[16], [(10, first)])
        self.assertEqual(track[19], [(10, first)])
        try:
            track[20] # pylint: disable=pointless-statement
            self.fail("Expected exception")
        except IndexError:
            # expected
            pass

    def test_slicing(self) -> None:
        first = Sound(volume=1, note=65, duration=10)
        second = Sound(volume=1, note=65, duration=5)
        track = Track([(10, first), (11, second)], ticks_per_beat=1)
        self.assertEqual(track[0:10], [])
        self.assertEqual(track[0:11], [(10, first)])
        self.assertEqual(track[9:11], [(10, first)])
        self.assertEqual(track[9:12], [(10, first), (11, second)])
        self.assertEqual(track[11:16], [(10, first), (11, second)])
        self.assertEqual(track[15:16], [(10, first), (11, second)])
        self.assertEqual(track[:], [(10, first), (11, second)])
        self.assertEqual(track[8:18], [(10, first), (11, second)])
        self.assertEqual(track[16:20], [(10, first)])
        self.assertEqual(track[16:21], [(10, first)])
        self.assertEqual(track[19:21], [(10, first)])

    def test_duration(self) -> None:
        sounds = [(10, Sound(volume=1, note=65, duration=10)), (11, Sound(volume=1, note=65, duration=5))]
        track = Track(sounds, ticks_per_beat=1)
        self.assertEqual(track.duration, 20)
