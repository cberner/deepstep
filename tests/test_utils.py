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

from deepstep.midi import Sound
from deepstep.midi import midi_to_score, bpm_of_midi, write_score_as_midi, midi_to_metadata

class TestMidiUtils(unittest.TestCase):
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
        score = [Sound(volume=50, notes=[65], duration=1.0)]
        read_score = None
        try:
            write_score_as_midi(score, 110, midi_file.name)
            midi_file.flush()
            read_score = midi_to_score(midi_file.name)
        finally:
            midi_file.close()
        self.assertEqual(read_score, score)
