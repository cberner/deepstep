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

from deepstep.utils import bpm_of_midi

class TestMidiUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.midi_file = tempfile.NamedTemporaryFile(suffix='.midi')
        self.midi_file.write(pkgutil.get_data('test_resources', 'test_score.midi'))
        self.midi_file.flush()

    def tearDown(self) -> None:
        self.midi_file.close()

    def test_bpm_parsing(self) -> None:
        self.assertEqual(bpm_of_midi(self.midi_file.name), 140)
