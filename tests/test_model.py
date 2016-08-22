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

from hyperflow import Hyperparameters

from deepstep.sound import Sound
from deepstep.model import expand_rest_notes, Model


class TestModel(unittest.TestCase):
    def test_expand_rest_notes(self) -> None:
        score = [Sound(volume=50, notes=[65], duration=1.0),
                 Sound(volume=0, notes=[], duration=2.0),
                 Sound(volume=50, notes=[65], duration=0.5)]
        expected = [Sound(volume=50, notes=[65], duration=1.0),
                    Sound(volume=0, notes=[], duration=1.0),
                    Sound(volume=0, notes=[], duration=1.0),
                    Sound(volume=50, notes=[65], duration=0.5)]
        self.assertEqual(expand_rest_notes(score, 1.0), expected)

    def test_model(self) -> None:
        sound = Sound(volume=50, notes=[65], duration=1.0)
        rest = Sound(volume=0, notes=[], duration=1.0)

        main_score = [sound, rest]
        for _ in range(99):
            main_score.append(sound)
            main_score.append(rest)

        hyperparameters = Hyperparameters([10, 4], epochs=2, look_back=10)
        model = Model(hyperparameters, notes=set([65]), look_back=10, sound_volume=50, sound_duration=1.0)
        model.train([[], [sound, rest], main_score], epochs=2)
        model.evaluate([main_score])

        generated = model.generate(main_score[:20], 25)
        self.assertEqual(len(generated), 100)
        found_note = False
        found_rest = False
        for sound in generated:
            self.assertEqual(sound.duration, 1.0)
            if sound.notes:
                self.assertEqual(sound.volume, 50)
                self.assertEqual(len(sound.notes), 1)
                self.assertEqual(sound.notes[0], 65)
                found_note = True
            else:
                self.assertEqual(sound.volume, 0)
                found_rest = True
        self.assertTrue(found_rest)
        self.assertTrue(found_note)
