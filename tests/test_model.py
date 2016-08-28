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

from deepstep.midi import Sound, Track
from deepstep.model import DNN


class TestModel(unittest.TestCase):
    def test_model(self) -> None:
        sound = Sound(volume=50, note=65, duration=1)

        main_score = []
        for start in range(0, 100, 2):
            main_score.append((start, sound))

        hyperparameters = Hyperparameters([10, 4], epochs=2, look_back=10)
        model = DNN(hyperparameters, notes={65}, look_back=10, sound_volume=50)
        tracks = [Track([], ticks_per_beat=4),
                  Track([(0, sound)], ticks_per_beat=4),
                  Track(main_score, ticks_per_beat=4)]
        model.train(tracks, epochs=2)
        model.evaluate([Track(main_score, ticks_per_beat=4)])

        generated = model.generate(Track(main_score[:20], ticks_per_beat=4), 25)
        self.assertEqual(generated.duration, 100)
        found_note = False
        for start, sound in generated:
            self.assertEqual(sound.duration, 1)
            self.assertEqual(sound.volume, 50)
            self.assertEqual(sound.note, 65)
            found_note = True
        self.assertTrue(found_note)
