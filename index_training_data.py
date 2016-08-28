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
import os
import os.path
import shutil

from deepstep.midi import midi_to_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Search midi files for good training data")
    parser.add_argument('files', type=str, help="Directory of training data")
    parser.add_argument('-o', '--output', type=str, default='', help="Copy all 'good' files here")
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Verbosity")

    args = parser.parse_args()

    expanded_name = os.path.expanduser(args.files)
    paths = []
    assert os.path.isdir(expanded_name)
    for dirname, _, filenames in os.walk(expanded_name):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.mid' or ext == '.midi':
                paths.append(os.path.join(dirname, filename))

    drums_only = 0
    one_other = 0
    more_than_one = 0
    too_few_notes = 0
    for path in paths:
        metadata = midi_to_metadata(path)
        all_drums = True
        notes = 0
        for score in metadata:
            if score.instrument != 'Drumset':
                all_drums = False
            notes += score.notes
        if all_drums and notes > 20:
            drums_only += 1
            if args.output:
                shutil.copy(path, os.path.expanduser(args.output))
        elif notes <= 20:
            too_few_notes += 1
        elif len(metadata) == 1:
            one_other += 1
        else:
            more_than_one += 1

    print("Drums only: {}, One other: {}, More than one other: {}, Too few notes: {}".format(
        drums_only, one_other, more_than_one, too_few_notes))


if __name__ == '__main__':
    main()
