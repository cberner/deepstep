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

import random
import time
import math

from typing import List, Callable, Any, Tuple


class Hyperparameters:
    def __init__(self, layers: List[int], epochs: int, look_back: int) -> None:
        self.__layers = layers
        self.__epochs = epochs
        self.__look_back = look_back

    @property
    def layers(self) -> List[int]:
        return self.__layers

    @property
    def epochs(self) -> int:
        return self.__epochs

    @property
    def look_back(self) -> int:
        return self.__look_back

    def __repr__(self) -> str:
        return "layers:{},epochs:{},look_back:{}".format(self.layers, self.epochs, self.look_back)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Hyperparameters):
            return False
        return self.layers == other.layers and self.epochs == other.epochs and self.look_back == other.look_back

    def __hash__(self) -> int:
        return hash((self.layers, self.epochs, self.look_back))


class HyperparameterSpace:
    def __init__(self,
                 layer_mins: List[int],
                 layer_maxs: List[int],
                 min_epochs: int,
                 max_epochs: int,
                 min_look_back: int,
                 max_look_back: int) -> None:
        assert len(layer_mins) == len(layer_maxs)
        self.__layer_mins = layer_mins
        self.__layer_maxs = layer_maxs
        self.__min_epochs = min_epochs
        self.__max_epochs = max_epochs
        self.__min_look_back = min_look_back
        self.__max_look_back = max_look_back

    def sample(self) -> Hyperparameters:
        layers = []
        for i in range(len(self.__layer_mins)):
            layers.append(random.randint(self.__layer_mins[i], self.__layer_maxs[i]))
        epochs = random.randint(self.__min_epochs, self.__max_epochs)
        look_back = random.randint(self.__min_look_back, self.__max_look_back)
        return Hyperparameters(layers, epochs=epochs, look_back=look_back)


class RandomWalk:
    def __init__(self, space: HyperparameterSpace) -> None:
        self.__space = space

    def minimize(self,
                 objective: Callable[[Hyperparameters], float],
                 budget_secs: int,
                 results: int=10) -> List[Tuple[float, Hyperparameters]]:
        ranked_results = [(math.inf, None)] # type: List[Tuple[float, Hyperparameters]]
        start_time = time.monotonic() # type: ignore. Mypy seems to be broken. It can't find "monotonic"
        try:
            while time.monotonic() - start_time < budget_secs: # type: ignore
                parameters = self.__space.sample()
                loss = objective(parameters)
                print("{} -> {} loss".format(parameters, loss))
                if loss < ranked_results[-1][0]:
                    ranked_results.append((loss, parameters))
                    ranked_results.sort(key=lambda x: x[0])
                    ranked_results = ranked_results[:results]
        except KeyboardInterrupt:
            # Stop optimizing and return
            print("Stopping optimization...")

        return [x for x in ranked_results if x[0] < math.inf]
