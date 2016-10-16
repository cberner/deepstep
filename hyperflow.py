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
from enum import Enum


class NeuralLayerType(Enum):
    LSTM = 1
    DENSE = 2


class NeuralLayer:
    @staticmethod
    def dense(neurons: int) -> 'NeuralLayer':
        return NeuralLayer(NeuralLayerType.DENSE, neurons)

    @staticmethod
    def lstm(neurons: int) -> 'NeuralLayer':
        return NeuralLayer(NeuralLayerType.LSTM, neurons)

    def __init__(self, layer_type: NeuralLayerType, neurons: int) -> None:
        self.__layer_type = layer_type
        self.__neurons = neurons

    @property
    def layer_type(self) -> NeuralLayerType:
        return self.__layer_type

    @property
    def neurons(self) -> int:
        return self.__neurons

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NeuralLayer):
            return False
        return self.layer_type == other.layer_type and self.neurons == other.neurons

    def __hash__(self) -> int:
        return hash((self.layer_type, self.neurons))


class Hyperparameters:
    def __init__(self, layers: List[NeuralLayer], epochs: int, look_back: int) -> None:
        self.__layers = layers
        self.__epochs = epochs
        self.__look_back = look_back

    @property
    def layers(self) -> List[NeuralLayer]:
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
                 lstm_layer_mins: List[int],
                 lstm_layer_maxs: List[int],
                 dense_layer_mins: List[int],
                 dense_layer_maxs: List[int],
                 min_epochs: int,
                 max_epochs: int,
                 min_look_back: int,
                 max_look_back: int) -> None:
        assert len(lstm_layer_mins) == len(lstm_layer_maxs)
        assert len(dense_layer_mins) == len(dense_layer_maxs)
        self.__lstm_layer_mins = lstm_layer_mins
        self.__lstm_layer_maxs = lstm_layer_maxs
        self.__dense_layer_mins = dense_layer_mins
        self.__dense_layer_maxs = dense_layer_maxs
        self.__min_epochs = min_epochs
        self.__max_epochs = max_epochs
        self.__min_look_back = min_look_back
        self.__max_look_back = max_look_back

    def sample(self) -> Hyperparameters:
        layers = []
        for i in range(len(self.__lstm_layer_mins)):
            layers.append(NeuralLayer.lstm(random.randint(self.__lstm_layer_mins[i], self.__lstm_layer_maxs[i])))
        for i in range(len(self.__dense_layer_mins)):
            layers.append(NeuralLayer.dense(random.randint(self.__dense_layer_mins[i], self.__dense_layer_maxs[i])))
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
