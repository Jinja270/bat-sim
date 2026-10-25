from __future__ import annotations
import os
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np
INPUT_DIM: int = 5
HIDDEN_DIM_1: int = 32
HIDDEN_DIM_2: int = 32
OUTPUT_DIM: int = 2
W1_SHAPE: Tuple[int, int] = (INPUT_DIM, HIDDEN_DIM_1)
W1_SIZE: int = INPUT_DIM * HIDDEN_DIM_1
B1_SHAPE: Tuple[int] = (HIDDEN_DIM_1,)
B1_SIZE: int = HIDDEN_DIM_1
W2_SHAPE: Tuple[int, int] = (HIDDEN_DIM_1, HIDDEN_DIM_2)
W2_SIZE: int = HIDDEN_DIM_1 * HIDDEN_DIM_2
B2_SHAPE: Tuple[int] = (HIDDEN_DIM_2,)
B2_SIZE: int = HIDDEN_DIM_2
W3_SHAPE: Tuple[int, int] = (HIDDEN_DIM_2, OUTPUT_DIM)
W3_SIZE: int = HIDDEN_DIM_2 * OUTPUT_DIM
B3_SHAPE: Tuple[int] = (OUTPUT_DIM,)
B3_SIZE: int = OUTPUT_DIM
TOTAL_WEIGHTS: int = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE + W3_SIZE + B3_SIZE

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))

class BatNeuralNet:
    INPUT_DIM: int = INPUT_DIM
    HIDDEN_DIM_1: int = HIDDEN_DIM_1
    HIDDEN_DIM_2: int = HIDDEN_DIM_2
    OUTPUT_DIM: int = OUTPUT_DIM
    TOTAL_WEIGHTS: int = TOTAL_WEIGHTS

    def __init__(self, weights: Optional[Union[np.ndarray, Sequence[float]]]=None, rng: Optional[np.random.Generator]=None) -> None:
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self._weights: np.ndarray = np.empty(self.TOTAL_WEIGHTS, dtype=np.float64)
        self._bind_views()
        if weights is not None:
            self.set_flat_weights(weights)
        else:
            self._init_weights()

    def _bind_views(self) -> None:
        idx = 0
        self.w1: np.ndarray = self._weights[idx:idx + W1_SIZE].reshape(W1_SHAPE)
        idx += W1_SIZE
        self.b1: np.ndarray = self._weights[idx:idx + B1_SIZE].reshape(B1_SHAPE)
        idx += B1_SIZE
        self.w2: np.ndarray = self._weights[idx:idx + W2_SIZE].reshape(W2_SHAPE)
        idx += W2_SIZE
        self.b2: np.ndarray = self._weights[idx:idx + B2_SIZE].reshape(B2_SHAPE)
        idx += B2_SIZE
        self.w3: np.ndarray = self._weights[idx:idx + W3_SIZE].reshape(W3_SHAPE)
        idx += W3_SIZE
        self.b3: np.ndarray = self._weights[idx:idx + B3_SIZE].reshape(B3_SHAPE)
        idx += B3_SIZE

    def _init_weights(self) -> None:
        self.w1[:] = self._rng.normal(0.0, 0.02, size=W1_SHAPE)
        self.b1[:] = self._rng.normal(0.0, 0.01, size=B1_SHAPE)
        self.w2[:] = self._rng.normal(0.0, 0.02, size=W2_SHAPE)
        self.b2[:] = self._rng.normal(0.0, 0.01, size=B2_SHAPE)
        self.w3[:] = self._rng.normal(0.0, 0.02, size=W3_SHAPE)
        self.b3[:] = self._rng.normal(0.0, 0.01, size=B3_SHAPE)
        w1 = self.w1
        b1 = self.b1
        w2 = self.w2
        w3 = self.w3
        w1[0, 0] += -5.0
        b1[0] += 2.0
        w1[1, 1] += -5.0
        b1[1] += 2.0
        w1[2, 2] += -5.0
        b1[2] += 2.25
        w1[2, 3] += -3.0
        w1[0, 3] += -3.0
        w1[1, 3] += 3.0
        b1[3] += 2.0
        w1[2, 4] += -3.0
        w1[1, 4] += -3.0
        w1[0, 4] += 3.0
        b1[4] += 2.0
        w1[4, 5] += 3.0
        w1[2, 5] += -3.0
        b1[5] += 2.0
        for i in range(32):
            w2[i, i] += 1.0
        w3[0, 0] += 2.0
        w3[1, 0] += -2.0
        w3[2, 0] += 1.5
        w3[3, 0] += 2.5
        w3[4, 0] += -2.5
        w3[2, 1] += 2.0
        w3[5, 1] += 2.0
        w3[0, 1] += 1.5
        w3[1, 1] += 1.5

    def forward(self, inputs: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim == 1:
            if x.shape[0] != self.INPUT_DIM:
                raise ValueError(f'Expected 1D input array with shape ({self.INPUT_DIM},), got shape {x.shape}')
        elif x.ndim == 2:
            if x.shape[1] != self.INPUT_DIM:
                raise ValueError(f'Expected 2D input array with shape (N, {self.INPUT_DIM}), got shape {x.shape}')
        else:
            raise ValueError(f'Expected 1D or 2D array, got {x.ndim}D array with shape {x.shape}')
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        out = h2 @ self.w3 + self.b3
        res = np.empty_like(out)
        res[..., 0] = np.tanh(out[..., 0])
        z_safe = np.clip(out[..., 1], -60.0, 60.0)
        res[..., 1] = 1.0 / (1.0 + np.exp(-z_safe))
        if x.ndim == 1:
            self.last_inputs = x
            self.last_h1 = h1
            self.last_h2 = h2
            self.last_outputs = res
        return res

    def get_last_activations(self) -> Dict[str, np.ndarray]:
        return {'inputs': getattr(self, 'last_inputs', np.zeros(self.INPUT_DIM, dtype=np.float64)), 'h1': getattr(self, 'last_h1', np.zeros(self.HIDDEN_DIM_1, dtype=np.float64)), 'h2': getattr(self, 'last_h2', np.zeros(self.HIDDEN_DIM_2, dtype=np.float64)), 'outputs': getattr(self, 'last_outputs', np.zeros(self.OUTPUT_DIM, dtype=np.float64))}

    def __call__(self, inputs: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        return self.forward(inputs)

    def get_flat_weights(self) -> np.ndarray:
        return self._weights.copy()

    def set_flat_weights(self, flat_weights: Union[np.ndarray, Sequence[float]]) -> None:
        arr = np.asarray(flat_weights, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self.TOTAL_WEIGHTS:
            raise ValueError(f'Expected flat weights array with shape ({self.TOTAL_WEIGHTS},), got shape {arr.shape}')
        np.copyto(self._weights, arr)

    @property
    def flat_weights(self) -> np.ndarray:
        return self.get_flat_weights()

    @flat_weights.setter
    def flat_weights(self, val: Union[np.ndarray, Sequence[float]]) -> None:
        self.set_flat_weights(val)

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        path_str = str(filepath)
        dirname = os.path.dirname(os.path.abspath(path_str))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(path_str, weights=self._weights)

    @classmethod
    def load(cls, filepath: Union[str, os.PathLike], rng: Optional[np.random.Generator]=None) -> BatNeuralNet:
        path_str = str(filepath)
        if not os.path.exists(path_str) and (not path_str.endswith('.npz')):
            if os.path.exists(path_str + '.npz'):
                path_str = path_str + '.npz'
        if not os.path.exists(path_str):
            raise FileNotFoundError(f'Weight archive not found: {path_str}')
        with np.load(path_str) as data:
            if 'weights' in data:
                weights = np.array(data['weights'], dtype=np.float64)
            elif 'flat_weights' in data:
                weights = np.array(data['flat_weights'], dtype=np.float64)
            elif 'arr_0' in data:
                weights = np.array(data['arr_0'], dtype=np.float64)
            elif all((k in data for k in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3'))):
                w1 = np.asarray(data['w1'], dtype=np.float64).ravel()
                b1 = np.asarray(data['b1'], dtype=np.float64).ravel()
                w2 = np.asarray(data['w2'], dtype=np.float64).ravel()
                b2 = np.asarray(data['b2'], dtype=np.float64).ravel()
                w3 = np.asarray(data['w3'], dtype=np.float64).ravel()
                b3 = np.asarray(data['b3'], dtype=np.float64).ravel()
                weights = np.concatenate([w1, b1, w2, b2, w3, b3])
            else:
                raise KeyError(f"No recognized weight keys ('weights', 'flat_weights', 'arr_0') found in {path_str}. Keys present: {list(data.keys())}")
        return cls(weights=weights, rng=rng)

    @staticmethod
    def uniform_crossover(parent_a: BatNeuralNet, parent_b: BatNeuralNet, rng: Optional[np.random.Generator]=None) -> BatNeuralNet:
        w_a = parent_a.get_flat_weights()
        w_b = parent_b.get_flat_weights()
        if w_a.shape != (BatNeuralNet.TOTAL_WEIGHTS,) or w_b.shape != (BatNeuralNet.TOTAL_WEIGHTS,):
            raise ValueError(f'Parent weight shapes must match ({BatNeuralNet.TOTAL_WEIGHTS},), got {w_a.shape} and {w_b.shape}')
        r = rng if rng is not None else getattr(parent_a, '_rng', np.random.default_rng())
        mask = r.random(BatNeuralNet.TOTAL_WEIGHTS) < 0.5
        child_weights = np.where(mask, w_a, w_b)
        return BatNeuralNet(weights=child_weights, rng=rng)

    def mutate(self, mutation_rate: float=0.1, sigma: float=0.1, rng: Optional[np.random.Generator]=None) -> None:
        if mutation_rate <= 0.0 or sigma <= 0.0:
            return
        r = rng if rng is not None else self._rng
        if mutation_rate >= 1.0:
            self._weights += r.normal(0.0, sigma, size=self.TOTAL_WEIGHTS)
        else:
            mask = r.random(self.TOTAL_WEIGHTS) < mutation_rate
            count = int(np.count_nonzero(mask))
            if count > 0:
                self._weights[mask] += r.normal(0.0, sigma, size=count)

    def copy(self) -> BatNeuralNet:
        return BatNeuralNet(weights=self.get_flat_weights(), rng=self._rng)

    def __copy__(self) -> BatNeuralNet:
        return self.copy()

    def __deepcopy__(self, memo: Any) -> BatNeuralNet:
        return self.copy()

    def __len__(self) -> int:
        return self.TOTAL_WEIGHTS

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BatNeuralNet):
            return NotImplemented
        return bool(np.array_equal(self._weights, other._weights))

    def __repr__(self) -> str:
        return f'BatNeuralNet(inputs={self.INPUT_DIM}, hidden=({self.HIDDEN_DIM_1}, {self.HIDDEN_DIM_2}), outputs={self.OUTPUT_DIM}, total_weights={self.TOTAL_WEIGHTS})'

def compute_mutation_sigma(generation: int, sigma_0: float=0.1, decay_factor: float=0.98, min_sigma: float=0.01) -> float:
    if generation < 0:
        raise ValueError(f'Generation must be non-negative, got {generation}')
    if sigma_0 <= 0.0:
        raise ValueError(f'sigma_0 must be strictly positive, got {sigma_0}')
    if not 0.0 < decay_factor <= 1.0:
        raise ValueError(f'decay_factor must be in (0, 1], got {decay_factor}')
    if min_sigma < 0.0:
        raise ValueError(f'min_sigma must be non-negative, got {min_sigma}')
    decayed = sigma_0 * decay_factor ** generation
    return float(max(min_sigma, decayed))

def breed_population_vectorized(current_population_weights: np.ndarray, fitnesses: np.ndarray, generation: int, pop_size: int=100, n_elites: int=2, truncation_fraction: float=0.2, mutation_rate: float=0.1, sigma_0: float=0.1, decay_factor: float=0.98, min_sigma: float=0.01, rng: Optional[np.random.Generator]=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    ranked_indices = np.argsort(fitnesses)[::-1]
    sorted_weights = current_population_weights[ranked_indices]
    elites = np.copy(sorted_weights[:n_elites])
    n_parents = max(1, int(pop_size * truncation_fraction))
    parent_pool = sorted_weights[:n_parents]
    n_offspring = pop_size - n_elites
    n_weights = current_population_weights.shape[1]
    idx_a = rng.integers(0, n_parents, size=n_offspring)
    idx_b = rng.integers(0, n_parents, size=n_offspring)
    cross_mask = rng.random((n_offspring, n_weights)) < 0.5
    offspring = np.where(cross_mask, parent_pool[idx_a], parent_pool[idx_b])
    sigma = compute_mutation_sigma(generation, sigma_0, decay_factor, min_sigma)
    mut_mask = rng.random((n_offspring, n_weights)) < mutation_rate
    noise = rng.normal(0.0, sigma, size=(n_offspring, n_weights))
    offspring += mut_mask * noise
    return np.vstack([elites, offspring])
