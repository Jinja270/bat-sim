from __future__ import annotations
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

INPUT_DIM: int = 9
HIDDEN_DIM_1: int = 32
HIDDEN_DIM_2: int = 32
OUTPUT_DIM: int = 3

W1_SHAPE: Tuple[int, int] = (INPUT_DIM, HIDDEN_DIM_1)
W1_SIZE: int = INPUT_DIM * HIDDEN_DIM_1

W_REC_SHAPE: Tuple[int, int] = (HIDDEN_DIM_1, HIDDEN_DIM_1)
W_REC_SIZE: int = HIDDEN_DIM_1 * HIDDEN_DIM_1

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

W_SKIP_SHAPE: Tuple[int, int] = (INPUT_DIM, OUTPUT_DIM)
W_SKIP_SIZE: int = INPUT_DIM * OUTPUT_DIM

TOTAL_WEIGHTS: int = (
    W1_SIZE + W_REC_SIZE + B1_SIZE + W2_SIZE + B2_SIZE + W3_SIZE + B3_SIZE + W_SKIP_SIZE
)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def leaky_relu(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * x)

def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))

class BatNeuralNet:
    INPUT_DIM: int = INPUT_DIM
    HIDDEN_DIM_1: int = HIDDEN_DIM_1
    HIDDEN_DIM_2: int = HIDDEN_DIM_2
    OUTPUT_DIM: int = OUTPUT_DIM
    TOTAL_WEIGHTS: int = TOTAL_WEIGHTS

    def __init__(self, weights: Optional[Union[np.ndarray, Sequence[float]]] = None, rng: Optional[np.random.Generator] = None) -> None:
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self._weights: np.ndarray = np.empty(self.TOTAL_WEIGHTS, dtype=np.float64)
        self.h_state: np.ndarray = np.zeros(self.HIDDEN_DIM_1, dtype=np.float64)
        self._bind_views()
        if weights is not None:
            self.set_flat_weights(weights)
        else:
            self._init_weights()

    def reset_state(self) -> None:
        self.h_state.fill(0.0)

    def _bind_views(self) -> None:
        idx = 0
        self.w1: np.ndarray = self._weights[idx:idx + W1_SIZE].reshape(W1_SHAPE)
        idx += W1_SIZE
        self.w_rec: np.ndarray = self._weights[idx:idx + W_REC_SIZE].reshape(W_REC_SHAPE)
        idx += W_REC_SIZE
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
        self.w_skip: np.ndarray = self._weights[idx:idx + W_SKIP_SIZE].reshape(W_SKIP_SHAPE)
        idx += W_SKIP_SIZE

    def _init_weights(self) -> None:
        self.w1[:] = self._rng.normal(0.0, 0.02, size=W1_SHAPE)
        self.w_rec[:] = self._rng.normal(0.0, 0.02, size=W_REC_SHAPE)
        self.b1[:] = self._rng.normal(0.0, 0.01, size=B1_SHAPE)
        self.w2[:] = self._rng.normal(0.0, 0.02, size=W2_SHAPE)
        self.b2[:] = self._rng.normal(0.0, 0.01, size=B2_SHAPE)
        self.w3[:] = self._rng.normal(0.0, 0.02, size=W3_SHAPE)
        self.b3[:] = self._rng.normal(0.0, 0.01, size=B3_SHAPE)
        self.w_skip[:] = self._rng.normal(0.0, 0.01, size=W_SKIP_SHAPE)

        # Recurrent persistence: slight self-loop inertia
        for i in range(HIDDEN_DIM_1):
            self.w_rec[i, i] += 0.35
            self.w2[i, i] += 1.0

        # Direct Spinal Reflex Skip Connections:
        # Reflex 1: Goal bearing tracking - decisive heading authority toward acoustic beacon
        self.w_skip[7, 0] = 1.4

        # Reflex 2: Throttle & Braking (Output 2)
        # Open front corridor (r3 high) -> sprint; blocked front (r3 low) -> brake
        self.w_skip[3, 2] += 2.8  # Center ray directly excites throttle
        self.w_skip[2, 2] += 0.8  # Forward-left open helps maintain speed
        self.w_skip[4, 2] += 0.8  # Forward-right open helps maintain speed

        # Reflex 3: Urgency / Pulse Frequency (Output 1)
        self.w_skip[3, 1] -= 2.2
        self.b3[1] += 1.8

        # Cortical Hidden Layer feature wiring:
        # Non-linear proximity reflexes: strictly activate when wall is close (<40px)
        # Neuron 0: Left proximity reflex -> steer right
        self.w1[0, 0] -= 2.5
        self.w1[1, 0] -= 3.5
        self.w1[2, 0] -= 5.0
        self.b1[0] += 1.5
        self.w3[0, 0] += 2.5

        # Neuron 1: Right proximity reflex -> steer left
        self.w1[4, 1] -= 5.0
        self.w1[5, 1] -= 3.5
        self.w1[6, 1] -= 2.5
        self.b1[1] += 1.5
        self.w3[1, 0] -= 2.5

        # Neurons 4 & 5: Cortical beacon neurons
        # Neuron 4: Goal bearing right
        self.w1[7, 4] += 3.5
        self.w3[4, 0] += 1.2

        # Neuron 5: Goal bearing left
        self.w1[7, 5] -= 3.5
        self.w3[5, 0] -= 1.2

    def init_scratch(self, rng: Optional[np.random.Generator] = None) -> None:
        r = rng if rng is not None else self._rng
        self._weights[:] = r.normal(0.0, 0.08, size=self.TOTAL_WEIGHTS)
        self.reset_state()

    def forward(self, inputs: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        x_in = np.asarray(inputs, dtype=np.float64)
        if x_in.ndim == 1:
            if x_in.shape[0] == 5:
                # Backward compatibility with 5-element input
                x = np.zeros(self.INPUT_DIM, dtype=np.float64)
                x[0] = x_in[0]
                x[1] = x_in[0]
                x[2] = x_in[0]
                x[3] = x_in[2]
                x[4] = x_in[1]
                x[5] = x_in[1]
                x[6] = x_in[1]
                x[7] = x_in[3]
                x[8] = x_in[4]
            elif x_in.shape[0] != self.INPUT_DIM:
                raise ValueError(f'Expected 1D input array with shape ({self.INPUT_DIM},) or (5,), got shape {x_in.shape}')
            else:
                x = x_in

            pre_h1 = x @ self.w1 + self.h_state @ self.w_rec + self.b1
            h1 = np.where(pre_h1 > 0.0, pre_h1, 0.05 * pre_h1)
            self.h_state = h1.copy()

            pre_h2 = h1 @ self.w2 + self.b2
            h2 = np.where(pre_h2 > 0.0, pre_h2, 0.05 * pre_h2)

            out = h2 @ self.w3 + self.b3 + x @ self.w_skip
            res = np.empty(self.OUTPUT_DIM, dtype=np.float64)
            res[0] = np.tanh(out[0])
            res[1] = safe_sigmoid(out[1])
            res[2] = safe_sigmoid(out[2])

            self.last_inputs = x
            self.last_h1 = h1
            self.last_h2 = h2
            self.last_outputs = res
            return res
        elif x_in.ndim == 2:
            if x_in.shape[1] == 5:
                x = np.zeros((x_in.shape[0], self.INPUT_DIM), dtype=np.float64)
                x[:, 0] = x_in[:, 0]
                x[:, 1] = x_in[:, 0]
                x[:, 2] = x_in[:, 0]
                x[:, 3] = x_in[:, 2]
                x[:, 4] = x_in[:, 1]
                x[:, 5] = x_in[:, 1]
                x[:, 6] = x_in[:, 1]
                x[:, 7] = x_in[:, 3]
                x[:, 8] = x_in[:, 4]
            elif x_in.shape[1] != self.INPUT_DIM:
                raise ValueError(f'Expected 2D input array with shape (N, {self.INPUT_DIM}), got shape {x_in.shape}')
            else:
                x = x_in

            pre_h1 = x @ self.w1 + self.h_state @ self.w_rec + self.b1
            h1 = np.where(pre_h1 > 0.0, pre_h1, 0.05 * pre_h1)
            if x.shape[0] == 1:
                self.h_state = h1[0].copy()

            pre_h2 = h1 @ self.w2 + self.b2
            h2 = np.where(pre_h2 > 0.0, pre_h2, 0.05 * pre_h2)

            out = h2 @ self.w3 + self.b3 + x @ self.w_skip
            res = np.empty_like(out)
            res[..., 0] = np.tanh(out[..., 0])
            res[..., 1] = safe_sigmoid(out[..., 1])
            res[..., 2] = safe_sigmoid(out[..., 2])
            return res
        else:
            raise ValueError(f'Expected 1D or 2D array, got {x_in.ndim}D array with shape {x_in.shape}')

    def get_last_activations(self) -> Dict[str, np.ndarray]:
        return {
            'inputs': getattr(self, 'last_inputs', np.zeros(self.INPUT_DIM, dtype=np.float64)),
            'h1': getattr(self, 'last_h1', np.zeros(self.HIDDEN_DIM_1, dtype=np.float64)),
            'h2': getattr(self, 'last_h2', np.zeros(self.HIDDEN_DIM_2, dtype=np.float64)),
            'outputs': getattr(self, 'last_outputs', np.zeros(self.OUTPUT_DIM, dtype=np.float64)),
        }

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
    def load(cls, filepath: Union[str, os.PathLike], rng: Optional[np.random.Generator] = None) -> BatNeuralNet:
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
        if weights.shape[0] == cls.TOTAL_WEIGHTS:
            return cls(weights=weights, rng=rng)
        else:
            net = cls(rng=rng)
            return net

    @staticmethod
    def uniform_crossover(parent_a: BatNeuralNet, parent_b: BatNeuralNet, rng: Optional[np.random.Generator] = None) -> BatNeuralNet:
        w_a = parent_a.get_flat_weights()
        w_b = parent_b.get_flat_weights()
        if w_a.shape != (BatNeuralNet.TOTAL_WEIGHTS,) or w_b.shape != (BatNeuralNet.TOTAL_WEIGHTS,):
            raise ValueError(f'Parent weight shapes must match ({BatNeuralNet.TOTAL_WEIGHTS},), got {w_a.shape} and {w_b.shape}')
        r = rng if rng is not None else getattr(parent_a, '_rng', np.random.default_rng())
        mask = r.random(BatNeuralNet.TOTAL_WEIGHTS) < 0.5
        child_weights = np.where(mask, w_a, w_b)
        return BatNeuralNet(weights=child_weights, rng=rng)

    def mutate(self, mutation_rate: float = 0.1, sigma: float = 0.1, rng: Optional[np.random.Generator] = None) -> None:
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
        c = BatNeuralNet(weights=self.get_flat_weights(), rng=self._rng)
        c.h_state = self.h_state.copy()
        return c

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

def compute_mutation_sigma(generation: int, sigma_0: float = 0.1, decay_factor: float = 0.98, min_sigma: float = 0.01) -> float:
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

def breed_population_vectorized(current_population_weights: np.ndarray, fitnesses: np.ndarray, generation: int, pop_size: int = 100, n_elites: int = 2, truncation_fraction: float = 0.2, mutation_rate: float = 0.1, sigma_0: float = 0.1, decay_factor: float = 0.98, min_sigma: float = 0.01, rng: Optional[np.random.Generator] = None) -> np.ndarray:
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
