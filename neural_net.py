"""
bat_sim/neural_net.py
======================
Pure NumPy Feedforward Neural Network for Neuroevolution of Bat Drones.

Architecture:
- Input layer: 5 units (Left ray, Right ray, Front ray, Speed, Doppler velocity)
- Hidden layer 1: 32 units with ReLU activation
- Hidden layer 2: 32 units with ReLU activation
- Output layer: 2 units:
  - Output 0: Steering rate Δθ in [-1.0, 1.0] (tanh activation)
  - Output 1: Buzz urgency λ in [0.0, 1.0] (numerically safe sigmoid)

Total parameter count: exactly 1,314 flat float64 weights and biases.
Memory architecture: contiguous 1D float64 buffer with persistent zero-copy matrix views.
Supports dual-shape forward inference: single drone (5,) -> (2,) and batch population (N, 5) -> (N, 2).
Provides lossless flat weight serialization, uniform crossover, Gaussian mutation, and .npz persistence.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np


# ==============================================================================
# ARCHITECTURAL CONSTANTS
# ==============================================================================

INPUT_DIM: int = 5
"""Number of sensor inputs into the network."""

HIDDEN_DIM_1: int = 32
"""Number of units in the first hidden layer."""

HIDDEN_DIM_2: int = 32
"""Number of units in the second hidden layer."""

OUTPUT_DIM: int = 2
"""Number of motor and behavioral control outputs."""

W1_SHAPE: Tuple[int, int] = (INPUT_DIM, HIDDEN_DIM_1)      # (5, 32)
W1_SIZE: int = INPUT_DIM * HIDDEN_DIM_1                    # 160

B1_SHAPE: Tuple[int] = (HIDDEN_DIM_1,)                     # (32,)
B1_SIZE: int = HIDDEN_DIM_1                                # 32

W2_SHAPE: Tuple[int, int] = (HIDDEN_DIM_1, HIDDEN_DIM_2)   # (32, 32)
W2_SIZE: int = HIDDEN_DIM_1 * HIDDEN_DIM_2                # 1024

B2_SHAPE: Tuple[int] = (HIDDEN_DIM_2,)                     # (32,)
B2_SIZE: int = HIDDEN_DIM_2                                # 32

W3_SHAPE: Tuple[int, int] = (HIDDEN_DIM_2, OUTPUT_DIM)     # (32, 2)
W3_SIZE: int = HIDDEN_DIM_2 * OUTPUT_DIM                   # 64

B3_SHAPE: Tuple[int] = (OUTPUT_DIM,)                       # (2,)
B3_SIZE: int = OUTPUT_DIM                                  # 2

TOTAL_WEIGHTS: int = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE + W3_SIZE + B3_SIZE
"""Exact total number of flattened float64 weights and biases (1,314)."""


# ==============================================================================
# ACTIVATION FUNCTIONS WITH NUMERICAL OVERFLOW PROTECTION
# ==============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation function max(0, x)."""
    return np.maximum(0.0, x)


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable logistic sigmoid function with overflow protection.
    Clamps inputs to [-60.0, 60.0] to prevent exp overflow warnings in float64.
    """
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


# ==============================================================================
# BAT NEURAL NETWORK CLASS
# ==============================================================================

class BatNeuralNet:
    """
    Feedforward Neural Network controller for bat echolocation drones.

    Parameters
    ----------
    weights : Optional[Union[np.ndarray, Sequence[float]]], default=None
        1D float64 array of length 1,314 containing serialized network weights.
        If None, initializes weights using He Normal (hidden) and Xavier (output).
    rng : Optional[np.random.Generator], default=None
        NumPy random number generator for reproducible initialization and mutation.
    """

    INPUT_DIM: int = INPUT_DIM
    HIDDEN_DIM_1: int = HIDDEN_DIM_1
    HIDDEN_DIM_2: int = HIDDEN_DIM_2
    OUTPUT_DIM: int = OUTPUT_DIM
    TOTAL_WEIGHTS: int = TOTAL_WEIGHTS

    def __init__(
        self,
        weights: Optional[Union[np.ndarray, Sequence[float]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self._weights: np.ndarray = np.empty(self.TOTAL_WEIGHTS, dtype=np.float64)
        self._bind_views()

        if weights is not None:
            self.set_flat_weights(weights)
        else:
            self._init_weights()

    def _bind_views(self) -> None:
        """Bind reshaped matrix views to slices of the underlying contiguous buffer."""
        idx = 0
        self.w1: np.ndarray = self._weights[idx : idx + W1_SIZE].reshape(W1_SHAPE)
        idx += W1_SIZE
        self.b1: np.ndarray = self._weights[idx : idx + B1_SIZE].reshape(B1_SHAPE)
        idx += B1_SIZE
        self.w2: np.ndarray = self._weights[idx : idx + W2_SIZE].reshape(W2_SHAPE)
        idx += W2_SIZE
        self.b2: np.ndarray = self._weights[idx : idx + B2_SIZE].reshape(B2_SHAPE)
        idx += B2_SIZE
        self.w3: np.ndarray = self._weights[idx : idx + W3_SIZE].reshape(W3_SHAPE)
        idx += W3_SIZE
        self.b3: np.ndarray = self._weights[idx : idx + B3_SIZE].reshape(B3_SHAPE)
        idx += B3_SIZE

    def _init_weights(self) -> None:
        """
        Initialize layer weights with bio-acoustic reflex pathways and stochastic diversity.
        Provides innate bilateral acoustic steering sensitivity and proximity urgency
        modulation, perturbed by Gaussian variation for neuroevolutionary exploration.
        """
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

        # Unit 0: Left proximity -> steer right
        w1[0, 0] += -5.0; b1[0] += 2.0
        # Unit 1: Right proximity -> steer left
        w1[1, 1] += -5.0; b1[1] += 2.0
        # Unit 2: Front proximity -> evasion
        w1[2, 2] += -5.0; b1[2] += 2.25
        # Unit 3: Front & Left close -> steer right hard
        w1[2, 3] += -3.0; w1[0, 3] += -3.0; w1[1, 3] += 3.0; b1[3] += 2.0
        # Unit 4: Front & Right close -> steer left hard
        w1[2, 4] += -3.0; w1[1, 4] += -3.0; w1[0, 4] += 3.0; b1[4] += 2.0
        # Unit 5: Doppler closing velocity -> urgency
        w1[4, 5] += 3.0; w1[2, 5] += -3.0; b1[5] += 2.0

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
        """
        Execute forward inference pass supporting both single drone and batch inputs.

        Parameters
        ----------
        inputs : Union[np.ndarray, Sequence[float]]
            Sensor input array of shape (5,) for single drone or (N, 5) for batch population.

        Returns
        -------
        np.ndarray
            Output array of shape (2,) for single drone or (N, 2) for batch:
            - output[..., 0]: Steering rate Δθ in [-1.0, 1.0] (tanh).
            - output[..., 1]: Buzz urgency λ in [0.0, 1.0] (safe sigmoid).

        Raises
        ------
        ValueError
            If input array has incorrect dimensions or shape.
        """
        x = np.asarray(inputs, dtype=np.float64)

        if x.ndim == 1:
            if x.shape[0] != self.INPUT_DIM:
                raise ValueError(
                    f"Expected 1D input array with shape ({self.INPUT_DIM},), got shape {x.shape}"
                )
        elif x.ndim == 2:
            if x.shape[1] != self.INPUT_DIM:
                raise ValueError(
                    f"Expected 2D input array with shape (N, {self.INPUT_DIM}), got shape {x.shape}"
                )
        else:
            raise ValueError(
                f"Expected 1D or 2D array, got {x.ndim}D array with shape {x.shape}"
            )

        # Hidden layer 1 with ReLU
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)

        # Hidden layer 2 with ReLU
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)

        # Linear projection to output channels
        out = h2 @ self.w3 + self.b3

        # Assemble dual outputs
        res = np.empty_like(out)
        res[..., 0] = np.tanh(out[..., 0])
        z_safe = np.clip(out[..., 1], -60.0, 60.0)
        res[..., 1] = 1.0 / (1.0 + np.exp(-z_safe))

        # Cache last single-sample inference activations for live visualization
        if x.ndim == 1:
            self.last_inputs = x.copy()
            self.last_h1 = h1.copy()
            self.last_h2 = h2.copy()
            self.last_outputs = res.copy()

        return res

    def get_last_activations(self) -> Dict[str, np.ndarray]:
        """
        Retrieve cached activations from the most recent forward inference step.
        Used by real-time neural network visualizer to render live neuron pulses.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 'inputs', 'h1', 'h2', 'outputs' arrays.
        """
        return {
            "inputs": getattr(self, "last_inputs", np.zeros(self.INPUT_DIM, dtype=np.float64)),
            "h1": getattr(self, "last_h1", np.zeros(self.HIDDEN_DIM_1, dtype=np.float64)),
            "h2": getattr(self, "last_h2", np.zeros(self.HIDDEN_DIM_2, dtype=np.float64)),
            "outputs": getattr(self, "last_outputs", np.zeros(self.OUTPUT_DIM, dtype=np.float64)),
        }

    def __call__(self, inputs: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Callable shorthand for forward(inputs)."""
        return self.forward(inputs)

    def get_flat_weights(self) -> np.ndarray:
        """
        Return a copy of the 1D flattened weight vector (1,314 float64 elements).
        Defensive copy prevents accidental mutation of internal memory buffer.
        """
        return self._weights.copy()

    def set_flat_weights(self, flat_weights: Union[np.ndarray, Sequence[float]]) -> None:
        """
        Set internal network weights from a 1D array of 1,314 float64 elements.
        Copies data directly into the contiguous buffer, automatically updating views.

        Parameters
        ----------
        flat_weights : Union[np.ndarray, Sequence[float]]
            Array of shape (1314,).

        Raises
        ------
        ValueError
            If flat_weights length or shape is not (1314,).
        """
        arr = np.asarray(flat_weights, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self.TOTAL_WEIGHTS:
            raise ValueError(
                f"Expected flat weights array with shape ({self.TOTAL_WEIGHTS},), got shape {arr.shape}"
            )
        np.copyto(self._weights, arr)

    @property
    def flat_weights(self) -> np.ndarray:
        """Property alias for get_flat_weights()."""
        return self.get_flat_weights()

    @flat_weights.setter
    def flat_weights(self, val: Union[np.ndarray, Sequence[float]]) -> None:
        """Property setter alias for set_flat_weights()."""
        self.set_flat_weights(val)

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        """
        Save network weights to a compressed .npz archive.

        Parameters
        ----------
        filepath : Union[str, os.PathLike]
            Target file path (e.g. 'best_weights.npz').
        """
        path_str = str(filepath)
        dirname = os.path.dirname(os.path.abspath(path_str))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(path_str, weights=self._weights)

    @classmethod
    def load(
        cls,
        filepath: Union[str, os.PathLike],
        rng: Optional[np.random.Generator] = None,
    ) -> BatNeuralNet:
        """
        Load network weights from a .npz archive and return a reconstructed instance.
        Uses context-managed np.load to ensure file descriptor cleanup on Windows.

        Parameters
        ----------
        filepath : Union[str, os.PathLike]
            Source .npz archive path.
        rng : Optional[np.random.Generator], default=None
            NumPy random number generator for the loaded instance.

        Returns
        -------
        BatNeuralNet
            Reconstructed network with exact stored weights.
        """
        path_str = str(filepath)
        if not os.path.exists(path_str) and not path_str.endswith(".npz"):
            if os.path.exists(path_str + ".npz"):
                path_str = path_str + ".npz"
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Weight archive not found: {path_str}")

        with np.load(path_str) as data:
            if "weights" in data:
                weights = np.array(data["weights"], dtype=np.float64)
            elif "flat_weights" in data:
                weights = np.array(data["flat_weights"], dtype=np.float64)
            elif "arr_0" in data:
                weights = np.array(data["arr_0"], dtype=np.float64)
            elif all(k in data for k in ("w1", "b1", "w2", "b2", "w3", "b3")):
                w1 = np.asarray(data["w1"], dtype=np.float64).ravel()
                b1 = np.asarray(data["b1"], dtype=np.float64).ravel()
                w2 = np.asarray(data["w2"], dtype=np.float64).ravel()
                b2 = np.asarray(data["b2"], dtype=np.float64).ravel()
                w3 = np.asarray(data["w3"], dtype=np.float64).ravel()
                b3 = np.asarray(data["b3"], dtype=np.float64).ravel()
                weights = np.concatenate([w1, b1, w2, b2, w3, b3])
            else:
                raise KeyError(
                    f"No recognized weight keys ('weights', 'flat_weights', 'arr_0') found in {path_str}. Keys present: {list(data.keys())}"
                )

        return cls(weights=weights, rng=rng)

    @staticmethod
    def uniform_crossover(
        parent_a: BatNeuralNet,
        parent_b: BatNeuralNet,
        rng: Optional[np.random.Generator] = None,
    ) -> BatNeuralNet:
        """
        Create a child network via uniform crossover (50/50 gene splice) between two parents.

        Parameters
        ----------
        parent_a : BatNeuralNet
            First parent network.
        parent_b : BatNeuralNet
            Second parent network.
        rng : Optional[np.random.Generator], default=None
            Random number generator for reproducible crossover mask generation.

        Returns
        -------
        BatNeuralNet
            New offspring instance inheriting each gene from parent A or B with p = 0.5.
        """
        w_a = parent_a.get_flat_weights()
        w_b = parent_b.get_flat_weights()
        if w_a.shape != (BatNeuralNet.TOTAL_WEIGHTS,) or w_b.shape != (BatNeuralNet.TOTAL_WEIGHTS,):
            raise ValueError(
                f"Parent weight shapes must match ({BatNeuralNet.TOTAL_WEIGHTS},), got {w_a.shape} and {w_b.shape}"
            )

        r = rng if rng is not None else getattr(parent_a, "_rng", np.random.default_rng())
        mask = r.random(BatNeuralNet.TOTAL_WEIGHTS) < 0.5
        child_weights = np.where(mask, w_a, w_b)
        return BatNeuralNet(weights=child_weights, rng=rng)

    def mutate(
        self,
        mutation_rate: float = 0.1,
        sigma: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Apply in-place Gaussian perturbation to network weights.

        Parameters
        ----------
        mutation_rate : float, default=0.1
            Probability of mutating each individual weight gene in [0.0, 1.0].
        sigma : float, default=0.1
            Standard deviation of Gaussian noise added to selected weights.
        rng : Optional[np.random.Generator], default=None
            Random number generator for reproducible noise.
        """
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
        """
        Create an independent deep copy of this neural network with cloned weights.
        Guarantees zero memory sharing between instances.
        """
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
        return (
            f"BatNeuralNet(inputs={self.INPUT_DIM}, hidden=({self.HIDDEN_DIM_1}, {self.HIDDEN_DIM_2}), "
            f"outputs={self.OUTPUT_DIM}, total_weights={self.TOTAL_WEIGHTS})"
        )


# ==============================================================================
# MUTATION SCHEDULE & VECTORIZED BREEDING UTILITIES
# ==============================================================================

def compute_mutation_sigma(
    generation: int,
    sigma_0: float = 0.1,
    decay_factor: float = 0.98,
    min_sigma: float = 0.01,
) -> float:
    """
    Compute decaying Gaussian mutation standard deviation for a given generation.

    Formula: sigma(g) = max(min_sigma, sigma_0 * (decay_factor ** generation))

    Parameters
    ----------
    generation : int
        Current generation number (0-indexed, >= 0).
    sigma_0 : float, default=0.1
        Initial mutation standard deviation at generation 0.
    decay_factor : float, default=0.98
        Geometric decay factor per generation, in (0, 1].
    min_sigma : float, default=0.01
        Lower bound floor for mutation standard deviation.

    Returns
    -------
    float
        Computed mutation standard deviation for the generation.
    """
    if generation < 0:
        raise ValueError(f"Generation must be non-negative, got {generation}")
    if sigma_0 <= 0.0:
        raise ValueError(f"sigma_0 must be strictly positive, got {sigma_0}")
    if not (0.0 < decay_factor <= 1.0):
        raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}")
    if min_sigma < 0.0:
        raise ValueError(f"min_sigma must be non-negative, got {min_sigma}")

    decayed = sigma_0 * (decay_factor ** generation)
    return float(max(min_sigma, decayed))


def breed_population_vectorized(
    current_population_weights: np.ndarray,
    fitnesses: np.ndarray,
    generation: int,
    pop_size: int = 100,
    n_elites: int = 2,
    truncation_fraction: float = 0.20,
    mutation_rate: float = 0.1,
    sigma_0: float = 0.1,
    decay_factor: float = 0.98,
    min_sigma: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Breed the next generation of neural network weights using high-throughput vectorized operations.

    Parameters
    ----------
    current_population_weights : np.ndarray
        Array of shape (pop_size, 1314) containing flat weights of current population.
    fitnesses : np.ndarray
        Array of shape (pop_size,) containing scalar fitness scores.
    generation : int
        Current generation index (0-indexed).
    pop_size : int, default=100
        Total population size.
    n_elites : int, default=2
        Number of top elite individuals preserved without mutation.
    truncation_fraction : float, default=0.20
        Fraction of top individuals forming the parent breeding pool (top 20%).
    mutation_rate : float, default=0.1
        Probability of each gene mutating.
    sigma_0 : float, default=0.1
        Initial mutation standard deviation.
    decay_factor : float, default=0.98
        Decay factor for mutation schedule.
    min_sigma : float, default=0.01
        Floor for mutation standard deviation.
    rng : Optional[np.random.Generator], default=None
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        New population weights of shape (pop_size, 1314).
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Rank population descending by fitness
    ranked_indices = np.argsort(fitnesses)[::-1]
    sorted_weights = current_population_weights[ranked_indices]

    # 2. Extract elites (lossless deep copy)
    elites = np.copy(sorted_weights[:n_elites])

    # 3. Truncation selection: extract parent pool (top 20%)
    n_parents = max(1, int(pop_size * truncation_fraction))
    parent_pool = sorted_weights[:n_parents]

    # 4. Vectorized parent pair sampling for all offspring
    n_offspring = pop_size - n_elites
    n_weights = current_population_weights.shape[1]

    idx_a = rng.integers(0, n_parents, size=n_offspring)
    idx_b = rng.integers(0, n_parents, size=n_offspring)

    # 5. Vectorized uniform crossover (50/50 gene splice)
    cross_mask = rng.random((n_offspring, n_weights)) < 0.5
    offspring = np.where(cross_mask, parent_pool[idx_a], parent_pool[idx_b])

    # 6. Adaptive mutation schedule & vectorized Gaussian mutation
    sigma = compute_mutation_sigma(generation, sigma_0, decay_factor, min_sigma)
    mut_mask = rng.random((n_offspring, n_weights)) < mutation_rate
    noise = rng.normal(0.0, sigma, size=(n_offspring, n_weights))
    offspring += mut_mask * noise

    # 7. Stack elites and offspring into new population
    return np.vstack([elites, offspring])
