"""Shared setup helpers for building discrete optimal transport instances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GaussianSpec:
    """Parameters for a Gaussian point cloud in R^d."""

    mean: Array
    covariance: Array

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)

        if mean.ndim != 1:
            raise ValueError("Gaussian mean must be a 1D array.")
        if covariance.ndim != 2:
            raise ValueError("Gaussian covariance must be a 2D matrix.")
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Gaussian covariance must be square.")
        if covariance.shape[0] != mean.shape[0]:
            raise ValueError("Mean and covariance dimensions must match.")
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("Gaussian covariance must be symmetric.")

        eigenvalues = np.linalg.eigvalsh(covariance)
        if np.any(eigenvalues <= 0):
            raise ValueError("Gaussian covariance must be positive definite.")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @property
    def dimension(self) -> int:
        """Return the ambient dimension of the Gaussian."""

        return self.mean.shape[0]


@dataclass(frozen=True)
class OTInstance:
    """Container for a fully specified discrete OT problem instance."""

    source_points: Array
    target_points: Array
    source_weights: Array
    target_weights: Array
    cost_matrix: Array
    source_spec: GaussianSpec
    target_spec: GaussianSpec
    seed: int

    @property
    def n_source(self) -> int:
        """Number of source points."""

        return self.source_points.shape[0]

    @property
    def n_target(self) -> int:
        """Number of target points."""

        return self.target_points.shape[0]

    @property
    def dimension(self) -> int:
        """Ambient dimension shared by the source and target clouds."""

        return self.source_points.shape[1]


def uniform_weights(n: int) -> Array:
    """Return the uniform probability vector of length n."""

    if n <= 0:
        raise ValueError("n must be positive.")
    return np.full(n, 1.0 / n, dtype=float)


def sample_gaussian_points(
    n: int,
    gaussian: GaussianSpec,
    rng: np.random.Generator,
) -> Array:
    """Sample n points from the provided Gaussian specification."""

    if n <= 0:
        raise ValueError("n must be positive.")
    return np.asarray(
        rng.multivariate_normal(
            mean=gaussian.mean,
            cov=gaussian.covariance,
            size=n,
        ),
        dtype=float,
    )


def squared_euclidean_cost_matrix(source_points: Array, target_points: Array) -> Array:
    """Compute the squared Euclidean ground cost matrix."""

    source_points = np.asarray(source_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("Point clouds must be 2D arrays.")
    if source_points.shape[1] != target_points.shape[1]:
        raise ValueError("Source and target point clouds must have the same dimension.")

    differences = source_points[:, None, :] - target_points[None, :, :]
    return np.sum(differences * differences, axis=2)


def generate_ot_instance(
    n: int,
    source_spec: GaussianSpec,
    target_spec: GaussianSpec,
    seed: int,
) -> OTInstance:
    """Generate a reproducible discrete OT instance from two Gaussian clouds."""

    if source_spec.dimension != target_spec.dimension:
        raise ValueError("Source and target Gaussians must live in the same dimension.")

    rng = np.random.default_rng(seed)
    source_points = sample_gaussian_points(n=n, gaussian=source_spec, rng=rng)
    target_points = sample_gaussian_points(n=n, gaussian=target_spec, rng=rng)
    source_weights = uniform_weights(n)
    target_weights = uniform_weights(n)
    cost_matrix = squared_euclidean_cost_matrix(source_points, target_points)

    instance = OTInstance(
        source_points=source_points,
        target_points=target_points,
        source_weights=source_weights,
        target_weights=target_weights,
        cost_matrix=cost_matrix,
        source_spec=source_spec,
        target_spec=target_spec,
        seed=seed,
    )
    validate_ot_instance(instance)
    return instance


def validate_ot_instance(instance: OTInstance, atol: float = 1e-12) -> None:
    """Validate that an OT instance satisfies the assumptions of the project."""

    if instance.source_points.ndim != 2 or instance.target_points.ndim != 2:
        raise ValueError("Point clouds must be stored as 2D arrays.")
    if instance.source_points.shape[1] != instance.target_points.shape[1]:
        raise ValueError("Source and target point clouds must have the same dimension.")

    if instance.cost_matrix.shape != (instance.n_source, instance.n_target):
        raise ValueError("Cost matrix shape does not match the point clouds.")
    if np.any(instance.cost_matrix < -atol):
        raise ValueError("Squared Euclidean costs must be nonnegative.")

    _validate_probability_vector(instance.source_weights, name="source_weights", atol=atol)
    _validate_probability_vector(instance.target_weights, name="target_weights", atol=atol)


def default_gaussian_specs() -> tuple[GaussianSpec, GaussianSpec]:
    """Return a clean default pair of non-axis-aligned 2D Gaussian specs."""

    source_spec = GaussianSpec(
        mean=np.array([-1.25, 0.5], dtype=float),
        covariance=np.array([[1.2, 0.55], [0.55, 0.9]], dtype=float),
    )
    target_spec = GaussianSpec(
        mean=np.array([1.5, 2.0], dtype=float),
        covariance=np.array([[0.95, -0.35], [-0.35, 1.35]], dtype=float),
    )
    return source_spec, target_spec


def _validate_probability_vector(weights: Array, name: str, atol: float) -> None:
    """Check that the provided vector is a valid probability distribution."""

    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if np.any(weights < -atol):
        raise ValueError(f"{name} must be nonnegative.")
    if not np.isclose(weights.sum(), 1.0, atol=atol):
        raise ValueError(f"{name} must sum to 1.")
