"""Gaussian optimal transport utilities for Part 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm

from .problem_setup import GaussianSpec

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GaussianW2Result:
    """Closed-form Wasserstein-2 summary for two Gaussians."""

    mean_term: float
    covariance_term: float
    w2_squared: float


def gaussian_w2_squared(
    source_spec: GaussianSpec,
    target_spec: GaussianSpec,
) -> GaussianW2Result:
    """Compute the Bures-Wasserstein closed form for two Gaussian measures."""

    if source_spec.dimension != target_spec.dimension:
        raise ValueError("Both Gaussians must have the same dimension.")

    mean_difference = source_spec.mean - target_spec.mean
    mean_term = float(mean_difference @ mean_difference)

    sigma1_sqrt = _real_matrix_sqrt(source_spec.covariance)
    middle = sigma1_sqrt @ target_spec.covariance @ sigma1_sqrt
    middle_sqrt = _real_matrix_sqrt(middle)

    covariance_term = float(
        np.trace(source_spec.covariance + target_spec.covariance - 2.0 * middle_sqrt)
    )
    w2_squared = mean_term + covariance_term

    return GaussianW2Result(
        mean_term=mean_term,
        covariance_term=covariance_term,
        w2_squared=float(max(w2_squared, 0.0)),
    )


def covariance_ellipse_patch(
    mean: Array,
    covariance: Array,
    *,
    n_std: float,
    edgecolor: str,
    linestyle: str = "-",
    linewidth: float = 1.8,
    alpha: float = 0.95,
) -> Ellipse:
    """Create a matplotlib ellipse patch for one covariance contour."""

    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    width = 2.0 * n_std * np.sqrt(max(eigenvalues[0], 0.0))
    height = 2.0 * n_std * np.sqrt(max(eigenvalues[1], 0.0))
    principal_vector = eigenvectors[:, 0]
    angle = float(np.degrees(np.arctan2(principal_vector[1], principal_vector[0])))

    return Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )


def _real_matrix_sqrt(matrix: Array) -> Array:
    """Compute a numerically real matrix square root for an SPD matrix."""

    sqrt_matrix = sqrtm(np.asarray(matrix, dtype=float))
    sqrt_matrix = np.real_if_close(sqrt_matrix, tol=1_000)
    return np.asarray(np.real(sqrt_matrix), dtype=float)
