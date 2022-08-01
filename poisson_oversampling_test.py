"""
poisson_oversampling_test.py
Author: Benjamin Floyd

Perform tests on how we can use a Poisson process to oversample a weighted population such that we get the expected
Poisson rate.
"""

import mpmath as mpm
import numpy as np
from scipy import stats
from scipy.special import factorial

seed = 123
rng = np.random.default_rng(seed)


class WeightedPoisson(stats.rv_discrete):
    @staticmethod
    def c(mu, r, a):
        return float(mpm.nsum(lambda k: float(mu) ** k * (k + float(a)) ** float(r) / mpm.factorial(k), [0, mpm.inf]))

    def _pmf(self, k, mu, r, a):
        return np.exp(k * np.log(mu) + r * np.log(k + a)) / (factorial(k) * self.c(mu, r, a))


def poisson_point_process(rate, dx, dy=None, lower_dx=0, lower_dy=0):
    """
    Uses a spatial Poisson point process to generate AGN candidate coordinates.

    Parameters
    ----------
    rate : float
        The model rate used in the Poisson distribution to determine the number of points being placed.
    dx, dy : int, Optional
        Upper bound on x- and y-axes respectively. If only `dx` is provided then `dy` = `dx`.
    lower_dx, lower_dy : int, Optional
        Lower bound on x- and y-axes respectively. If not provided, a default of 0 will be used

    Returns
    -------
    coord : np.ndarray
        Numpy array of (x, y) coordinates of AGN candidates
    """

    if dy is None:
        dy = dx

    # Draw from Poisson distribution to determine how many points we will place.
    p = stats.poisson(rate * np.abs(dx - lower_dx) * np.abs(dy - lower_dy)).rvs(random_state=rng)

    # Drop `p` points with uniform x and y coordinates
    x = rng.uniform(lower_dx, dx, size=p)
    y = rng.uniform(lower_dy, dy, size=p)

    # Combine the x and y coordinates.
    coord = np.vstack((x, y))

    return coord


# %% Test 1 Brute force interation
for _ in range(30):
    tol = 1e-4
    target = 5
    weighted_sum = 0.
    coords, weights = [], []

    while weighted_sum < target:
        coord = poisson_point_process(target, dx=1)
        weight = rng.random(size=len(coord[0]))

        weighted_sum += weight.sum()
        coords.append(coord)
        weights.append(weight)

        if np.abs(weighted_sum - target) <= tol:
            break

    total_draws = len(np.hstack(coords)[0])
    print(f'objects drawn: {total_draws}, {weighted_sum = :.3f}')

#%% Test 2 oversample then sample down

tol = 1e-2
target = 5.

# Massively oversample
width = 5
area = width * width
coords = poisson_point_process(target * 1000, dx=width)

# Assign weights to objects
weights = rng.random(size=coords.shape[1])

# Downsample
output_coords = []
output_weights = []
for i, w in enumerate(weights):
    proposal = [*output_weights, w]
    if np.abs(np.sum(proposal) / area - target) <= tol:
        output_weights.append(w)
        output_coords.append(coords[:, i])
        print(f'final rate: {np.sum(proposal)/area}, number drawn: {len(output_weights)}')
        break
    elif np.sum(proposal)/area > target:
        print(f'final rate: {np.sum(output_weights)/area}, number drawn {len(output_weights)} (overshot)')
        break
    else:
        output_weights.append(w)
        output_coords.append(coords[:, i])
        print(f'rate: {np.sum(proposal)/area}')

