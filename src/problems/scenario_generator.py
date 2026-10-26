import logging
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from src.utils.seed import get_rng

class BaseScenarioGenerator(ABC):
    """
    Strategy interface for generating (det, sce) costs.
    """

    def __init__(self, n_min_cost: int, n_max_cost: int, rng=None):
        self.n_min_cost = n_min_cost
        self.n_max_cost = n_max_cost
        self._rng = rng if rng is not None else get_rng()

    @abstractmethod
    def sample_costs(self, n_nodes: int, n_scenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            det: [n_nodes]
            sce: [n_scenarios, n_nodes]
        """
        pass


class IidUniformScenarioGenerator(BaseScenarioGenerator):
    """
    Original behaviour:
    - det: iid uniform in [n_min_cost, n_max_cost]
    - sce: iid uniform in [n_min_cost, n_max_cost]
    """

    def __init__(self, n_min_cost: int, n_max_cost: int, rng=None):
        super().__init__(n_min_cost, n_max_cost, rng)
        logging.info(f"Initialized IidUniformScenarioGenerator with n_min_cost={n_min_cost}, n_max_cost={n_max_cost}")

    def sample_costs(self, n_nodes: int, n_scenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        det = np.array(
            [self._rng.integers(self.n_min_cost, self.n_max_cost + 1) for _ in range(n_nodes)]
        )

        sce = np.array(
            [
                [self._rng.integers(self.n_min_cost, self.n_max_cost + 1) for _ in range(n_nodes)]
                for __ in range(n_scenarios)
            ]
        )
        return det, sce


class MultimodalScenarioGenerator(BaseScenarioGenerator):
    """Sample costs from a mixture of uniform modes.

    Draw K modes with means mu and deviations delta; sample each scenario
    uniformly within (1 +/- delta) * mu for its chosen mode. Deterministic
    costs are uniform; ``sample_costs`` returns ``(det [N], sce [S, N])``.
    """

    def __init__(
        self,
        det_min_cost: int = 25,
        det_max_cost: int = 75,
        mu_min_cost: int = 25,
        mu_max_cost: int = 75,
        delta_min: float = 0.1,
        delta_max: float = 0.5,
        K_min: int = 3,
        K_max: int = 8,
        rng: np.random.Generator | None = None,
    ):
        # we can still call super with mu-bounds if you want to reuse them
        super().__init__(n_min_cost=mu_min_cost, n_max_cost=mu_max_cost, rng=rng)
        logging.info(f"Initialized MultimodalScenarioGenerator with K={K_min}-{K_max}")
        self.det_min_cost = det_min_cost
        self.det_max_cost = det_max_cost
        self.mu_min_cost = mu_min_cost
        self.mu_max_cost = mu_max_cost
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.K_min = K_min
        self.K_max = K_max

    def sample_costs(self, n_nodes: int, n_scenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        # 1) deterministic costs: narrow interval
        det = self._rng.integers(
            low=self.det_min_cost,
            high=self.det_max_cost + 1,
            size=n_nodes,
        ).astype(float)

        # 2) sample K modes
        K = self._rng.integers(self.K_min, self.K_max + 1)

        # 3) mode midpoints mu_k and relative deviations delta_k
        mu = self._rng.uniform(
            low=self.mu_min_cost,
            high=self.mu_max_cost,
            size=(K, n_nodes),
        )
        delta = self._rng.uniform(
            low=self.delta_min,
            high=self.delta_max,
            size=(K, n_nodes),
        )

        # 4) scenarios
        sce = np.zeros((n_scenarios, n_nodes), dtype=float)
        for s in range(n_scenarios):
            k = self._rng.integers(0, K)
            low = (1.0 - delta[k]) * mu[k]
            high = (1.0 + delta[k]) * mu[k]
            sce[s] = self._rng.uniform(low=low, high=high)

        return det, sce


class NormalScenarioGenerator(BaseScenarioGenerator):
    """
    Normal-distribution cost sampler for graph problems (selection, vertex_cover).

    - det costs: per-node means mu_v sampled from U[mu_min, mu_max]
    - scenario costs: normal(mu_v, sigma_v), clipped to [n_min_cost, n_max_cost]
    - sigma_v = clip(sigma_ratio * mu_v, sigma_min, sigma_max)
    - fixed_profile=True: reuse (mu, sigma) across calls for same n_nodes
    """

    def __init__(
        self,
        n_min_cost: int = 1,
        n_max_cost: int = 100,
        mu_min: float = 25.0,
        mu_max: float = 75.0,
        sigma_ratio: float = 0.15,
        sigma_min: float = 3.0,
        sigma_max: float = 15.0,
        fixed_profile: bool = True,
        rng: np.random.Generator | None = None,
    ):
        if mu_max < mu_min:
            raise ValueError("mu_max must be >= mu_min.")
        if sigma_ratio < 0.0:
            raise ValueError("sigma_ratio must be nonnegative.")
        if sigma_min < 0.0 or sigma_max < sigma_min:
            raise ValueError("sigma bounds must satisfy 0 <= sigma_min <= sigma_max.")

        super().__init__(n_min_cost, n_max_cost, rng)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.sigma_ratio = float(sigma_ratio)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.fixed_profile = bool(fixed_profile)
        self._profile_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        logging.info(
            "Initialized NormalScenarioGenerator with mu=[%.1f, %.1f], "
            "sigma_ratio=%.3f, sigma_clip=[%.1f, %.1f], cost=[%d, %d], fixed_profile=%s",
            self.mu_min, self.mu_max,
            self.sigma_ratio, self.sigma_min, self.sigma_max,
            self.n_min_cost, self.n_max_cost, self.fixed_profile,
        )

    def sample_costs(self, n_nodes: int, n_scenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive.")
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive.")

        mu, sigma = self._get_profile(n_nodes=n_nodes)

        # det = rounded means
        det = np.round(mu).astype(int)
        det = np.clip(det, self.n_min_cost, self.n_max_cost)

        # sce = normal(mu, sigma), clipped
        sce = self._rng.normal(
            loc=mu.reshape(1, -1),
            scale=sigma.reshape(1, -1),
            size=(n_scenarios, n_nodes),
        )
        sce = np.round(sce).astype(int)
        sce = np.clip(sce, self.n_min_cost, self.n_max_cost)

        return det, sce

    def _get_profile(self, *, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
        if self.fixed_profile and n_nodes in self._profile_cache:
            mu_cached, sigma_cached = self._profile_cache[n_nodes]
            return mu_cached.copy(), sigma_cached.copy()

        mu = self._rng.uniform(low=self.mu_min, high=self.mu_max, size=n_nodes)
        sigma = np.clip(self.sigma_ratio * mu, self.sigma_min, self.sigma_max)

        if self.fixed_profile:
            self._profile_cache[n_nodes] = (mu.copy(), sigma.copy())
        return mu, sigma


class CFLPNormalDemandScenarioGenerator(BaseScenarioGenerator):
    """Sample normal customer demands for CFLP.

    Draw means in [mu_min, mu_max] and clip sigma_ratio * mu to the sigma bounds;
    ``fixed_profile=True`` reuses the profile for the same customer count.
    ``sample_costs`` returns means [n_customers] and clipped demands [n_scenarios, n_customers].
    """

    def __init__(
        self,
        demand_min: float = 10.0,
        demand_max: float = 500.0,
        mu_min: float = 180.0,
        mu_max: float = 260.0,
        sigma_ratio: float = 0.08,
        sigma_min: float = 12.0,
        sigma_max: float = 22.0,
        fixed_profile: bool = True,
        clip_to_bounds: bool = True,
        rng: np.random.Generator | None = None,
    ):
        if demand_min < 0.0:
            raise ValueError("demand_min must be nonnegative.")
        if demand_max <= demand_min:
            raise ValueError("demand_max must be larger than demand_min.")
        if mu_max < mu_min:
            raise ValueError("mu_max must be >= mu_min.")
        if sigma_ratio < 0.0:
            raise ValueError("sigma_ratio must be nonnegative.")
        if sigma_min < 0.0 or sigma_max < sigma_min:
            raise ValueError("sigma bounds must satisfy 0 <= sigma_min <= sigma_max.")

        super().__init__(
            n_min_cost=int(np.floor(demand_min)),
            n_max_cost=int(np.ceil(demand_max)),
            rng=rng,
        )
        self.demand_min = float(demand_min)
        self.demand_max = float(demand_max)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.sigma_ratio = float(sigma_ratio)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.fixed_profile = bool(fixed_profile)
        self.clip_to_bounds = bool(clip_to_bounds)
        self._profile_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        logging.info(
            "Initialized CFLPNormalDemandScenarioGenerator with mu=[%.3f, %.3f], "
            "sigma_ratio=%.4f sigma_clip=[%.3f, %.3f], demand=[%.3f, %.3f], fixed_profile=%s",
            self.mu_min,
            self.mu_max,
            self.sigma_ratio,
            self.sigma_min,
            self.sigma_max,
            self.demand_min,
            self.demand_max,
            self.fixed_profile,
        )

    def sample_costs(self, n_nodes: int, n_scenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive.")
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive.")

        mu, sigma = self._get_profile(n_nodes=n_nodes)
        det = mu.astype(np.float32, copy=True)
        sce = self._rng.normal(
            loc=mu.reshape(1, -1),
            scale=sigma.reshape(1, -1),
            size=(n_scenarios, n_nodes),
        ).astype(np.float32)

        if self.clip_to_bounds:
            det = np.clip(det, self.demand_min, self.demand_max)
            sce = np.clip(sce, self.demand_min, self.demand_max)
        return det.astype(np.float32), sce.astype(np.float32)

    def _get_profile(self, *, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
        if self.fixed_profile and n_nodes in self._profile_cache:
            mu_cached, sigma_cached = self._profile_cache[n_nodes]
            return mu_cached.copy(), sigma_cached.copy()

        mu = self._rng.uniform(
            low=self.mu_min,
            high=self.mu_max,
            size=n_nodes,
        ).astype(np.float32)
        sigma = np.clip(
            self.sigma_ratio * mu,
            self.sigma_min,
            self.sigma_max,
        ).astype(np.float32)
        if self.fixed_profile:
            self._profile_cache[n_nodes] = (mu.copy(), sigma.copy())
        return mu, sigma


class CFLPMultimodalDemandScenarioGenerator(BaseScenarioGenerator):
    """
    Multimodal demand sampler tailored for CFLP.

    Returns:
      - det: one sampled base-demand profile [n_customers]
      - sce: scenario demands [n_scenarios, n_customers]
    """

    def __init__(
        self,
        demand_min: float = 10.0,
        demand_max: float = 500.0,
        mode_count_min: int = 3,
        mode_count_max: int = 6,
        mode_mu_min: float = 80.0,
        mode_mu_max: float = 380.0,
        mode_delta_min: float = 0.05,
        mode_delta_max: float = 0.20,
        shared_shock_std: float = 0.08,
        clip_to_bounds: bool = True,
        rng: np.random.Generator | None = None,
    ):
        if demand_min < 0.0:
            raise ValueError("demand_min must be nonnegative.")
        if demand_max <= demand_min:
            raise ValueError("demand_max must be larger than demand_min.")
        if mode_count_min <= 0 or mode_count_max < mode_count_min:
            raise ValueError("mode_count bounds must satisfy 0 < mode_count_min <= mode_count_max.")
        if mode_mu_max <= mode_mu_min:
            raise ValueError("mode_mu_max must be larger than mode_mu_min.")
        if mode_delta_min < 0.0 or mode_delta_max < mode_delta_min:
            raise ValueError("mode_delta bounds must satisfy 0 <= min <= max.")
        if shared_shock_std < 0.0:
            raise ValueError("shared_shock_std must be nonnegative.")

        super().__init__(
            n_min_cost=int(np.floor(demand_min)),
            n_max_cost=int(np.ceil(demand_max)),
            rng=rng,
        )
        self.demand_min = float(demand_min)
        self.demand_max = float(demand_max)
        self.mode_count_min = int(mode_count_min)
        self.mode_count_max = int(mode_count_max)
        self.mode_mu_min = float(mode_mu_min)
        self.mode_mu_max = float(mode_mu_max)
        self.mode_delta_min = float(mode_delta_min)
        self.mode_delta_max = float(mode_delta_max)
        self.shared_shock_std = float(shared_shock_std)
        self.clip_to_bounds = bool(clip_to_bounds)

        logging.info(
            "Initialized CFLPMultimodalDemandScenarioGenerator with modes=%d-%d, "
            "mu=[%.3f, %.3f], delta=[%.3f, %.3f], demand=[%.3f, %.3f]",
            self.mode_count_min,
            self.mode_count_max,
            self.mode_mu_min,
            self.mode_mu_max,
            self.mode_delta_min,
            self.mode_delta_max,
            self.demand_min,
            self.demand_max,
        )

    def sample_costs(self, n_nodes: int, n_scenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive.")
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive.")

        mode_count = int(self._rng.integers(self.mode_count_min, self.mode_count_max + 1))
        mu = self._rng.uniform(
            low=self.mode_mu_min,
            high=self.mode_mu_max,
            size=(mode_count, n_nodes),
        )
        delta = self._rng.uniform(
            low=self.mode_delta_min,
            high=self.mode_delta_max,
            size=(mode_count, n_nodes),
        )

        det_mode = int(self._rng.integers(0, mode_count))
        det = np.asarray(mu[det_mode], dtype=np.float32)

        sce = np.zeros((n_scenarios, n_nodes), dtype=np.float32)
        for s in range(n_scenarios):
            k = int(self._rng.integers(0, mode_count))
            low = (1.0 - delta[k]) * mu[k]
            high = (1.0 + delta[k]) * mu[k]
            sampled = self._rng.uniform(low=low, high=high).astype(np.float32)

            # Shared multiplicative shock introduces customer co-movement within a scenario.
            if self.shared_shock_std > 0.0:
                shock = float(self._rng.normal(loc=0.0, scale=self.shared_shock_std))
                sampled = sampled * max(0.0, 1.0 + shock)
            sce[s] = sampled

        if self.clip_to_bounds:
            det = np.clip(det, self.demand_min, self.demand_max)
            sce = np.clip(sce, self.demand_min, self.demand_max)

        return det.astype(np.float32), sce.astype(np.float32)
