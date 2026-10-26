from abc import ABC, abstractmethod
from typing import Any, Dict, List

from torch_geometric.data import Data

from .types import Instance, Labels

class IProblem(ABC):
    @abstractmethod
    def sample(self) -> Instance:
        ...

    @abstractmethod
    def solve_exact(self, instance: Instance, time_limit: float = None, threads: int = 1) -> Labels:
        ...

    @abstractmethod
    def decision_dim(self, instance: Instance) -> int:
        ...

    @abstractmethod
    def collate_fn(self, batch: List[Instance]) -> Instance:
        """
        Collate a list of Instance objects into a batched Instance.
        Each problem defines how to batch its specific data structure.
        """
        ...
    @abstractmethod
    def to_pyg_data(self, batch: Instance) -> List[Data]:
        ...

    @abstractmethod
    def to_pyg_data_det(self, batch: Instance) -> List[Data]:
        ...

    @abstractmethod
    def get_scenarios(self, data_np: Dict[str, Any]):
        """
        Return scenario array from a numpy dict (problem-specific keying).
        """
        ...
