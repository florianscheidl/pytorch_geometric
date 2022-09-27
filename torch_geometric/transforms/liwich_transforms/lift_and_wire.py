
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms.lifts import LiftTransform
from torch_geometric.transforms.wirings import WiringTransform


class LiftAndWire(BaseTransform):
    def __init__(self, lift: LiftTransform, wiring: WiringTransform):
        self.lift = lift
        self.wiring = wiring
        super().__init__()

    def __call__(self, data) -> HeteroData:
        lifted_data = self.lift(data)
        self.wiring.boundary_adjacency_tensors = self.lift.boundary_adjacency_tensors
        return self.wiring(lifted_data)
