import torch
import torch.nn as nn
import pytest
from torch_geometric.data import Data, Batch

from src.models.gnn import ScenarioGCN, NodeNorm


# Adjust import to point to your actual file


@pytest.fixture
def batch_setup():
    """Creates a small dummy batch for testing."""
    in_dim = 9
    num_nodes = 20
    num_edges = 40
    B = 2

    # Random features
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 1)

    # Create batch index (half nodes in graph 0, half in graph 1)
    batch_idx = torch.cat([
        torch.zeros(num_nodes // 2, dtype=torch.long),
        torch.ones(num_nodes // 2, dtype=torch.long)
    ])

    return x, edge_index, edge_attr, batch_idx, B


def test_initial_embedding_projection(batch_setup):
    """
    Feature 1 Test: Verify raw inputs (9 dim) are projected to hidden_dim
    *before* the first convolution sees them.
    """
    x, edge_index, edge_attr, batch, _ = batch_setup
    in_dim = 9
    hidden_dim = 32
    out_dim = 16

    model = ScenarioGCN(in_dim, hidden_dim, out_dim, edge_attr_dim=1)

    # 1. Check layer existence
    assert hasattr(model, 'emb_mlp'), "Model must have emb_mlp for initial projection"
    assert isinstance(model.emb_mlp[0], nn.Linear)
    assert model.emb_mlp[0].in_features == in_dim
    assert model.emb_mlp[0].out_features == hidden_dim

    # 2. Check execution via Hook
    # We want to verify that the 'x' passed to the first conv layer has shape [N, hidden_dim]
    shapes_seen = []

    def hook_fn(module, input, output):
        # input is a tuple (x, edge_index, ...)
        x_in = input[0]
        shapes_seen.append(x_in.shape)

    handle = model.convs[0].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(x, edge_index, edge_attr, batch)

    handle.remove()

    # The first conv should see hidden_dim, NOT in_dim
    assert shapes_seen[0][1] == hidden_dim, \
        f"First convolution should receive hidden_dim ({hidden_dim}), got {shapes_seen[0][1]}"


def test_batchnorm_is_active(batch_setup):
    """
    Feature 2 Test: Verify BatchNorm layers are tracking statistics.
    This confirms we are using the 'Modern' robust normalization.
    """
    x, edge_index, edge_attr, batch, _ = batch_setup
    model = ScenarioGCN(node_in_dim=9, hidden_dim=16, out_dim=8, edge_attr_dim=1)
    model.train()  # BN only updates stats in train mode

    # Get the first BN layer in the conv block
    bn_layer = model.norms[0]

    # Snapshot initial running stats
    init_mean = bn_layer.running_mean.clone()
    init_var = bn_layer.running_var.clone()

    # Run Forward Pass
    model(x, edge_index, edge_attr, batch)

    # Check if stats moved
    assert not torch.equal(bn_layer.running_mean, init_mean), \
        "BatchNorm running_mean should update during training"
    assert not torch.equal(bn_layer.running_var, init_var), \
        "BatchNorm running_var should update during training"


def test_edge_projection_integration(batch_setup):
    """
    Feature 3 Test: Verify Edge Projection is working and trainable.
    """
    x, edge_index, edge_attr, batch, _ = batch_setup
    model = ScenarioGCN(node_in_dim=9, hidden_dim=16, out_dim=8, edge_attr_dim=1)
    model.train()

    # Enable gradients
    edge_attr.requires_grad = True

    out = model(x, edge_index, edge_attr, batch)
    loss = out.sum()
    loss.backward()

    # Check if gradient flowed back to edge projection weights
    assert model.edge_proj.weight.grad is not None
    assert model.edge_proj.weight.grad.abs().sum() > 0, \
        "Gradients must flow through edge_proj (proving edges are used)"


def test_end_to_end_dims(batch_setup):
    """
    Feature 4 Test: Final Output Shape Check.
    """
    x, edge_index, edge_attr, batch, B = batch_setup
    out_dim = 8
    model = ScenarioGCN(node_in_dim=9, hidden_dim=16, out_dim=out_dim, edge_attr_dim=1)

    out = model(x, edge_index, edge_attr, batch)

    assert out.shape == (B, out_dim)
    assert not torch.isnan(out).any()


def test_layernorm_config(batch_setup):
    """Motivation: support LayerNorm as a configurable option."""
    x, edge_index, edge_attr, batch, B = batch_setup
    model = ScenarioGCN(node_in_dim=9, hidden_dim=16, out_dim=8, edge_attr_dim=1, norm_type="layer")

    assert isinstance(model.emb_mlp[1], nn.LayerNorm)
    assert all(isinstance(n, nn.LayerNorm) for n in model.norms)

    out = model(x, edge_index, edge_attr, batch)
    assert out.shape == (B, 8)


def test_instancenorm_config(batch_setup):
    """Motivation: support InstanceNorm as a configurable option."""
    x, edge_index, edge_attr, batch, B = batch_setup
    model = ScenarioGCN(node_in_dim=9, hidden_dim=16, out_dim=8, edge_attr_dim=1, norm_type="instance")

    assert isinstance(model.emb_mlp[1], NodeNorm)
    assert all(isinstance(n, NodeNorm) for n in model.norms)

    out = model(x, edge_index, edge_attr, batch)
    assert out.shape == (B, 8)


@pytest.fixture
def gine_identity_model() -> ScenarioGCN:
    """Builds a ScenarioGCN configured so GINEConv behaves like a hand-computable GIN."""
    model = ScenarioGCN(
        node_in_dim=1,
        hidden_dim=1,
        out_dim=1,
        num_layers=1,
        dropout=0.0,
        edge_attr_dim=1,
        norm_type="layer",
    )
    model.emb_mlp = nn.Identity()
    model.edge_proj = nn.Identity()
    model.norms = nn.ModuleList([nn.Identity()])
    model.activation = nn.Identity()
    model.dropout = nn.Identity()

    conv = model.convs[0]
    conv.eps.data.fill_(0.0)
    for layer in conv.nn:
        if isinstance(layer, nn.Linear):
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()

    return model


# Motivation: with zero edge attributes, GINEConv reduces to GIN aggregation.
def test_gineconv_matches_gin_with_zero_edge_attr(gine_identity_model: ScenarioGCN) -> None:
    # Two nodes with scalar features: x0=1, x1=2.
    x = torch.tensor([[1.0], [2.0]])
    # Two directed edges (0->1 and 1->0), so each node has one neighbor.
    edge_index = torch.tensor([[0, 1], [1, 0]])
    # Zero edge features => GINE reduces to GIN (sum of neighbor features).
    edge_attr = torch.zeros(2, 1)
    # Both nodes belong to the same graph for mean pooling.
    scen_batch = torch.zeros(2, dtype=torch.long)

    out = gine_identity_model(x, edge_index, edge_attr, scen_batch)
    # GINE formula (eps=0, identity MLP, edge_attr=0):
    # h_i = x_i + sum_j x_j. So h0=1+2=3, h1=2+1=3, mean=3.
    assert out.shape == (1, 1)
    assert out.item() == pytest.approx(3.0)


# Motivation: non-zero edge attributes contribute additively in GINEConv aggregation.
def test_gineconv_includes_edge_attr(gine_identity_model: ScenarioGCN) -> None:
    # Two nodes with scalar features: x0=1, x1=2.
    x = torch.tensor([[1.0], [2.0]])
    # Two directed edges (0->1 and 1->0).
    edge_index = torch.tensor([[0, 1], [1, 0]])
    # Non-zero edge features; each edge_attr aligns with edge_index column.
    edge_attr = torch.tensor([[0.5], [1.5]])
    # Both nodes belong to the same graph for mean pooling.
    scen_batch = torch.zeros(2, dtype=torch.long)

    out = gine_identity_model(x, edge_index, edge_attr, scen_batch)
    # GINE formula (eps=0, identity MLP):
    # h0 = 1 + (2 + 1.5) = 4.5  [incoming edge 1->0 has attr 1.5]
    # h1 = 2 + (1 + 0.5) = 3.5  [incoming edge 0->1 has attr 0.5]
    # mean = (4.5 + 3.5) / 2 = 4.
    assert out.shape == (1, 1)
    assert out.item() == pytest.approx(4.0)


# Motivation: verify GINEConv aggregation on a simple 3-node star graph.
def test_gineconv_three_node_star_zero_edge_attr(
    gine_identity_model: ScenarioGCN,
) -> None:
    # Three nodes with scalar features: x0=1, x1=2, x2=3.
    x = torch.tensor([[1.0], [2.0], [3.0]])
    # Star centered at node 0 (undirected edges encoded as two directions).
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    edge_attr = torch.zeros(4, 1)
    scen_batch = torch.zeros(3, dtype=torch.long)

    out = gine_identity_model(x, edge_index, edge_attr, scen_batch)
    # GINE formula (eps=0, identity MLP, edge_attr=0):
    # h0 = 1 + (2 + 3) = 6, h1 = 2 + 1 = 3, h2 = 3 + 1 = 4.
    # mean = (6 + 3 + 4) / 3 = 13/3.
    assert out.shape == (1, 1)
    assert out.item() == pytest.approx(13.0 / 3.0)
