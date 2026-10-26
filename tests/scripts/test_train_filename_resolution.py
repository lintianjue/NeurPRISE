import importlib.util
import sys
from pathlib import Path
from hydra.core.hydra_config import HydraConfig


def _load_train_module():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    module_path = root / "scripts" / "03_train.py"
    spec = importlib.util.spec_from_file_location("train_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_p_for_filename_selection_uses_half_nodes():
    # Motivation: selection filenames should use p_select derived from n_nodes.
    module = _load_train_module()
    p_val = module._resolve_p_for_filename("selection", 20, p_select=25, p_edge=None)
    assert p_val == 10


def test_resolve_p_for_filename_vertex_cover_prefers_p_edge():
    # Motivation: non-selection problems should keep using p_edge for filenames.
    module = _load_train_module()
    p_val = module._resolve_p_for_filename("vertex_cover", 20, p_select=None, p_edge=10)
    assert p_val == 10


def test_hydra_choice_fragment_prefers_config_name(monkeypatch):
    # Motivation: use config file name when YAML name field is wrong.
    # Patching the real HydraConfig class: monkeypatch undoes it, otherwise every later test
    # in the process that resolves a checkpoint name would see the dummy
    # (tests/training/test_run_naming.py would lose its `_c-<config>` fragment).
    module = _load_train_module()

    class DummyRuntime:
        choices = {"model": "one_shot_crossattn_edge_dim1_small_instancenorm"}

    class DummyHydra:
        runtime = DummyRuntime()

    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: DummyHydra()))
    frag = module._hydra_choice_fragment("model", default="fallback")
    assert frag == "one_shot_crossattn_edge_dim1_small_instancenorm"


def test_hydra_choice_fragment_falls_back(monkeypatch):
    # Motivation: keep previous behavior when Hydra choices are unavailable.
    module = _load_train_module()
    monkeypatch.setattr(
        HydraConfig, "get",
        staticmethod(lambda: (_ for _ in ()).throw(Exception("no hydra"))),
    )
    frag = module._hydra_choice_fragment("model", default="fallback")
    assert frag == "fallback"
