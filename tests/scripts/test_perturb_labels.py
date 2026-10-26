"""scripts/pipeline/perturb_labels.py: each arm destroys only what it claims to."""
import importlib.util
import pickle
import sys
import pytest
from pathlib import Path

import numpy as np

from src.core.types import ForwardTrace, Labels

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pipeline" / "perturb_labels.py"
spec = importlib.util.spec_from_file_location("perturb_labels", MODULE_PATH)
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)


def _label():
    ft = ForwardTrace(K=6, S=10, selected_idx=np.array([7, 2, 5, 9, 0, 1]), candidate_scores=[],
                      best_scores=np.zeros(6), gains=np.array([500., 40., 12., 3., 0., 0.]),
                      stop_step=4, step_mask=np.array([True] * 4 + [False] * 2))
    return Labels(sce_ranking=np.arange(10)[::-1].copy(), sce_top_idx=np.array([7, 2, 5, 9]),
                  sce_target_idx=np.array([7]), forward_trace=ft)


def test_noise_zero_is_identity_and_noise_keeps_selection():
    lab = _label()
    pl.add_noise(lab, 0.0, np.random.default_rng(0))
    assert np.array_equal(lab.forward_trace.gains, _label().forward_trace.gains)
    pl.add_noise(lab, 0.7, np.random.default_rng(0))
    assert np.array_equal(lab.forward_trace.selected_idx, _label().forward_trace.selected_idx)
    assert not np.array_equal(lab.forward_trace.gains[:4], _label().forward_trace.gains[:4])
    assert np.all(lab.forward_trace.gains[4:] == 0)


def test_permute_keeps_selected_set_and_gain_values():
    lab = _label()
    pl.permute_gains(lab, np.random.default_rng(3))
    g = lab.forward_trace.gains
    assert sorted(g[:4]) == sorted(_label().forward_trace.gains[:4]) and np.all(g[4:] == 0)
    assert np.array_equal(lab.forward_trace.selected_idx, _label().forward_trace.selected_idx)


def test_remap_moves_every_scenario_index_by_one_permutation():
    clean, lab = _label(), _label()
    rng = np.random.default_rng(5)
    pl.remap_scenarios(lab, rng)
    pi = np.random.default_rng(5).permutation(10)
    assert np.array_equal(lab.forward_trace.selected_idx, pi[clean.forward_trace.selected_idx])
    for field in ("sce_ranking", "sce_top_idx", "sce_target_idx"):
        assert np.array_equal(getattr(lab, field), pi[getattr(clean, field)])
    assert np.array_equal(lab.forward_trace.gains, clean.forward_trace.gains)
    records = [{"label": clean}], [{"label": lab}]
    assert pl.fidelity(*records)["gain_kendall_tau"] == 1.0


@pytest.mark.parametrize("custom", [False, True])
def test_perturbed_labels_use_shared_data_by_default_and_keep_custom_layout(tmp_path, monkeypatch, custom):
    import src.utils.run_io as rio
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(rio, "REPO", repo)
    inputs = tmp_path / "custom" if custom else repo
    folder = "labels/vertex_cover" if custom else "data/vertex_cover"
    source = inputs / folder / "label_prise_test.pkl"
    source.parent.mkdir(parents=True)
    source.write_bytes(pickle.dumps([{"label": _label()}]))
    original = source.read_bytes()
    monkeypatch.setenv("NEURPRISE_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("NEURPRISE_RUN_ROOT", str(repo / "runs"))
    monkeypatch.setattr(sys, "argv", ["perturb_labels.py", "--label", f"{folder}/{source.name}",
                                    "--noise", "0.1", "--run-id", "one"])
    pl.main()
    root = repo / "runs/perturb_labels/one" if custom else repo
    output = root / folder / "label_prise_noise010_test.pkl"
    assert output.is_file()
    assert source.read_bytes() == original
    assert output.read_bytes() != original
    if not custom:
        before = output.read_bytes()
        monkeypatch.setattr(sys, "argv", ["perturb_labels.py", "--label", f"{folder}/{source.name}",
                                        "--noise", "0.1", "--run-id", "two"])
        with pytest.raises(FileExistsError, match="Data already exists"):
            pl.main()
        assert output.read_bytes() == before
