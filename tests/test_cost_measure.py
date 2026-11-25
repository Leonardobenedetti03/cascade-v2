import pytest

torch = pytest.importorskip("torch")

from src.cascade.evaluation.cost import measure_gmacs, measure_latency_ms  # noqa: E402


def test_linear_counts_exact():
    m = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5))
    out = measure_gmacs(m, (torch.zeros(1, 10),))
    assert out["gmacs_torch"] == pytest.approx((10 * 20 + 20 * 5) / 1e9)
    assert out["gmacs_fvcore"] == pytest.approx((10 * 20 + 20 * 5) / 1e9)
    assert out["method"] == "fvcore"


def test_transformer_fastpath_disabled_and_restored():
    layer = torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=True)
    enc = torch.nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
    before = torch.backends.mha.get_fastpath_enabled()
    out = measure_gmacs(enc, (torch.zeros(1, 10, 16),), counter="torch")
    n, d, ff = 10, 16, 32
    expected = n * 4 * d * d + 2 * n * n * d + 2 * n * d * ff
    assert out["gmacs"] == pytest.approx(expected / 1e9, rel=0.02)
    assert torch.backends.mha.get_fastpath_enabled() == before


def test_latency():
    m = torch.nn.Linear(4, 4)
    r = measure_latency_ms(m, (torch.zeros(1, 4),), reps=3)
    assert r["latency_ms"] >= 0 and r["device"] == "cpu"
