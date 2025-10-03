import torch

from luke import pipeline as luke_pipeline


class DummyCalc:
    """Lightweight stand-in for ANIForceCalculator."""

    def __init__(self, *_, **__):
        import torch
        self.device = torch.device("cpu")

    def process_dataset(self, dataset_path: str, batch_size: int = 2500):  # noqa: ARG002
        # Return a single "structure" with one bad atom
        species = torch.tensor([[1, 6]])
        coords = torch.tensor([[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]])
        good_or_bad = torch.tensor([[1, 0]])  # mark first atom bad
        energy = torch.tensor([0.0])
        qbc = torch.tensor([0.0])
        stdev = torch.tensor([4.0])  # > 3.5 threshold ensures retention
        return species, coords, good_or_bad, energy, qbc, stdev


def test_pipeline_with_mock(tmp_path, monkeypatch):
    # Monkeypatch the real ANIForceCalculator inside pipeline
    monkeypatch.setattr(luke_pipeline, "ANIForceCalculator", DummyCalc)
    input_xyz = tmp_path / "dummy.xyz"
    input_xyz.write_text("""2\ncomment\nH 0 0 0\nC 0.9 0 0\n""")
    out_dir = tmp_path / "out"
    result_dir = luke_pipeline.run_pipeline(input_xyz, out_dir, sanitize=False)
    assert result_dir.exists()
    # Expect at least one fragment file
    frags = list(result_dir.glob("frag_*.xyz"))
    assert frags, "Expected at least one fragment file from mock pipeline"
