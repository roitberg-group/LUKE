# Roadmap: next steps

1. CI and Quality Gates

    - Add GitHub Actions workflow running: install (with submodules), ruff, mypy, pytest (CPU), and packaging sanity (build sdist/wheel).
    - Cache pip and torch builds; matrix on Python 3.10–3.12.
    - Upload coverage artifact and publish HTML on PRs (artifact only).

2. Testing & Validation

    - Unit tests for `io_utils` edge cases (padding, lattice present/absent, malformed lines).
    - Mocked tests for `ANIForceCalculator` to avoid heavy model execution (patch torchani calls).
    - Golden-file tests for fragment naming and output patterns.
    - Integration test: small XYZ through CLI to results directory; smoke checks on files.

3. Error Handling & Logging

    - Replace print with `rich` logging; add verbosity levels and `--quiet/--verbose` flags.
    - Clear exceptions for: bad input path/format, empty frames, unsupported elements, CUDA unavailability.
    - Consistent exit codes in CLI; structured error messages.

4. Examples & Docs

    - Add `examples/` with: quickstart notebook, CLI walkthrough, and programmatic API usage.
    - Architecture docs: data flow diagram, component responsibilities, FAQs.
    - Troubleshooting guide for common install/runtime issues (CPU vs CUDA, RDKit/OpenBabel).

5. Performance

    - Batch-friendly isolation (vectorize neighbor queries where possible).
    - Optional CUDA acceleration for distance computations.
    - Lazy model loading and reuse; optional half precision on CUDA.

6. Packaging & Distribution

    - Publish to PyPI (semver, changelog). Add `python -m build` and `twine check` steps to CI.
    - Provide CPU-only and CUDA-ready `environment.yaml` variants.
    - Optional extras for examples: `examples`, `docs` (sphinx/nbconvert).

7. Data I/O Enhancements

    - Add writer for parquet/JSON summaries of fragments and uncertainties.
    - Optional HDF5 writer for fragments with metadata.
    - Hash-based deduplication integrated in pipeline (reusing io_utils hashing).

8. Extensibility Hooks

    - Plugin-style registry for isolation strategies (distance cutoff, graph-based, chemistry-aware).
    - Config file support (YAML) to drive pipeline parameters.

9. Reproducibility

    - Seed control across torch/numpy.
    - Command provenance: write a run manifest with CLI args, versions, and environment info into results dir.
