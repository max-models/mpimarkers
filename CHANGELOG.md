# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `ParticleStore`: per-rank marker storage in component-major arrays (positions, velocities,
  weights, alive, int64 ids) on NumPy or CuPy, with `reserve`, `add`, `deactivate`, `remove`,
  `clear`, `compact`, `compact_if_needed`, `sort_by_key`, `snapshot`/`restore` and a plasmacoll
  `MarkerSet` view (`as_marker_set`).
- `IdAllocator`: globally unique int64 ids with a rank stride, independent of the decomposition.
- `Decomposition` protocol and the reference `BoxDecomposition`.
- `migrate` (collective, sends only the leavers, ids as int64), `keep_owned`, `global_count` and
  `gather_columns`.
- Tests that launch `mpiexec -n 2` and `-n 4` jobs, a tutorial ("Moving markers between ranks"),
  guides and an example.

### Changed

- `__version__` is read from the installed package metadata, not hardcoded.
- Docstrings follow the NumPy convention.

### Removed

- The template's console entry point, example tutorial, example script and `setup_project.sh`.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
