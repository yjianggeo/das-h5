# das-h5 — Design Spec

**Date:** 2026-06-15
**Status:** Approved (pending written-spec review)

## 1. Purpose & Scope

Reorganize the DASh5 code that currently lives, flat and ungrouped, in
`D:/Code/Scatter_migration` into a clean, installable Python package named
`das_h5`, placed at `D:/Projects/das-h5/` (sibling of `das-ancc`). Ship a
bilingual (English / 中文) README in the same style as `das-ancc/README.md`.

**Hard constraints:**

- The original code in `Scatter_migration` is **not modified**. This is a new copy.
- **No algorithm changes.** Filter design, decimation, the strain-rate conversion
  constants in `read_h5`, the Welch PSD computation, spectrogram math — all
  preserved byte-for-behavior. "Optimization" here means *structure and packaging*,
  not numerical behavior.
- **CLI compatibility preserved.** All `DASh5.py` command-line flags keep their
  names, defaults, and semantics so the original manual's example commands still work.

### Files that ARE DASh5 (in scope)

`DASh5.py` plus the 10 modules it imports:

| Source file | Functions |
|---|---|
| `das_readdata.py` | `get_database_h5`, `make_dbfile`, `read_h5`, `read_sac` |
| `das_writedata.py` | `save_sac`, `save_sac_fast`, `save_npz` |
| `das_outsac.py` | `append_sac`, `h5_to_sac` |
| `das_processing.py` | `filter`, `detrend`, `rmean`, `norm_data`, `decimate`, `interpolate`, `nextpow2` (CLI-used subset only) |
| `das_imshow.py` | `das_imshow` |
| `das_fillplot.py` | `das_fillplot` |
| `das_single.py` | `spectrograms`, `single_plot`, `single_plot_sac` |
| `das_psd.py` | `das_psd`, `das_psd_type`, `das_psd_pkl` |
| `das_psd_plot.py` | `plot_time_psd`, `plot_psd_freq`, `plot_cure_single`, `plot_psd_freq_time`, `plot_psd_freq_time_single`, `plot_psd_channel_statistic` |
| `das_sound.py` | `das_sound` |

### Files explicitly EXCLUDED

- `das_spectrogram.py` — standalone script with its own `main()`, never imported by DASh5.
- `das_psd-Copy1.py` — duplicate of `das_psd.py`.
- `SAC_*.py`, `EDWav.py`, `H5_Decimate.py`, `Plot_*.py`, all `test*.ipynb` — unrelated to DASh5.
- From `das_processing.py`: `maps`, `phase_shift`, `fk_filter_bandpass`, `fkdip`,
  `stack_func`, `stack_smooth`, `stack_part` — not part of the DASh5 tool chain
  (decision: drop them to keep package responsibility single; they remain in the
  original source).

## 2. Package Layout

```
das-h5/
├── README.md                  # bilingual EN / 中文 (das-ancc style)
├── pyproject.toml             # pip install -e .  ; console entry dash5 = das_h5.cli:main
├── das_h5/
│   ├── __init__.py            # package version + curated re-exports
│   ├── cli.py                 # from DASh5.py: argparse + orchestration, wrapped in main()
│   ├── io/
│   │   ├── __init__.py
│   │   ├── h5_reader.py       # get_database_h5, make_dbfile, read_h5, read_sac
│   │   ├── writer.py          # save_sac, save_sac_fast, save_npz
│   │   └── sac_export.py      # append_sac, h5_to_sac
│   ├── process/
│   │   ├── __init__.py
│   │   ├── filters.py         # nextpow2, norm_data, rmean, detrend, filter
│   │   └── resample.py        # decimate, interpolate
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── waterfall.py       # das_imshow
│   │   ├── wiggle.py          # das_fillplot
│   │   └── trace.py           # spectrograms, single_plot, single_plot_sac
│   ├── psd/
│   │   ├── __init__.py
│   │   ├── compute.py         # das_psd, das_psd_type, das_psd_pkl
│   │   └── plots.py           # the 6 plot_* functions
│   └── sound/
│       ├── __init__.py
│       └── audio.py           # das_sound
├── tests/                     # synthetic-data unit tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_process.py
│   ├── test_psd.py
│   └── test_io.py
└── docs/superpowers/{specs,plans}/
```

## 3. What "optimize" concretely means

1. **Split the monolith.** `DASh5.py` runs as module-level script code today. Wrap
   the argparse setup + orchestration in `cli.py` as `def main(): ...`, guarded by
   `if __name__ == "__main__":`, and exposed as a console-script entry point.
   Behavior and flags unchanged.
2. **Fix imports.** Replace `from das_readdata import ...` style flat imports with
   package imports (`from das_h5.io.h5_reader import ...`). Internal cross-module
   deps (`h5_reader → resample`, `sac_export → writer/resample`, `compute → filters`)
   rewired the same way.
3. **Group & de-duplicate.** Drop the duplicate/excluded files listed above. Hoist
   function-local imports to module top where trivially safe (no behavior change).
   Normalize module docstrings.
4. **Do not touch numerics.** Every kept function's body is preserved as-is.

## 4. Tests (synthetic data only, no real DAS files)

- **`test_process.py`**: `filter` bandpass suppresses out-of-band energy; `decimate`
  yields the expected sample rate and length; `interpolate` factor correctness;
  `norm_data` peak == 1; `detrend` removes a known linear trend.
- **`test_psd.py`**: `das_psd` on a synthetic sinusoid peaks at the input frequency.
- **`test_io.py`**: build a minimal DAS-shaped HDF5 with `h5py` →
  `make_dbfile` / `read_h5` round-trip recovers shape & timing; `save_npz` round-trip.
- Pure plotting functions (`viz`, `psd.plots`, `sound`) are not unit-tested.

## 5. README (bilingual, das-ancc structure)

Sections: Features · Install · Data layout (DAS h5 dir + db file + optional
`das.loc`) · Quick start (h5 / sac / npy models) · CLI reference table grouped as in
the original manual (input · save · filter · single-trace · imshow · fillplot · PSD
six-figure suite · sound) · Modules table · Algorithm notes (strain-rate conversion,
Welch PSD, single-trace spectrogram) · Running tests · License (MIT).

## 6. Dependencies

`numpy`, `scipy`, `obspy`, `pandas`, `h5py`, `matplotlib`, `tqdm`, `psutil`.
Python ≥ 3.10. Dev extra: `pytest`.
