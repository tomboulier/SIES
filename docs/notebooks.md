# Notebooks

The `notebooks/` directory contains interactive [Marimo](https://marimo.io)
notebooks. Unlike Jupyter notebooks, Marimo notebooks are **pure Python
files**: they are versionable, reproducible (no hidden state, reactive
execution), and runnable as scripts.

## Running the notebooks

```bash
pip install -e .[dev]          # installs marimo
marimo edit notebooks/dictionary_matching.py   # interactive editing
marimo run notebooks/dictionary_matching.py    # read-only app mode
python notebooks/dictionary_matching.py        # plain script execution
```

## Available notebooks

### `cgpt_pipeline.py`

The forward and inverse problem step by step: build a shape, place an
acquisition system, simulate the MSR matrix, reconstruct the CGPT and compare
it with the theoretical value and with the closed-form ellipse formulas.

### `dictionary_matching.py`

The complete identification experiment of the FoCM 2014 paper: a dictionary
of shapes, an unknown (transformed) target, noisy measurements, and
identification by invariant shape descriptors, including a study of the
robustness to noise.

![Matching results](assets/matching_results.png)

### `target_tracking.py`

Tracking of a moving target with the Extended Kalman Filter: simulate a
random trajectory, generate the stream of noisy MSR matrices, and estimate
position and orientation online.

![Tracking](assets/tracking.png)
