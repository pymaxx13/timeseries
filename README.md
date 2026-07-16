# Engression-TS

A modular framework for probabilistic time series forecasting by integrating **Engression** with deep learning forecasting models from **Darts**.

## Overview

Engression-TS extends deterministic deep learning forecasting models with a unified probabilistic forecasting framework based on **Engression**.

Instead of predicting only a single point forecast, Engression-TS learns the **entire conditional predictive distribution** by injecting stochastic noise into deep forecasting architectures and optimizing them using the **Energy Score**, enabling sample-based probabilistic forecasting.

The project is built on top of the excellent [Darts](https://github.com/unit8co/darts) forecasting library while preserving the original model architectures.


---

## Current Features

- Modular Engression training framework
- Shared probabilistic training pipeline
- Energy Score optimization
- Sample-based probabilistic forecasting
- Configurable noise injection framework
- Extensible noise registry
- Reusable base module for future models

---

## Model Implementation Status

| Model | Status |
|-------|:------:|
| EnHiTS | ✅ Implemented |
| EnBEATS | ✅ Implemented |
| EnTCN | ✅ Implemented |
| EnTFT | ✅ Implemented |
| EnRNN | ⏳ WIP |
| EnBlockRNN | ⏳ WIP |
| EnTransformer | ⏳ WIP |
| EnTiDE | ⏳ WIP |
| EnTSMixer | ⏳ WIP |
| EnDLinear | ⏳ WIP |
| EnNLinear | ⏳ WIP |
| EnChronos | ⏳ WIP |
| EnTimesFM | ⏳ WIP |

### Implemented models currently support:

- Probabilistic forecasting
- Multiple forecast sampling
- Energy Score optimization
- Configurable noise injection
- Shared Engression training framework

---

## Project Structure

```text
engressionts/
│
├── base/
│   └── base_engression.py
│
├── losses/
│   └── energy_score.py
│
├── noise/
│   ├── gaussian.py
│   ├── uniform.py
│   └── __init__.py
│
├── models/
│   ├── nhits.py
│   ├── nbeats.py
│   ├── tcn_model.py
│   ├── tft_model.py
│   └── ...
│
├── metrics/
├── sampling/
├── calibration/
└── experiments/
```

---

## Architecture

Each forecasting model keeps its original Darts backbone while replacing the deterministic training objective with the Engression framework.

```
Historical Series
        │
        ▼
 Noise Injection
        │
        ▼
 Original Darts Model
        │
        ▼
 Multiple Forecast Samples
        │
        ▼
 Energy Score Loss
        │
        ▼
 Probabilistic Forecast
```


---

## Design Philosophy

The implementation is designed around three principles:

- Preserve the original Darts model architectures.
- Share all Engression-specific functionality through reusable base classes.
- Make extending new forecasting models require minimal code changes.


---

## Acknowledgements

This project builds upon:

- Darts — Unified Time Series Forecasting Library
- Engression — Distribution-free probabilistic regression via stochastic neural networks
