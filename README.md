# Water Kefir Neural ODE & SDE Models

This package provides tools for fitting Neural Ordinary Differential Equations (ODE) and simulating Stochastic Differential Equations (SDE) to analyze water kefir fermentation data.

## Installation

You can install this package locally for development by running:

```bash
pip install -e .
```

## Usage

After installation, the following command-line tools will be available:

- `kefir-ode-fit`
- `kefir-sde-compare`
- `kefir-logistic-pinn-compare`

### Example

To run the ODE fitting with the default configuration:

```bash
kefir-ode-fit --config configs/default_args.json
```

To fit the raw kefir trials with deterministic and stochastic logistic PINN
inverse models and generate comparison plots:

```bash
kefir-logistic-pinn-compare --config logistic_pinn_config_reference.json
```
