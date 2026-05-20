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

### Example

To run the ODE fitting with the default configuration:

```bash
kefir-ode-fit --config configs/default_args.json
```
