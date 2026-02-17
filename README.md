# Sequential Indicator Simulation

High-performance 3D sequential indicator simulation with Gaussian variogram for categorical spatial data.

## Features

- **Sequential indicator kriging (SIK)** on 3D grids
- **Anisotropic Gaussian variogram** with per-axis ranges
- **Anisotropic search radii** for layer-like simulations
- **Fast nearest-neighbor interpolation** using KDTree
- **Parallel execution** across multiple cores
- **Compressed NPZ output** for efficient storage
- **Multiple realizations** with independent random seeds

## Installation

Install from GitHub:

```bash
git clone https://github.com/timcjohnson/sequential-indicator-sim.git
cd sequential-indicator-sim
pip install -r requirements.txt
```

## Usage

### Basic example (unconditional):

```bash
python src/sequential_indicator_sim.py \
  --output simulation \
  --x-start 0 --x-end 10 --x-num 50 \
  --y-start 0 --y-end 10 --y-num 50 \
  --z-start 0 --z-end 5 --z-num 25 \
  --vrange-x 3 --vrange-y 3 --vrange-z 0.5 \
  --categories "0,1,2"
```

### With known points and interpolation:

```bash
python src/sequential_indicator_sim.py \
  --known known_points.txt \
  --output result \
  --x-start 0 --x-end 1.9304 --x-num 52 \
  --y-start 0 --y-end 0.7366 --y-num 28 \
  --z-start 0 --z-end 0.9652 --z-num 40 \
  --vrange-x 2 --vrange-y 2 --vrange-z 0.1 \
  --interp-x-num 104 --interp-y-num 56 --interp-z-num 80 \
  --num-realizations 10 --num-cores 4 --seed 123
```

### Input file format

`--known` file: whitespace-delimited text with columns `x y z category`

```
0.5 0.3 0.1 1
1.2 0.5 0.2 2
2.0 1.0 0.3 1
```

## Command-line Options

### Grid definition (required)

- `--x-start, --x-end, --x-num`: X dimension extent and point count
- `--y-start, --y-end, --y-num`: Y dimension extent and point count
- `--z-start, --z-end, --z-num`: Z dimension extent and point count

### Variogram (required)

- `--vrange-x, --vrange-y, --vrange-z`: Gaussian range per axis
- `--nugget`: Nugget effect (default: 0.0)

### Search (optional)

- `--max-neighbors`: Maximum neighbors for kriging (default: 12)
- `--search-radius-x/y/z`: Anisotropic search radii (default: unlimited)

### Interpolation (optional)

- `--interp-x-num, --interp-y-num, --interp-z-num`: Finer output grid

### Realizations

- `--num-realizations`: Number of independent realizations (default: 1)
- `--num-cores`: CPU cores for parallelization (default: all available)
- `--seed`: Random seed for reproducibility

### I/O

- `--known`: Path to known points file (optional)
- `--output`: Output file prefix (without `.npz` extension)
- `--categories`: Comma-separated category labels (auto-detected from known points if omitted)

## Output

Saves compressed NPZ files with datasets:

```python
import numpy as np
data = np.load('result.npz')
x, y, z, cat = data['x'], data['y'], data['z'], data['category']
```

For multiple realizations: `result_1.npz`, `result_2.npz`, ...

## Performance

- KDTree-based interpolation: **100-1000× faster** than naive loops
- Parallel realizations: **Linear scaling** up to core count
- Precomputed interpolation indices: **Reusable** across all realizations

## Algorithm

1. **Load** conditioning data (optional)
2. **Compute** interpolation mapping once (if upsampling)
3. For each realization:
   - Visit grid nodes in **random order** (sequential step)
   - **Indicator kriging** with nearby conditioning data
   - **Stochastic sampling** from posterior probabilities
   - Add simulated point to conditioning dataset
4. **Interpolate** and **save** results

## References

- Goovaerts, P. (1997). Geostatistics for Natural Resources Evaluation.
- Journel, A. G., & Huijbregts, C. J. (1978). Mining Geostatistics.

## License

MIT - See LICENSE file

## Contributing

Pull requests welcome. Please open issues for bugs and feature requests.
