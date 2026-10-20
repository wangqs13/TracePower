# Trace estimation of quantum state powers

Numerical experiments illustrating the theoretical results in
[Trace Estimation of Quantum State Powers: Sample Complexity and Computational Hardness](https://arxiv.org/pdf/2505.09563).
Weak Schur sampling is simulated using the Robinson–Schensted–Knuth correspondence.

## Run

Python 3.10–3.12.

```bash
python3 -m pip install -r requirements.txt
./run.sh --workers 8
```

Use `--figure 1` or `--figure 2` to run one experiment.
Results are saved as CSV tables and PDF figures in `outputs/`.

## Experiments

**Figure 1: effect of truncation.** Compare full and truncated power sums of the
same coordinate-median spectrum at `d=32`, with `q=1.25,1.5,2.5,3.5`. Spectra are
random `GG†/tr(GG†)` (complex Gaussian G), uniform, and geometric with ratio `0.85`.
Each trial uses five independent Young diagrams, `N=5n` total copies, and
`m=min(d,ceil(sqrt(n)/2))` retained coordinates. Medians are not renormalized.
There are 33 budgets: 31 rounded logarithmically spaced values from 64 to 2048,
plus 255 and 257. We use 12 random states × 192 trials, or 2304 trials for each
fixed spectrum. Mean absolute errors have 95% bootstrap bands from 1000 resamples.

**Figure 2: sample requirements versus dimension.** For uniform spectra and a
half-mass head with uniform tail, report the smallest tested total-copy budget
with success fraction at least `2/3`, where success means absolute error ≤ `0.1`.

| q | Dimensions | Trials |
| --- | --- | --- |
| 0.5 | 128, 256, …, 2048 | 64 |
| 1.5, 2.5 | Powers of two from 8 to 1024; then 2048, 4096, …, 65536 | 512 |

For `q=0.5`, use one normalized Young diagram and all coordinates (`k=1,m=d`).
For `q=1.5` and `q=2.5`, use `m=min(d,2500)` and `m=min(d,55)`, respectively;
`k` is the smallest odd integer at least `ln(6m)`, and `N=kn`.
Fixed budget ranges and random seeds are specified in `configs/`.

## Acknowledgment

The code was developed with the assistance of Codex (ChatGPT-6 Astra).
