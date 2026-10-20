"""Estimators and experiments for quantum power-trace estimation."""
from concurrent.futures import ProcessPoolExecutor
import csv
import math

import numpy as np

from sampling import (prefix_shapes, counted_prefix_shapes, streamed_prefix_shapes,
                      estimator_parameters, word_from_uniforms, target)


def rng(seed, *coordinates):
    return np.random.default_rng(np.random.SeedSequence([seed, *coordinates]))


def budget_grid(maximum, extra=()):
    small = list(range(1, min(maximum, 32) + 1))
    powers = np.arange(1, 1 + math.ceil(math.log(maximum / 32, 1.08))) if maximum > 32 else []
    large = np.ceil(32 * 1.08 ** np.asarray(powers)).astype(int)
    return np.array(sorted(set(small + [int(n) for n in large if n <= maximum] + [maximum] + list(extra))))


def make_spectra(config, family):
    d, name = family["d"], family["family"]
    if family["random"]:
        spectra = []
        for state in range(config["states"]):
            generator = rng(config["seed"], 10, family["stream_id"], d, state)
            g = generator.standard_normal((d, d)) + 1j * generator.standard_normal((d, d))
            values = np.maximum(np.linalg.eigvalsh(g @ g.conj().T)[::-1], 0)
            spectra.append(values / values.sum())
        return np.asarray(spectra)
    values = np.ones(d) if name == "uniform32" else 0.85 ** np.arange(d)
    return (values / values.sum())[None, :]


def sample_budget(config, family, n):
    spectra = make_spectra(config, family)
    trials = config["trials"] * (1 if family["random"] else config["states"])
    k, d = config["groups"], family["d"]
    blocks = np.zeros((len(spectra), trials, k, d), dtype=np.int32)
    for state, spectrum in enumerate(spectra):
        generator = rng(config["seed"], 11, family["stream_id"], n, state)
        words = generator.choice(d, size=(trials, n * k), p=spectrum / spectrum.sum())
        for trial in range(trials):
            for group in range(k):
                shape = prefix_shapes(words[trial, group*n:(group+1)*n], d, np.array([n]), d)[0]
                blocks[state, trial, group, :len(shape)] = shape
    return spectra, blocks


def budget_statistics(config, family, n, spectra, blocks):
    """Paired mean absolute errors and hierarchical bootstrap intervals."""
    qs = np.asarray(config["qs"])
    medians = np.median(blocks, axis=2) / n  # Coordinate medians are not renormalized.
    cumulative = np.cumsum(medians[..., None] ** qs, axis=2)
    m = min(family["d"], math.ceil(config["c"] * math.sqrt(n)))
    estimates = np.stack((cumulative[:, :, -1], cumulative[:, :, m-1]), axis=2)
    truth = (spectra[..., None] ** qs).sum(axis=1)
    losses = np.abs(estimates - truth[:, None, None, :])
    states, trials = losses.shape[:2]
    generator = rng(config["seed"], 12, family["stream_id"], n)
    boot = np.empty((config["bootstrap_replicates"], 2, len(qs)))
    # Resample states, then trials within states, sharing indices across estimators.
    for start in range(0, len(boot), 16):
        count = min(16, len(boot) - start)
        si = generator.integers(states, size=(count, states, 1))
        ti = generator.integers(trials, size=(count, states, trials))
        boot[start:start+count] = losses[si, ti].mean(axis=(1, 2))
    lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
    mae = losses.mean(axis=(0, 1))
    return [dict(family=family["family"], n=n, N=n*config["groups"], q=float(q),
                 method=method, mae=float(mae[mi, qi]),
                 ci_lower=float(lower[mi, qi]), ci_upper=float(upper[mi, qi]))
            for qi, q in enumerate(qs) for mi, method in enumerate(("full", "truncated"))]


def budget_job(args):
    config, family, n = args
    spectra, blocks = sample_budget(config, family, n)
    rows = budget_statistics(config, family, n, spectra, blocks)
    print(f"Figure 1: {family['family']}, n={n}", flush=True)
    return rows


def sample_trial(config, curve, sizes, trial):
    """Independent Schur samples, coupled across budgets by word prefixes."""
    q, d, family = curve["q"], curve["d"], curve["family"]
    m, k = estimator_parameters(q, config["epsilon"], d)
    shapes = []
    for group in range(k):
        generator = rng(config["seed"], curve["curve_id"], trial, group)
        if q < 1 and sizes[-1] >= 2**20:
            shape = streamed_prefix_shapes(generator, family, d, sizes, m)
        else:
            word = word_from_uniforms(generator.random(int(sizes[-1])), family, d)
            sampler = counted_prefix_shapes if d <= 2048 and sizes[-1] >= 4096 else prefix_shapes
            shape = sampler(word, d, sizes, m)
        shapes.append(shape)
    width = max(shape.shape[-1] for shape in shapes)
    blocks = np.zeros((k, len(sizes), width), dtype=np.int64)
    for group, shape in enumerate(shapes):
        blocks[group, :, :shape.shape[-1]] = shape
    return ((np.median(blocks, axis=0) / sizes[:, None]) ** q).sum(axis=1)


def dimension_job(args):
    config, curve = args
    q, d, family = curve["q"], curve["d"], curve["family"]
    sizes = budget_grid(config["maximum_n"], config["extra_budgets"])
    sizes = sizes[sizes <= curve["max_n"]]
    estimates = np.array([sample_trial(config, curve, sizes, trial) for trial in range(config["trials"])])
    success = (np.abs(estimates - target(family, d, q)) <= config["epsilon"]).mean(axis=0)
    hits = np.flatnonzero(success >= config["success_probability"])
    _, k = estimator_parameters(q, config["epsilon"], d)
    total = int(sizes[hits[0]] * k) if len(hits) else ""
    print(f"Figure 2: {family}, d={d}, q={q}, N={total or 'not reached'}", flush=True)
    return dict(family=family, d=d, q=q, N_empirical=total)


def run_experiments(configs, output, workers=1, figure="all"):
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if figure in ("all", "1"):
            config = configs["figure1"]
            jobs = [(config, family, n) for family in config["families"] for n in config["group_sizes"]]
            results["1"] = [row for rows in pool.map(budget_job, jobs) for row in rows]
        if figure in ("all", "2"):
            jobs = [(config, curve) for config in configs["figure2"].values() for curve in config["curves"]]
            results["2"] = list(pool.map(dimension_job, jobs))
    for number, rows in results.items():
        with (output / f"figure{number}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return results
