#!/usr/bin/env python3

import numpy as np
import pandas as pd
import re


# ============================================================================
# Some helper functions to explore priors and niche space
# ============================================================================

def _parse_args(argstr):
    return [float(a.strip()) for a in argstr.split(",") if a.strip()]

def sample_from_spec(spec, size=20000):
    # spec can be a string like "uniform(0,1)" or a dict
    if isinstance(spec, str):
        m = re.match(r"(\w+)\s*\(([^)]*)\)", spec)
        if m:
            name, args = m.group(1).lower(), _parse_args(m.group(2))
        else:
            # fallback: try comma-separated numbers -> treat as choices
            parts = [s.strip() for s in spec.split(",")]
            try:
                vals = [float(p) for p in parts]
                return np.random.choice(vals, size=size)
            except Exception:
                raise ValueError(f"Can't parse prior spec string: {spec}")
    elif isinstance(spec, dict):
        name = (spec.get("type") or spec.get("dist") or spec.get("distribution") or "").lower()
        args = None
        # common keys mapping
        if not name:
            if "lower" in spec and "upper" in spec:
                name = "uniform"
                args = [spec["lower"], spec["upper"]]
            elif "choices" in spec or "values" in spec:
                vals = spec.get("choices", spec.get("values"))
                return np.random.choice(vals, size=size)
        else:
            # collect numeric args if present
            if name in ("uniform", "unif"):
                lo = spec.get("lower", spec.get("low", spec.get("a")))
                hi = spec.get("upper", spec.get("high", spec.get("b")))
                args = [lo, hi]
            elif name in ("normal", "gaussian", "norm"):
                mu = spec.get("mu", spec.get("mean", 0))
                sigma = spec.get("sigma", spec.get("sd", spec.get("std", 1)))
                args = [mu, sigma]
            elif name in ("loguniform", "log-uniform", "log_uniform", "uniform_log"):
                lo = spec.get("lower", spec.get("low"))
                hi = spec.get("upper", spec.get("high"))
                args = [lo, hi]
            elif name in ("choice", "categorical"):
                vals = spec.get("choices", spec.get("values"))
                return np.random.choice(vals, size=size)
    else:
        raise ValueError(f"Unsupported prior spec type: {type(spec)}")

    if args is None:
        args = args or []

    if name in ("uniform", "unif"):
        a, b = args
        return np.random.uniform(a, b, size=size)
    if name in ("normal", "gaussian", "norm"):
        mu, sigma = args
        return np.random.normal(mu, sigma, size=size)
    if name in ("loguniform", "log-uniform", "log_uniform", "uniform_log"):
        lo, hi = args
        lo, hi = float(lo), float(hi)
        return np.exp(np.random.uniform(np.log(lo), np.log(hi), size=size))
    if name in ("exponential", "exp"):
        rate = args[0]
        return np.random.exponential(1.0 / rate, size=size)
    if name in ("beta",):
        a, b = args
        return np.random.beta(a, b, size=size)
    if name in ("gamma",):
        shape, scale = args
        return np.random.gamma(shape, scale, size=size)

    raise ValueError(f"Unknown distribution type: {name}")


def plot_gamma(shape, scale):
    
    import matplotlib.pyplot as plt
    from scipy.stats import gamma


    x = np.linspace(0, gamma.ppf(0.99, shape, scale=scale), 1000)
    y = gamma.pdf(x, shape, scale=scale)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2, color='#2b8cbe')
    plt.fill_between(x, y, alpha=0.3, color='#2b8cbe')
    plt.title(f'Gamma Distribution (shape={shape:.4f}, scale={scale:.6f})', fontsize=14)
    plt.xlabel('Fitness effect')
    plt.ylabel('Probability density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()