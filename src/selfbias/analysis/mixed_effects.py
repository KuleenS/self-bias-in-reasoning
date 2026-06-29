"""The self-bias mixed-effects estimator.

Fits the headline model

    response ~ is_self  +  (1 | generator) + (1 | evaluator) + (1 | dataset) + (1 | prompt)

`method="bayes"` (default) fits a logistic crossed-random-effects GLMM via statsmodels'
`BinomialBayesMixedGLM` (pure Python — no R/lme4 needed). `method="lpm"` fits the fast linear
prob. model with a prompt random intercept (the original notebook model) for a frequentist
p-value. `per_evaluator_breakdown` reproduces the Holm-corrected per-evaluator analysis.

Grouping factors with <2 levels (e.g. a single dataset) are dropped automatically.
"""

from __future__ import annotations

import math

import pandas as pd

_GROUP_COLS = {"generator": "generator", "evaluator": "evaluator",
               "dataset": "dataset", "prompt": "prompt_id"}


def _usable_groups(df: pd.DataFrame) -> list[str]:
    return [g for g, col in _GROUP_COLS.items() if col in df and df[col].nunique() >= 2]


def fit_bayes_glmm(df: pd.DataFrame, groups: list[str] | None = None) -> dict:
    from scipy.stats import norm
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    groups = groups or _usable_groups(df)
    vc = {g: f"0 + C({_GROUP_COLS[g]})" for g in groups}
    model = BinomialBayesMixedGLM.from_formula("response ~ is_self", vc, df)
    res = model.fit_vb()
    names = list(res.model.exog_names)
    i = names.index("is_self")
    beta, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
    z = beta / sd if sd else float("nan")
    return {
        "method": "bayes_glmm",
        "beta_self": beta,
        "sd": sd,
        "odds_ratio": math.exp(beta),
        "p_gt_0": float(norm.cdf(z)),
        "p_two_sided": float(2 * norm.sf(abs(z))),
        "random_effects": groups,
        "n_obs": int(len(df)),
    }


def fit_lpm_prompt(df: pd.DataFrame) -> dict:
    from scipy.stats import norm
    from statsmodels.formula.api import mixedlm

    res = mixedlm("response ~ is_self", df, groups=df["prompt_id"]).fit(reml=False)
    beta, se = float(res.params["is_self"]), float(res.bse["is_self"])
    z = beta / se if se else float("nan")
    return {
        "method": "lpm_prompt",
        "beta_self": beta,
        "se": se,
        "z": z,
        "p_one_sided": float(norm.sf(z)),  # H1: beta > 0
        "p_two_sided": float(2 * norm.sf(abs(z))),
        "n_obs": int(len(df)),
    }


def per_evaluator_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import norm
    from statsmodels.formula.api import mixedlm
    from statsmodels.stats.multitest import multipletests

    rows, pvals = [], []
    for ev in sorted(df["evaluator"].unique()):
        sub = df[df["evaluator"] == ev]
        if sub["is_self"].nunique() < 2:
            continue
        res = mixedlm("response ~ is_self", sub, groups=sub["prompt_id"]).fit(reml=False)
        beta, se = float(res.params["is_self"]), float(res.bse["is_self"])
        z = beta / se if se else float("nan")
        p = float(norm.sf(z))
        rows.append({"evaluator": ev, "beta_self": beta, "se": se, "z": z, "p_one_sided": p})
        pvals.append(p)
    out = pd.DataFrame(rows)
    if pvals:
        reject, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        out["p_holm"] = p_holm
        out["reject"] = reject
    return out


def fit_self_bias(df: pd.DataFrame, method: str = "bayes") -> dict:
    if df.empty:
        raise ValueError("empty judgments table")
    if df["is_self"].nunique() < 2:
        raise ValueError("no variation in is_self — need both self and cross cells")
    if method == "bayes":
        return fit_bayes_glmm(df)
    if method == "lpm":
        return fit_lpm_prompt(df)
    raise ValueError(f"unknown method '{method}' (use 'bayes' or 'lpm')")
