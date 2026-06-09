"""Bayesian Hierarchical Poisson model for football goal prediction.

Replaces Dixon-Coles. Uses a log-linear additive model with Gaussian priors
(MAP estimation) — more numerically stable than the multiplicative MLE form
and never fails to converge the way Dixon-Coles does on sparse/unbalanced data.

Model (Baio & Blangiardo 2010):
    log(λ_h) = μ + att_h + def_a + home_adv * I(not neutral)
    log(λ_a) = μ + att_a + def_h

    home_goals_i ~ Poisson(λ_h)
    away_goals_i ~ Poisson(λ_a)

Priors (applied as L2 regularisation in MAP):
    att_t ~ N(0, σ_att²)   — shrinks weak/inactive teams toward league average
    def_t ~ N(0, σ_def²)
    Corner constraint: Σ att_t = 0, Σ def_t = 0  (identifiability)

Time decay: exp(-ln2 * days_ago / half_life) × match_weight multiplier.

References:
    Baio, G. & Blangiardo, M. (2010). Bayesian hierarchical model for the
    prediction of football results. Journal of Applied Statistics, 37(2), 253-264.

    Koopman, S.J. & Lit, R. (2015). A dynamic bivariate Poisson model for
    analysing and forecasting match results. JRSS-A, 178(1), 167-186.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)

_MAX_GOALS = 10


class BayesianPoissonModel:
    """MAP-estimated Bayesian Hierarchical Poisson (log-linear additive form).

    Attack/defense parameters are estimated with Gaussian priors so sparse
    or newly-added teams shrink toward the mean rather than overfitting.
    The log-linear parameterisation ensures L-BFGS-B converges reliably.
    """

    def __init__(
        self,
        half_life_days: int = 120,
        sigma_att: float = 1.0,
        sigma_def: float = 1.0,
    ) -> None:
        self.half_life_days = half_life_days
        self.sigma_att = sigma_att
        self.sigma_def = sigma_def
        self._params: Optional[dict] = None
        self._teams: list[str] = []

    def fit(self, matches: pd.DataFrame) -> "BayesianPoissonModel":
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"])
        ref = df["date"].max()

        mw = df.get("match_weight", pd.Series(1.0, index=df.index))
        days = (ref - df["date"]).dt.days.values.astype(float)
        w = np.exp(-math.log(2) * days / self.half_life_days) * mw.values

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        hi = np.array([idx[t] for t in df["home_team"]], dtype=np.int32)
        ai = np.array([idx[t] for t in df["away_team"]], dtype=np.int32)
        hg = df["home_goals"].values.astype(np.int32)
        ag = df["away_goals"].values.astype(np.int32)
        neutral = df.get("neutral", pd.Series(False, index=df.index)).values.astype(bool)
        not_neutral = (~neutral).astype(float)

        # Free parameters: [mu, home_adv, att_0..att_{n-2}, def_0..def_{n-2}]
        # Corner: att_{n-1} = -sum(att[0..n-2]), def_{n-1} = -sum(def[0..n-2])
        s2_att = self.sigma_att ** 2
        s2_def = self.sigma_def ** 2

        def objective(x: np.ndarray) -> float:
            mu, ha = x[0], x[1]
            att_free = x[2: 2 + n - 1]
            def_free = x[2 + n - 1:]

            att = np.append(att_free, -att_free.sum())
            dff = np.append(def_free, -def_free.sum())

            lam_h = np.exp(mu + att[hi] + dff[ai] + ha * not_neutral).clip(min=1e-6)
            lam_a = np.exp(mu + att[ai] + dff[hi]).clip(min=1e-6)

            nll = -np.dot(w, poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a))
            reg = (att ** 2).sum() / (2 * s2_att) + (dff ** 2).sum() / (2 * s2_def) + ha ** 2 / 0.5
            return nll + reg

        x0 = np.zeros(2 + 2 * (n - 1))
        x0[0] = math.log(1.3)
        x0[1] = 0.2

        res = minimize(objective, x0, method="L-BFGS-B",
                       options={"maxiter": 5000, "maxfun": 200_000, "ftol": 1e-9})
        if not res.success:
            logger.warning("BayesianPoisson MAP did not converge: %s", res.message)

        att_free = res.x[2: 2 + n - 1]
        def_free = res.x[2 + n - 1:]
        att = np.append(att_free, -att_free.sum())
        dff = np.append(def_free, -def_free.sum())

        self._params = {
            "mu": float(res.x[0]),
            "home_adv": float(res.x[1]),
            "attack": dict(zip(teams, att.tolist())),
            "defense": dict(zip(teams, dff.tolist())),
        }
        return self

    def _lambda(self, attacking: str, defending: str, home: bool) -> float:
        p = self._params
        return max(math.exp(
            p["mu"]
            + p["attack"].get(attacking, 0.0)
            + p["defense"].get(defending, 0.0)
            + (p["home_adv"] if home else 0.0)
        ), 1e-6)

    def predict_proba(self, home: str, away: str, neutral: bool = False) -> dict[str, float]:
        if hasattr(self, "_mcmc_trace") and self._mcmc_trace is not None:
            return self.predict_proba_mcmc(home, away, neutral=neutral)
        if self._params is None:
            raise RuntimeError("Model must be fit() before predict_proba().")

        lam_h = self._lambda(home, away, home=not neutral)
        lam_a = self._lambda(away, home, home=False)

        ph = pd = pa = 0.0
        for h in range(_MAX_GOALS + 1):
            for a in range(_MAX_GOALS + 1):
                p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
                if h > a:
                    ph += p
                elif h == a:
                    pd += p
                else:
                    pa += p

        total = max(ph + pd + pa, 1e-10)
        return {"home_win": ph / total, "draw": pd / total, "away_win": pa / total}

    def get_lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        """(lambda_home, lambda_away) for goal-count simulation."""
        if self._params is None:
            return 1.3, 1.0
        return (
            self._lambda(home, away, home=not neutral),
            self._lambda(away, home, home=False),
        )

    # ── Full MCMC posterior (PyMC) ────────────────────────────────────────────

    def fit_mcmc(
        self,
        matches: pd.DataFrame,
        draws: int = 1000,
        tune: int = 500,
        target_accept: float = 0.90,
        random_seed: int = 42,
    ) -> "BayesianPoissonModel":
        """Full MCMC posterior sampling via PyMC 5.

        Falls back silently to MAP (fit()) if PyMC is not installed or sampling
        fails.  After a successful MCMC run, predict_proba() uses the posterior
        predictive average over all samples rather than a single MAP point.

        References:
            Baio & Blangiardo (2010) — original hierarchical Poisson model
            Rue & Salvesen (2000)   — MCMC for football, JRSS-C
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError:
            logger.warning("PyMC not installed — falling back to MAP. Run: pip install pymc")
            return self.fit(matches)

        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"])
        ref = df["date"].max()

        mw = df.get("match_weight", pd.Series(1.0, index=df.index))
        days = (ref - df["date"]).dt.days.values.astype(float)
        w_arr = np.exp(-math.log(2) * days / self.half_life_days) * mw.values

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        hi = np.array([idx[t] for t in df["home_team"]], dtype=np.int32)
        ai = np.array([idx[t] for t in df["away_team"]], dtype=np.int32)
        hg = df["home_goals"].values.astype(np.int32)
        ag = df["away_goals"].values.astype(np.int32)
        neutral = df.get("neutral", pd.Series(False, index=df.index)).values.astype(bool)
        not_neutral = (~neutral).astype(float)

        logger.info("Fitting BayesianPoisson via MCMC (%d teams, %d matches, %d draws)...",
                    n, len(df), draws)

        try:
            with pm.Model() as pymc_model:  # noqa: F841
                # Non-centered parameterization avoids Neal's funnel (Betancourt & Girolami 2015)
                sigma_att = pm.HalfNormal("sigma_att", sigma=self.sigma_att)
                sigma_def = pm.HalfNormal("sigma_def", sigma=self.sigma_def)

                mu_g = pm.Normal("mu_global", mu=math.log(1.3), sigma=0.5)
                home_adv = pm.Normal("home_adv", mu=0.2, sigma=0.3)

                att_nc = pm.Normal("att_raw", mu=0.0, sigma=1.0, shape=n)
                def_nc = pm.Normal("def_raw", mu=0.0, sigma=1.0, shape=n)

                # Scale by hierarchical std; zero-sum constraint (identifiability)
                att_raw = att_nc * sigma_att
                def_raw = def_nc * sigma_def
                att = att_raw - pt.mean(att_raw)
                dff = def_raw - pt.mean(def_raw)

                lam_h = pt.exp(mu_g + att[hi] + dff[ai] + home_adv * not_neutral)
                lam_a = pt.exp(mu_g + att[ai] + dff[hi])

                # Weighted log-likelihood via Potential (time-decay weights)
                log_lik = (
                    pm.logp(pm.Poisson.dist(lam_h), hg)
                    + pm.logp(pm.Poisson.dist(lam_a), ag)
                )
                pm.Potential("weighted_loglik", (w_arr * log_lik).sum())

                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    target_accept=target_accept,
                    random_seed=random_seed,
                    progressbar=True,
                    return_inferencedata=True,
                )

            # att_raw/def_raw in trace are non-centered (unit Normal); scale back
            att_nc_post = trace.posterior["att_raw"].values   # (chains, draws, n)
            def_nc_post = trace.posterior["def_raw"].values
            sig_att_post = trace.posterior["sigma_att"].values[..., None]  # (chains, draws, 1)
            sig_def_post = trace.posterior["sigma_def"].values[..., None]
            mu_post = trace.posterior["mu_global"].values
            ha_post = trace.posterior["home_adv"].values

            att_post = att_nc_post * sig_att_post   # scaled attack strengths
            def_post = def_nc_post * sig_def_post   # scaled defense strengths

            att_flat = att_post.reshape(-1, n)
            def_flat = def_post.reshape(-1, n)
            att_mean = att_flat.mean(axis=0)
            def_mean = def_flat.mean(axis=0)
            att_mean -= att_mean.mean()
            def_mean -= def_mean.mean()

            self._params = {
                "mu": float(mu_post.mean()),
                "home_adv": float(ha_post.mean()),
                "attack": dict(zip(teams, att_mean.tolist())),
                "defense": dict(zip(teams, def_mean.tolist())),
            }

            # Store full scaled posterior for predict_proba_mcmc()
            self._mcmc_trace = {
                "att": att_flat,
                "dff": def_flat,
                "mu": mu_post.flatten(),
                "ha": ha_post.flatten(),
                "teams": teams,
                "team_idx": idx,
            }
            logger.info("MCMC sampling complete. R-hat: check with arviz.plot_posterior(trace).")

        except Exception as exc:
            logger.warning("MCMC sampling failed (%s) — falling back to MAP.", exc)
            return self.fit(matches)

        return self

    def predict_proba_mcmc(
        self, home: str, away: str, neutral: bool = False, n_samples: int = 500
    ) -> dict[str, float]:
        """Average win/draw/loss probabilities over MCMC posterior samples.

        Uses Monte Carlo integration over the posterior: for each MCMC sample
        s, compute (lambda_h^s, lambda_a^s), evaluate P(H>A), P(H=A), P(H<A)
        via truncated Poisson PMF, then average across samples.  This propagates
        full parameter uncertainty into the probability estimates.
        """
        if not hasattr(self, "_mcmc_trace") or self._mcmc_trace is None:
            return self.predict_proba(home, away, neutral)

        tr = self._mcmc_trace
        idx = tr["team_idx"]
        n_teams = len(tr["teams"])
        n_s = min(n_samples, len(tr["mu"]))
        s_idx = np.random.choice(len(tr["mu"]), size=n_s, replace=False)

        hi_i = idx.get(home, 0)
        ai_i = idx.get(away, 0)
        ha_flag = 0.0 if neutral else 1.0

        att_s = tr["att"][s_idx]
        dff_s = tr["dff"][s_idx]
        mu_s = tr["mu"][s_idx]
        ha_s = tr["ha"][s_idx]

        att_s -= att_s.mean(axis=1, keepdims=True)
        dff_s -= dff_s.mean(axis=1, keepdims=True)

        lam_h_s = np.exp(mu_s + att_s[:, hi_i] + dff_s[:, ai_i] + ha_s * ha_flag)
        lam_a_s = np.exp(mu_s + att_s[:, ai_i] + dff_s[:, hi_i])

        goals = np.arange(_MAX_GOALS + 1)
        ph_acc = pd_acc = pa_acc = 0.0
        for lh, la in zip(lam_h_s, lam_a_s):
            p_h = poisson.pmf(goals, float(lh))
            p_a = poisson.pmf(goals, float(la))
            mat = np.outer(p_h, p_a)
            sample_ph = float(np.triu(mat, k=1).sum())
            sample_pd = float(np.diag(mat).sum())
            sample_pa = float(np.tril(mat, k=-1).sum())
            total_s = max(sample_ph + sample_pd + sample_pa, 1e-10)
            ph_acc += sample_ph / total_s
            pd_acc += sample_pd / total_s
            pa_acc += sample_pa / total_s

        return {
            "home_win": ph_acc / n_s,
            "draw": pd_acc / n_s,
            "away_win": pa_acc / n_s,
        }
