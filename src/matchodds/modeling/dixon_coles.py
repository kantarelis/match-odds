"""The Dixon-Coles bivariate-Poisson goals model — the domain-canonical generative approach.

Where the discriminative models (logistic regression, XGBoost) map engineered features straight to
1X2 probabilities, Dixon-Coles models the *score line*: home and away goals as Poisson counts whose
rates come from per-team **attack** / **defense** strengths plus a **home-advantage** term, with the
Dixon-Coles low-score correction (``rho`` / ``tau``) for the dependence between the two counts in the
0-0 / 1-0 / 0-1 / 1-1 games a plain double-Poisson misprices. The 1X2 probabilities are read off the
resulting score matrix (summed to ``settings.dc_max_goals``).

It is fit **per league** by maximum likelihood (``scipy.optimize.minimize``, L-BFGS-B) on full-time
goals — the Task-6 label-side passthrough — optionally exponentially time-decayed so recent matches
count for more (``settings.dc_half_life_days``). Predicting needs only identifiers, so it serves from
the same ``{home, away, league, date}`` request the discriminative models do; an unseen team falls
back to its league's average strength, an unseen league to the global average. Being generative it is
naturally calibrated and is reliability-*checked* rather than calibration-*wrapped* (PLAN Decision 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from matchodds.config import settings
from matchodds.modeling import base, metrics

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

# The attack/defense ridge (adding c to every attack and subtracting it from every defense leaves the
# rates unchanged) is pinned by re-centring attack to mean 0 after the fit; HOME_ADV_INIT seeds the
# optimiser with a mild positive home edge, and rho is bounded so the tau correction stays sane.
_HOME_ADV_INIT = 0.1
_RHO_BOUND = 0.99
_MIN_TAU = 1e-10


@dataclass(frozen=True, slots=True)
class _LeagueFit:
    """MLE parameters for one league, plus the averages used as the unseen-team fallback."""

    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    avg_attack: float
    avg_defense: float

    def strength(self, team: str) -> tuple[float, float]:
        """``(attack, defense)`` for a team, falling back to the league average if it is unseen."""
        return self.attack.get(team, self.avg_attack), self.defense.get(team, self.avg_defense)


def _tau(home_goals: IntArray, away_goals: IntArray, lam: FloatArray, mu: FloatArray, rho: float) -> FloatArray:
    """The Dixon-Coles low-score correction, applied to the four dependent cells (else 1)."""
    tau = np.ones_like(lam)
    tau = np.where((home_goals == 0) & (away_goals == 0), 1.0 - lam * mu * rho, tau)
    tau = np.where((home_goals == 0) & (away_goals == 1), 1.0 + lam * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 0), 1.0 + mu * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 1), 1.0 - rho, tau)
    return np.asarray(tau, dtype=np.float64)


def _poisson_pmf(counts: IntArray, rate: float) -> FloatArray:
    """Poisson probability mass for ``counts`` at the given ``rate`` (computed in log space)."""
    log_pmf = counts * np.log(rate) - rate - gammaln(counts + 1)
    return np.asarray(np.exp(log_pmf), dtype=np.float64)


def _fit_league(
    home_idx: IntArray,
    away_idx: IntArray,
    home_goals: IntArray,
    away_goals: IntArray,
    weights: FloatArray,
    teams: list[str],
) -> _LeagueFit:
    """Maximise the (optionally time-decayed) Dixon-Coles log-likelihood for one league."""
    n = len(teams)
    log_factorial = gammaln(home_goals + 1) + gammaln(away_goals + 1)

    def neg_log_likelihood(params: FloatArray) -> float:
        attack = params[:n]
        defense = params[n : 2 * n]
        home_adv = float(params[2 * n])
        rho = float(params[2 * n + 1])
        log_lam = attack[home_idx] + defense[away_idx] + home_adv
        log_mu = attack[away_idx] + defense[home_idx]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)
        log_poisson = home_goals * log_lam - lam + away_goals * log_mu - mu - log_factorial
        log_tau = np.log(np.clip(_tau(home_goals, away_goals, lam, mu, rho), _MIN_TAU, None))
        return -float(np.sum(weights * (log_poisson + log_tau)))

    init = np.zeros(2 * n + 2, dtype=np.float64)
    init[2 * n] = _HOME_ADV_INIT
    bounds = [(None, None)] * (2 * n + 1) + [(-_RHO_BOUND, _RHO_BOUND)]
    result = minimize(neg_log_likelihood, init, method="L-BFGS-B", bounds=bounds)

    params = np.asarray(result.x, dtype=np.float64)
    # Re-centre attack to mean 0 (rates are invariant to the shift): keeps the parameters
    # interpretable and makes the league-average fallback a true neutral team.
    shift = float(params[:n].mean())
    attack = params[:n] - shift
    defense = params[n : 2 * n] + shift
    attack_map = {team: float(value) for team, value in zip(teams, attack)}
    defense_map = {team: float(value) for team, value in zip(teams, defense)}
    return _LeagueFit(
        attack=attack_map,
        defense=defense_map,
        home_adv=float(params[2 * n]),
        rho=float(params[2 * n + 1]),
        avg_attack=float(np.mean(list(attack_map.values()))),
        avg_defense=float(np.mean(list(defense_map.values()))),
    )


def _outcome_probabilities(lam: float, mu: float, rho: float, max_goals: int) -> list[float]:
    """``[home_win, draw, away_win]`` summed from the corrected score matrix, normalised to 1."""
    goals = np.arange(max_goals + 1, dtype=np.intp)
    matrix = np.outer(_poisson_pmf(goals, lam), _poisson_pmf(goals, mu))
    matrix[0, 0] *= 1.0 - lam * mu * rho
    matrix[0, 1] *= 1.0 + lam * rho
    matrix[1, 0] *= 1.0 + mu * rho
    matrix[1, 1] *= 1.0 - rho
    matrix = np.clip(matrix, 0.0, None)  # the corrected cells can dip negative; clip before summing
    home_win = float(np.tril(matrix, -1).sum())  # home goals > away goals
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())  # away goals > home goals
    total = home_win + draw + away_win
    return [home_win / total, draw / total, away_win / total]


class DixonColesModel(base.OutcomeModel):
    """Per-league bivariate-Poisson goals model fit by MLE; 1X2 read from the score matrix."""

    def __init__(self) -> None:
        self.max_goals = settings.dc_max_goals
        self.half_life_days = settings.dc_half_life_days
        self._leagues: dict[str, _LeagueFit] = {}
        # Overwritten in fit(); a neutral default keeps predict() total before a fit from erroring.
        self._global = _LeagueFit({}, {}, _HOME_ADV_INIT, 0.0, 0.0, 0.0)

    def fit(self, table: pd.DataFrame, /) -> DixonColesModel:
        league_col = table["league"].astype(str)
        self._leagues = {league: self._fit_one(table[league_col == league]) for league in sorted(league_col.unique())}
        self._global = self._global_fallback()
        return self

    def _fit_one(self, group: pd.DataFrame) -> _LeagueFit:
        home = group["home"].astype(str).to_numpy()
        away = group["away"].astype(str).to_numpy()
        teams = sorted(set(home) | set(away))
        index = {team: position for position, team in enumerate(teams)}
        home_idx = np.asarray([index[team] for team in home], dtype=np.intp)
        away_idx = np.asarray([index[team] for team in away], dtype=np.intp)
        home_goals = group["ft_home_goals"].to_numpy(dtype=np.intp)
        away_goals = group["ft_away_goals"].to_numpy(dtype=np.intp)
        return _fit_league(home_idx, away_idx, home_goals, away_goals, self._time_weights(group["date"]), teams)

    def _time_weights(self, dates: pd.Series) -> FloatArray:
        """Exponential recency weights (``0.5`` per half-life relative to the latest match), else 1."""
        if self.half_life_days is None:
            return np.ones(len(dates), dtype=np.float64)
        parsed = pd.to_datetime(dates)
        age_days = (parsed.max() - parsed).dt.days.to_numpy(dtype=np.float64)
        return np.asarray(0.5 ** (age_days / self.half_life_days), dtype=np.float64)

    def _global_fallback(self) -> _LeagueFit:
        """Average parameters across the fitted leagues — the fallback for an unseen league."""
        fits = list(self._leagues.values())
        if not fits:
            return self._global
        return _LeagueFit(
            attack={},
            defense={},
            home_adv=float(np.mean([fit.home_adv for fit in fits])),
            rho=float(np.mean([fit.rho for fit in fits])),
            avg_attack=float(np.mean([fit.avg_attack for fit in fits])),
            avg_defense=float(np.mean([fit.avg_defense for fit in fits])),
        )

    def predict_proba(self, table: pd.DataFrame, /) -> FloatArray:
        leagues = table["league"].astype(str).to_numpy()
        homes = table["home"].astype(str).to_numpy()
        aways = table["away"].astype(str).to_numpy()
        rows = [self._match(league, home, away) for league, home, away in zip(leagues, homes, aways)]
        return np.asarray(rows, dtype=np.float64).reshape(len(table), len(metrics.CLASSES))

    def _match(self, league: str, home: str, away: str) -> list[float]:
        fit = self._leagues.get(league, self._global)
        attack_home, defense_home = fit.strength(home)
        attack_away, defense_away = fit.strength(away)
        lam = float(np.exp(attack_home + defense_away + fit.home_adv))
        mu = float(np.exp(attack_away + defense_home))
        return _outcome_probabilities(lam, mu, fit.rho, self.max_goals)
