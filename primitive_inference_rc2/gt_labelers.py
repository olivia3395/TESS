"""Ground-truth primitive labelers derived from historical_data + ground_truth.

Thresholds are always fitted on the train split and then reused for train,
vali, and test. This mirrors the legacy extractor and prevents validation/test
distribution information from leaking into GT primitive definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


EPS = 1e-8
MEAN_ABS_QUANTILES = (0.25, 0.75)
VOLATILITY_QUANTILES = (0.33, 0.66)
DEFAULT_SHAPE_QUANTILE = 0.33
ALPHA = 1.0
TEMPORAL_ETA = 0.15
TEMPORAL_KAPPA_1 = 0.50
TEMPORAL_KAPPA_2 = 0.55
TEMPORAL_RHO = 0.25
VALID_DISTRIBUTION_SHIFT_LABELS = (
    "STRONG-RISE",
    "MILD-RISE",
    "STABLE",
    "MILD-DROP",
    "STRONG-DROP",
)
VALID_VOLATILITY_LABELS = ("High", "Medium", "Low")
VALID_SHAPE_LABELS = ("Rise", "Fall", "Peak", "Recover", "Oscillate")
VALID_TEMPORAL_INFLUENCE_LABELS = ("Immediate", "Delayed", "Sustained")


@dataclass(frozen=True)
class DistributionShiftThresholds:
    method: str
    values: dict[str, float]
    fit_split: str
    score_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "values": self.values,
            "fit_split": self.fit_split,
            "score_name": self.score_name,
        }


class DistributionShiftLabeler:
    def fit(self, train_records: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [self._shift_score(record) for record in train_records]
        abs_scores = sorted(abs(score) for score in scores)
        if not abs_scores:
            raise ValueError("No valid train records available for threshold fitting.")
        thresholds = DistributionShiftThresholds(
            method="legacy_train_only_abs_delta_mu_quantiles",
            values={
                "tau_mu_1": _quantile(abs_scores, MEAN_ABS_QUANTILES[0]),
                "tau_mu_2": _quantile(abs_scores, MEAN_ABS_QUANTILES[1]),
                "quantile_low": MEAN_ABS_QUANTILES[0],
                "quantile_high": MEAN_ABS_QUANTILES[1],
            },
            fit_split="train",
            score_name="delta_mu",
        )
        return thresholds.to_dict()

    def compute(
        self,
        record: dict[str, Any],
        thresholds: dict[str, Any],
    ) -> tuple[str, dict[str, float]]:
        shift_score = self._shift_score(record)
        values = thresholds["values"]
        tau_mu_1 = float(values["tau_mu_1"])
        tau_mu_2 = float(values["tau_mu_2"])

        if shift_score > tau_mu_2:
            label = "STRONG-RISE"
        elif tau_mu_1 < shift_score <= tau_mu_2:
            label = "MILD-RISE"
        elif -tau_mu_1 <= shift_score <= tau_mu_1:
            label = "STABLE"
        elif -tau_mu_2 <= shift_score < -tau_mu_1:
            label = "MILD-DROP"
        else:
            label = "STRONG-DROP"

        return label, {"shift_score": shift_score}

    def _shift_score(self, record: dict[str, Any]) -> float:
        x = parse_numeric_sequence(record.get("historical_data"))
        y = parse_numeric_sequence(record.get("ground_truth"))
        if not x or not y:
            raise ValueError("Record must contain non-empty historical_data and ground_truth.")
        if len(x) != len(y):
            raise ValueError(
                "historical_data and ground_truth must have equal sequence lengths "
                f"(got {len(x)} and {len(y)})."
            )
        return (sum(y) / len(y) - sum(x) / len(x)) / (_population_std(x) + EPS)


@dataclass(frozen=True)
class VolatilityThresholds:
    method: str
    values: dict[str, float]
    fit_split: str
    score_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "values": self.values,
            "fit_split": self.fit_split,
            "score_name": self.score_name,
        }


class VolatilityLabeler:
    def fit(self, train_records: list[dict[str, Any]]) -> dict[str, Any]:
        scores = sorted(self._volatility_score(record) for record in train_records)
        if not scores:
            raise ValueError("No valid train records available for threshold fitting.")
        thresholds = VolatilityThresholds(
            method="legacy_train_only_r_sigma_quantiles",
            values={
                "vol_q_low": _quantile(scores, VOLATILITY_QUANTILES[0]),
                "vol_q_high": _quantile(scores, VOLATILITY_QUANTILES[1]),
                "quantile_low": VOLATILITY_QUANTILES[0],
                "quantile_high": VOLATILITY_QUANTILES[1],
            },
            fit_split="train",
            score_name="r_sigma",
        )
        return thresholds.to_dict()

    def compute(
        self,
        record: dict[str, Any],
        thresholds: dict[str, Any],
    ) -> tuple[str, dict[str, float]]:
        volatility_score = self._volatility_score(record)
        values = thresholds["values"]
        vol_q_low = float(values["vol_q_low"])
        vol_q_high = float(values["vol_q_high"])

        if volatility_score <= vol_q_low:
            label = "Low"
        elif vol_q_low < volatility_score <= vol_q_high:
            label = "Medium"
        else:
            label = "High"
        return label, {"volatility_score": volatility_score, "r_sigma": volatility_score}

    def _volatility_score(self, record: dict[str, Any]) -> float:
        x = parse_numeric_sequence(record.get("historical_data"))
        y = parse_numeric_sequence(record.get("ground_truth"))
        if not x or not y:
            raise ValueError("Record must contain non-empty historical_data and ground_truth.")
        if len(x) != len(y):
            raise ValueError(
                "historical_data and ground_truth must have equal sequence lengths "
                f"(got {len(x)} and {len(y)})."
            )
        sigma_x = _population_std(_diff(x))
        sigma_y = _population_std(_diff(y))
        return math.log((sigma_y + EPS) / (sigma_x + EPS))


@dataclass(frozen=True)
class ShapeThresholds:
    method: str
    values: dict[str, float]
    fit_split: str
    score_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "values": self.values,
            "fit_split": self.fit_split,
            "score_name": self.score_name,
        }


class ShapeLabeler:
    def fit(self, train_records: list[dict[str, Any]]) -> dict[str, Any]:
        patch_diffs: list[float] = []
        for record in train_records:
            y = parse_numeric_sequence(record.get("ground_truth"))
            if not y:
                continue
            patches = _split_patches(y, _n_fcst_for_sequence(y))
            if len(patches) >= 2:
                patch_means = [sum(patch) / len(patch) for patch in patches]
                patch_diffs.extend(abs(value) for value in _diff(patch_means))
        if not patch_diffs:
            patch_diffs = [0.0]
        sorted_diffs = sorted(patch_diffs)
        thresholds = ShapeThresholds(
            method="legacy_train_only_patch_diff_quantile",
            values={
                "tau_shape": _quantile(sorted_diffs, DEFAULT_SHAPE_QUANTILE),
                "shape_quantile": DEFAULT_SHAPE_QUANTILE,
            },
            fit_split="train",
            score_name="shape_score",
        )
        return thresholds.to_dict()

    def compute(
        self,
        record: dict[str, Any],
        thresholds: dict[str, Any],
    ) -> tuple[str, dict[str, float]]:
        y = parse_numeric_sequence(record.get("ground_truth"))
        if not y:
            raise ValueError("Record must contain non-empty ground_truth.")
        label, shape_score = _shape_profile(
            y,
            n_fcst=_n_fcst_for_sequence(y),
            tau_shape=float(thresholds["values"]["tau_shape"]),
        )
        return label, {"shape_score": shape_score}


@dataclass(frozen=True)
class TemporalInfluenceThresholds:
    method: str
    values: dict[str, float]
    fit_split: str
    score_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "values": self.values,
            "fit_split": self.fit_split,
            "score_name": self.score_name,
        }


class TemporalInfluenceLabeler:
    def fit(self, train_records: list[dict[str, Any]]) -> dict[str, Any]:
        thresholds = TemporalInfluenceThresholds(
            method="legacy_fixed_temporal_parameters",
            values={
                "eta": TEMPORAL_ETA,
                "kappa_1": TEMPORAL_KAPPA_1,
                "kappa_2": TEMPORAL_KAPPA_2,
                "rho": TEMPORAL_RHO,
                "alpha": ALPHA,
            },
            fit_split="train",
            score_name="centroid_tail_peak",
        )
        return thresholds.to_dict()

    def compute(
        self,
        record: dict[str, Any],
        thresholds: dict[str, Any],
    ) -> tuple[str, dict[str, float]]:
        x = parse_numeric_sequence(record.get("historical_data"))
        y = parse_numeric_sequence(record.get("ground_truth"))
        if not x or not y:
            raise ValueError("Record must contain non-empty historical_data and ground_truth.")
        if len(x) != len(y):
            raise ValueError(
                "historical_data and ground_truth must have equal sequence lengths "
                f"(got {len(x)} and {len(y)})."
            )
        c, d, q = _temporal_influence_features(x, y, _n_fcst_for_sequence(y))
        values = thresholds["values"]
        eta = float(values["eta"])
        kappa_1 = float(values["kappa_1"])
        kappa_2 = float(values["kappa_2"])
        rho = float(values["rho"])

        if q <= eta:
            label = "Sustained"
        elif c <= kappa_1:
            label = "Immediate"
        elif c <= kappa_2:
            label = "Sustained" if d > rho else "Delayed"
        else:
            label = "Delayed"
        return label, {
            "centroid_c": c,
            "tail_mass_d": d,
            "peak_prominence_q": q,
        }


def parse_numeric_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.strip().split(",")]
        values = [float(part) for part in parts if part]
    elif isinstance(value, (list, tuple)):
        values = [float(part) for part in value]
    else:
        return []
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Numeric sequence contains non-finite values.")
    return values


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _diff(values: list[float]) -> list[float]:
    return [right - left for left, right in zip(values, values[1:])]


def _n_fcst_for_sequence(values: list[float]) -> int:
    return 4 if len(values) == 5 else 6


def _split_patches(values: list[float], n_fcst: int) -> list[list[float]]:
    if not values:
        return []
    m = min(n_fcst, len(values))
    base_size, remainder = divmod(len(values), m)
    patches = []
    start = 0
    for idx in range(m):
        size = base_size + (1 if idx < remainder else 0)
        patch = values[start : start + size]
        if patch:
            patches.append(patch)
        start += size
    return patches


def _shape_profile(
    y: list[float],
    n_fcst: int,
    tau_shape: float,
) -> tuple[str, float]:
    patches = _split_patches(y, n_fcst)
    if len(patches) < 2:
        return "Oscillate", 0.0

    patch_means = [sum(patch) / len(patch) for patch in patches]
    diffs = _diff(patch_means)
    signs = []
    for value in diffs:
        if value > tau_shape:
            signs.append(1)
        elif value < -tau_shape:
            signs.append(-1)
        else:
            signs.append(0)

    shape_score = sum(abs(value) for value in diffs) / len(diffs)
    if all(sign >= 0 for sign in signs) and any(sign == 1 for sign in signs):
        return "Rise", shape_score
    if all(sign <= 0 for sign in signs) and any(sign == -1 for sign in signs):
        return "Fall", shape_score

    non_zero = [sign for sign in signs if sign != 0]
    if not non_zero:
        return "Oscillate", shape_score

    flips = sum(
        right != left for left, right in zip(non_zero, non_zero[1:])
    )
    if flips == 1 and non_zero[0] == 1 and non_zero[-1] == -1:
        return "Peak", shape_score
    if flips == 1 and non_zero[0] == -1 and non_zero[-1] == 1:
        return "Recover", shape_score
    return "Oscillate", shape_score


def _temporal_influence_features(
    x: list[float],
    y: list[float],
    n_fcst: int,
) -> tuple[float, float, float]:
    patches = _split_patches(y, n_fcst)
    if not patches:
        return 0.0, 0.0, 0.0

    mean_x = sum(x) / len(x)
    std_x = _population_std(x)
    diff_std_x = _population_std(_diff(x))
    a_values = []
    for patch in patches:
        patch_mean = sum(patch) / len(patch)
        mean_part = abs((patch_mean - mean_x) / (std_x + EPS))
        diff_std_patch = _population_std(_diff(patch))
        vol_part = ALPHA * abs(math.log((diff_std_patch + EPS) / (diff_std_x + EPS)))
        a_values.append(float(mean_part + vol_part))

    total = sum(a_values)
    if total <= EPS:
        pi = [1.0 / len(a_values) for _ in a_values]
    else:
        pi = [value / total for value in a_values]

    if len(pi) == 1:
        c = 0.0
    else:
        c = sum(value * (idx / (len(pi) - 1)) for idx, value in enumerate(pi))

    i_star = max(range(len(pi)), key=lambda idx: pi[idx])
    d = sum(pi[i_star + 1 :]) if i_star + 1 < len(pi) else 0.0
    q = max(pi)
    return float(c), float(d), float(q)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of an empty sequence.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)
