"""Confidence intervals for eTPR, eFPR, and d-prime.

The input is the ``binary_judgements_df`` dataframe created in
``investigate_llm_judgements.ipynb``. Macro estimates average topic-level
metrics and uses the topics directly as the confidence-interval samples, while
micro estimates pool all rows before calculating each metric and uses a
bootstrap confidence interval.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


DEFAULT_JUDGES = (
    "alt1_binary",
    "alt2_binary",
    "alt3_binary",
    "sota_llm_binary",
    "local_new_graded_binary",
    "local_old",
    "local_new_binary",
)
METRIC_NAMES = ("etpr", "efpr", "d_prime")


def _metrics_from_counts(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    positive_count: np.ndarray,
    negative_count: np.ndarray,
) -> np.ndarray:
    """Return eTPR, eFPR, and d-prime for one or more count vectors."""
    if positive_count.ndim < true_positive.ndim:
        positive_count = positive_count[..., None]
    if negative_count.ndim < false_positive.ndim:
        negative_count = negative_count[..., None]
    etpr = (true_positive + 0.5) / (positive_count + 1)
    efpr = (false_positive + 0.5) / (negative_count + 1)
    d_prime = norm.ppf(etpr) - norm.ppf(efpr)
    return np.stack((etpr, efpr, d_prime), axis=-1)


def _validate_input(
    binary_judgements_df: pd.DataFrame,
    truth_column: str,
    judges: Sequence[str],
    expected_topic_count: int,
) -> None:
    required_columns = {truth_column, "topic", *judges}
    missing_columns = required_columns.difference(binary_judgements_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if binary_judgements_df.empty:
        raise ValueError("binary_judgements_df must contain at least one row")

    topic_count = binary_judgements_df["topic"].nunique()
    if topic_count != expected_topic_count:
        raise ValueError(
            f"Expected {expected_topic_count} topics, found {topic_count}"
        )

    binary_columns = [truth_column, *judges]
    for column in binary_columns:
        values = binary_judgements_df[column].dropna().astype(int)
        if len(values) != len(binary_judgements_df) or not values.isin([0, 1]).all():
            raise ValueError(f"{column!r} must contain only non-missing 0/1 values")


def _bootstrap_ci(
    samples: np.ndarray,
    point_estimate: np.ndarray,
    confidence_level: float,
) -> np.ndarray:
    alpha = 1 - confidence_level
    lower, upper = np.quantile(
        samples,
        (alpha / 2, 1 - alpha / 2),
        axis=0,
    )
    return np.stack((point_estimate, lower, upper), axis=-1)


def _topic_sample_ci(
    topic_metrics: np.ndarray,
    confidence_level: float,
) -> np.ndarray:
    """Return a Student-t CI using the topic-level metrics as samples."""
    sample_count = topic_metrics.shape[0]
    point_estimate = topic_metrics.mean(axis=0)
    standard_error = topic_metrics.std(axis=0, ddof=1) / np.sqrt(sample_count)
    alpha = 1 - confidence_level
    critical_value = student_t.ppf(1 - alpha / 2, df=sample_count - 1)
    margin = critical_value * standard_error
    return np.stack(
        (point_estimate, point_estimate - margin, point_estimate + margin),
        axis=-1,
    )


def calculate_metric_confidence_intervals(
    binary_judgements_df: pd.DataFrame,
    *,
    truth_column: str = "athome4_truth_binary",
    judges: Sequence[str] = DEFAULT_JUDGES,
    expected_topic_count: int = 34,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 20261002,
) -> pd.DataFrame:
    """Calculate point estimates and 95% confidence intervals.

    Macro calculates metrics for each topic and uses the 34 topic-level
    metrics directly as samples for a Student-t confidence interval. Micro
    samples individual rows with replacement for a percentile bootstrap
    confidence interval. The returned dataframe has one row per aggregation
    and judge, with ``estimate``, ``ci_lower``, and ``ci_upper`` for each
    metric.
    """
    judges = tuple(judges)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    _validate_input(
        binary_judgements_df,
        truth_column,
        judges,
        expected_topic_count,
    )

    rng = np.random.default_rng(random_seed)
    truth = binary_judgements_df[truth_column].to_numpy(dtype=bool)
    predictions = binary_judgements_df[list(judges)].to_numpy(dtype=bool)

    # Micro: resample rows, then pool the sampled rows before calculating rates.
    row_samples = rng.integers(0, len(truth), size=(n_bootstrap, len(truth)))
    sampled_truth = truth[row_samples]
    sampled_predictions = predictions[row_samples]
    micro_tp = np.sum(sampled_predictions & sampled_truth[:, :, None], axis=1)
    micro_fp = np.sum(sampled_predictions & ~sampled_truth[:, :, None], axis=1)
    positive_count = sampled_truth.sum(axis=1)
    negative_count = (~sampled_truth).sum(axis=1)
    micro_bootstrap = _metrics_from_counts(
        micro_tp,
        micro_fp,
        positive_count,
        negative_count,
    )
    micro_point = _metrics_from_counts(
        np.sum(predictions & truth[:, None], axis=0),
        np.sum(predictions & ~truth[:, None], axis=0),
        np.array(truth.sum()),
        np.array((~truth).sum()),
    )

    # Macro: calculate metrics within each topic and use the topics directly
    # as the confidence-interval samples.
    topic_metrics = []
    for _, topic_df in binary_judgements_df.groupby("topic", sort=True):
        topic_truth = topic_df[truth_column].to_numpy(dtype=bool)
        topic_predictions = topic_df[list(judges)].to_numpy(dtype=bool)
        topic_metrics.append(
            _metrics_from_counts(
                np.sum(topic_predictions & topic_truth[:, None], axis=0),
                np.sum(topic_predictions & ~topic_truth[:, None], axis=0),
                np.array(topic_truth.sum()),
                np.array((~topic_truth).sum()),
            )
        )
    topic_metrics_array = np.stack(topic_metrics)
    macro_interval_values = _topic_sample_ci(
        topic_metrics_array,
        confidence_level,
    )

    results = []
    for aggregation, interval_values in (
        ("macro", macro_interval_values),
        (
            "micro",
            _bootstrap_ci(
                micro_bootstrap,
                micro_point,
                confidence_level,
            ),
        )
    ):
        for judge_index, judge in enumerate(judges):
            for metric_index, metric in enumerate(METRIC_NAMES):
                estimate, ci_lower, ci_upper = interval_values[
                    judge_index, metric_index
                ]
                results.append(
                    {
                        "aggregation": aggregation,
                        "judge": judge,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )
    return pd.DataFrame(results)


if __name__ == "__main__":
    raise SystemExit(
        "Import calculate_metric_confidence_intervals and pass "
        "binary_judgements_df from the notebook."
    )
