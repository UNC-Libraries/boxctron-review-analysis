"""Analyze human review decisions exported in a CSV file.

Usage:
    python analyze_review_responses.py path/to/reviews.csv
"""

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Sequence
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import beta as beta_distribution


REQUIRED_COLUMNS = {"id", "mgRiskScore", "mgContentTags"}
REVIEWER_COLUMN_PREFIX = "review"
TAG_PRIOR_STRENGTH_IMAGES = 5.0
_RATE_BOUND_EPSILON = 1e-12


def needs_review(value: str | None) -> bool:
    """Return whether a reviewer response is an explicit positive decision."""
    return value in {"y", "Y"}


def score_summary(scores: list[float]) -> dict[str, float | None]:
    """Return the requested descriptive statistics, or nulls when no scores exist."""
    if not scores:
        return {"average": None, "median": None, "min": None, "max": None}

    return {
        "average": statistics.mean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def percentage(numerator: int, denominator: int) -> float:
    """Calculate a percentage, returning zero when no responses are present."""
    return 100 * numerator / denominator if denominator else 0.0


def tag_bayesian_metrics(
    images_needing_review: int, image_count: int, global_image_review_rate: float
) -> dict[str, Any]:
    """Calculate prior-smoothed review metrics for one tag from image-level outcomes."""
    baseline = min(max(global_image_review_rate, _RATE_BOUND_EPSILON), 1.0 - _RATE_BOUND_EPSILON)
    prior_alpha = TAG_PRIOR_STRENGTH_IMAGES * baseline
    prior_beta = TAG_PRIOR_STRENGTH_IMAGES * (1.0 - baseline)
    posterior_alpha = prior_alpha + images_needing_review
    posterior_beta = prior_beta + image_count - images_needing_review

    credible_range_min, credible_range_max = beta_distribution.interval(
        0.95, posterior_alpha, posterior_beta
    )
    return {
        "images_majority_agree_needs_review_percentage": percentage(
            images_needing_review, image_count
        ),
        "baseline_smoothed_review_rate_percentage": round(
            100 * posterior_alpha / (posterior_alpha + posterior_beta), 2
        ),
        "estimated_review_rate_credible_range_min": round(100 * credible_range_min, 2),
        "estimated_review_rate_credible_range_max": round(100 * credible_range_max, 2),
        "probability_review_rate_exceeds_baseline_percentage": round(
            100 * beta_distribution.sf(global_image_review_rate, posterior_alpha, posterior_beta), 2
        ),
    }


def analyze_rows(rows: list[dict[str, str]], fieldnames: Sequence[str] | None) -> dict[str, Any]:
    """Calculate image, reviewer, and content-tag review rates from CSV rows."""
    if not fieldnames:
        raise ValueError("CSV file must include a header row.")

    missing_columns = REQUIRED_COLUMNS - set(fieldnames)
    if missing_columns:
        raise ValueError(f"CSV file is missing required columns: {', '.join(sorted(missing_columns))}")

    reviewer_columns = [
        fieldname for fieldname in fieldnames if fieldname.startswith(REVIEWER_COLUMN_PREFIX)
    ]
    if not reviewer_columns:
        raise ValueError("CSV file must include at least one reviewer column beginning with 'review'.")

    reviewer_scores: dict[str, dict[str, list[float]]] = {
        reviewer: {"needs_review": [], "does_not_need_review": []} for reviewer in reviewer_columns
    }
    all_reviewer_scores = {"needs_review": [], "does_not_need_review": []}
    tag_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"needs_review": 0, "total_responses": 0}
    )
    tag_image_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"images_needing_review": 0, "image_count": 0}
    )
    total_images_needing_review = 0
    images = []

    for row_number, row in enumerate(rows, start=2):
        try:
            risk_score = float(row["mgRiskScore"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Row {row_number} has an invalid mgRiskScore: {row['mgRiskScore']!r}") from error

        decisions = {reviewer: needs_review(row.get(reviewer)) for reviewer in reviewer_columns}
        positive_decisions = sum(decisions.values())
        image_needs_review = positive_decisions > len(reviewer_columns) / 2
        total_images_needing_review += image_needs_review
        images.append(
            {
                "id": row["id"],
                "title": row.get("title", ""),
                "risk_score": risk_score,
                "needs_review_count": positive_decisions,
                "reviewer_count": len(reviewer_columns),
                "needs_review_percentage": percentage(positive_decisions, len(reviewer_columns)),
            }
        )

        tags = {tag.strip() for tag in row["mgContentTags"].split(",") if tag.strip()}
        for tag in tags:
            tag_image_counts[tag]["image_count"] += 1
            tag_image_counts[tag]["images_needing_review"] += image_needs_review
        for reviewer, decision in decisions.items():
            category = "needs_review" if decision else "does_not_need_review"
            reviewer_scores[reviewer][category].append(risk_score)
            all_reviewer_scores[category].append(risk_score)
            for tag in tags:
                tag_counts[tag]["total_responses"] += 1
                tag_counts[tag]["needs_review"] += decision

    reviewers = {
        reviewer: {
            category: score_summary(scores)
            for category, scores in categories.items()
        }
        for reviewer, categories in reviewer_scores.items()
    }
    global_image_review_rate = total_images_needing_review / len(rows) if rows else 0.0
    content_tags = {}
    for tag, counts in sorted(tag_counts.items()):
        image_counts = tag_image_counts[tag]
        content_tags[tag] = {
            "total_times_marked_for_review": counts["needs_review"],
            "total_responses": counts["total_responses"],
            "reviewer_marked_for_review_percentage": percentage(
                counts["needs_review"], counts["total_responses"]
            ),
            "images_with_tag_needing_review_count": image_counts["images_needing_review"],
            "images_with_tag_total_count": image_counts["image_count"],
            **tag_bayesian_metrics(
                image_counts["images_needing_review"],
                image_counts["image_count"],
                global_image_review_rate,
            ),
        }

    return {
        "images": images,
        "reviewers": reviewers,
        "all_reviewers": {
            category: score_summary(scores) for category, scores in all_reviewer_scores.items()
        },
        "content_tag_analysis": {
            "global_image_review_rate_percentage": round(100 * global_image_review_rate, 2),
            "prior_strength_images": TAG_PRIOR_STRENGTH_IMAGES,
        },
        "content_tag_definitions": {
            "total_times_marked_for_review": "Number of reviewer decisions marking an image with the tag for review; it shows the raw positive evidence before accounting for sample size.",
            "total_responses": "Number of reviewer decisions on images with the tag; it is the denominator for the raw reviewer-level rate.",
            "reviewer_marked_for_review_percentage": "Percentage of reviewer decisions that marked images with the tag for review; it preserves the existing response-level result.",
            "images_with_tag_needing_review_count": "Number of tag-bearing images with a strict reviewer majority for review; it supplies independent image-level evidence for the Bayesian metrics.",
            "images_with_tag_total_count": "Number of distinct CSV rows containing the tag; it shows how much image-level evidence supports the tag's estimates.",
            "images_majority_agree_needs_review_percentage": "Percentage of tag-bearing images with a strict reviewer majority for review; it is the unsmoothed image-level rate.",
            "baseline_smoothed_review_rate_percentage": "Image-level review rate after combining the tag data with the global baseline and a five-image prior; it prevents sparse tags from appearing overly certain.",
            "estimated_review_rate_credible_range_min": "Lower end of the range that has a 95% probability of containing the tag's underlying image-level review rate, given the model, prior assumptions, and observed data.",
            "estimated_review_rate_credible_range_max": "Upper end of the range that has a 95% probability of containing the tag's underlying image-level review rate, given the model, prior assumptions, and observed data.",
            "probability_review_rate_exceeds_baseline_percentage": "Probability that the tag's underlying image-level review rate exceeds the global image-level review rate; higher values indicate stronger evidence that the tag is predictive.",
            "global_image_review_rate_percentage": "Strict-majority review rate across all images; it is the baseline used to evaluate whether a tag is elevated.",
            "prior_strength_images": "Number of image-equivalents assigned to the global baseline before tag-specific data is considered; increase it for more caution with sparse tags or decrease it to trust tag data sooner.",
        },
        "content_tags": content_tags,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze human review decisions from a CSV export.")
    parser.add_argument("csv_file", type=Path, help="CSV file containing image metadata and review columns.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.csv_file.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            report = analyze_rows(list(reader), reader.fieldnames)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
