"""Analyze human review decisions exported in a CSV file.

Usage:
    python analyze_review_responses.py path/to/reviews.csv [--format json|markdown|html]
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Sequence
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from scipy.stats import beta as beta_distribution


REQUIRED_COLUMNS = {"id", "mgRiskScore", "mgContentTags"}
REVIEWER_COLUMN_PREFIX = "review"
TAG_PRIOR_STRENGTH_IMAGES = 5.0
_RATE_BOUND_EPSILON = 1e-12
DCR_RECORD_URL_BASE = "https://dcr-test.lib.unc.edu/record/"
DCR_THUMBNAIL_URL_BASE = "https://dcr-test.lib.unc.edu/services/api/thumb/"


def needs_review(value: str | None) -> bool:
    """Return whether a reviewer response is an explicit positive decision."""
    return isinstance(value, str) and value.strip().lower() == "y"


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
    baseline = min(
        max(global_image_review_rate, _RATE_BOUND_EPSILON), 1.0 - _RATE_BOUND_EPSILON
    )
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
            100
            * beta_distribution.sf(
                global_image_review_rate, posterior_alpha, posterior_beta
            ),
            2,
        ),
    }


def markdown_cell(value: object) -> str:
    """Escape a value for use in a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def format_percentage(value: float) -> str:
    """Format a percentage for the Markdown report."""
    return f"{value:.1f}%"


def format_score(value: float | None) -> str:
    """Format a risk-score statistic for the Markdown report."""
    return "N/A" if value is None else f"{value:.2f}"


def dcr_image_urls(image_id: str | None) -> dict[str, str] | None:
    """Build DCR record and thumbnail URLs, or no URLs for a missing ID."""
    if not image_id or image_id in {".", ".."}:
        return None

    encoded_id = quote(image_id, safe="")
    return {
        "record": f"{DCR_RECORD_URL_BASE}{encoded_id}",
        "thumbnail": f"{DCR_THUMBNAIL_URL_BASE}{encoded_id}/small",
    }


def format_markdown_report(report: dict[str, Any]) -> str:
    """Render an analysis report as a shareable Markdown document."""
    analysis = report["content_tag_analysis"]
    lines = [
        "# Human Review Analysis",
        "",
        "## Summary",
        "",
        f"- Images analyzed: {len(report['images'])}",
        f"- Reviewers: {len(report['reviewers'])}",
        f"- Content tags: {len(report['content_tags'])}",
        "- Global image-level review rate: "
        f"{format_percentage(analysis['global_image_review_rate_percentage'])}",
        f"- Tag prior strength: {analysis['prior_strength_images']} image-equivalents",
        "",
        "## Reviewer Risk Scores",
        "",
        "| Reviewer | Decision | Average | Median | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for reviewer, categories in report["reviewers"].items():
        for decision, scores in categories.items():
            lines.append(
                "| "
                f"{markdown_cell(reviewer)} | {markdown_cell(decision)} | "
                f"{format_score(scores['average'])} | {format_score(scores['median'])} | "
                f"{format_score(scores['min'])} | {format_score(scores['max'])} |"
            )
    for decision, scores in report["all_reviewers"].items():
        lines.append(
            "| All reviewers | "
            f"{markdown_cell(decision)} | {format_score(scores['average'])} | "
            f"{format_score(scores['median'])} | {format_score(scores['min'])} | "
            f"{format_score(scores['max'])} |"
        )

    lines.extend(
        [
            "",
            "## Content Tag Review Signals",
            "",
            "| Tag | Images | Majority needs review | Reviewer marked for review | Smoothed review rate | 95% credible range | Probability above baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for tag, metrics in report["content_tags"].items():
        lines.append(
            "| "
            f"{markdown_cell(tag)} | {metrics['images_with_tag_total_count']} | "
            f"{metrics['images_with_tag_needing_review_count']} / "
            f"{metrics['images_with_tag_total_count']} "
            f"({format_percentage(metrics['images_majority_agree_needs_review_percentage'])}) | "
            f"{metrics['total_times_marked_for_review']} / {metrics['total_responses']} "
            f"({format_percentage(metrics['reviewer_marked_for_review_percentage'])}) | "
            f"{format_percentage(metrics['baseline_smoothed_review_rate_percentage'])} | "
            f"{format_percentage(metrics['estimated_review_rate_credible_range_min'])} to "
            f"{format_percentage(metrics['estimated_review_rate_credible_range_max'])} | "
            f"{format_percentage(metrics['probability_review_rate_exceeds_baseline_percentage'])} |"
        )

    lines.extend(
        [
            "",
            "### Interpreting Tag Signals",
            "",
            "- **Majority needs review** counts images where more than half of reviewers marked the image for review.",
            "- **Smoothed review rate** combines the tag's image data with the global review rate, weighted as five image-equivalents, so rare tags are not treated as certain.",
            "- **95% credible range** is the range that has a 95% probability of containing the tag's underlying long-term review rate, given the model and available data.",
            "- **Probability above baseline** estimates how likely the tag's underlying review rate is higher than the global image-level rate.",
            "",
            "## Per-Image Review Decisions",
            "",
            "| Image ID | Title | Risk score | Positive decisions | Reviewers | Marked for review |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for image in report["images"]:
        lines.append(
            "| "
            f"{markdown_cell(image['id'])} | {markdown_cell(image['title'])} | "
            f"{image['risk_score']:.2f} | {image['needs_review_count']} | "
            f"{image['reviewer_count']} | {format_percentage(image['needs_review_percentage'])} |"
        )

    return "\n".join(lines) + "\n"


HUMAN_TRUTH_CRITERIA = {"any", "half", "majority", "all"}


def human_truth(image: dict[str, Any], criterion: str = "majority") -> bool:
    """Apply a human-review truth criterion to normalized positive vote counts."""
    if criterion not in HUMAN_TRUTH_CRITERIA:
        raise ValueError("criterion must be 'any', 'half', 'majority', or 'all'.")

    if "needs_review_count" not in image or "reviewer_count" not in image:
        if criterion == "majority" and "human_needs_review" in image:
            return bool(image["human_needs_review"])
        raise ValueError("image needs normalized vote counts for this truth criterion.")

    positive_votes = image["needs_review_count"]
    reviewer_count = image["reviewer_count"]
    if criterion == "any":
        return positive_votes > 0
    if criterion == "all":
        return positive_votes == reviewer_count
    if criterion == "half":
        return positive_votes >= reviewer_count / 2
    return positive_votes > reviewer_count / 2


def predicts_review(
    image: dict[str, Any],
    selected_tags: set[str],
    match_mode: str,
    risk_threshold: float | None = None,
) -> bool:
    """Return whether the category rule or optional score threshold predicts review."""
    if match_mode not in {"any", "all"}:
        raise ValueError("match_mode must be 'any' or 'all'.")
    risk_threshold = validated_risk_threshold(risk_threshold)
    image_tags = set(image["content_tags"])
    category_matches = bool(selected_tags) and (
        bool(image_tags & selected_tags)
        if match_mode == "any"
        else selected_tags <= image_tags
    )
    return category_matches or (
        risk_threshold is not None and image["risk_score"] >= risk_threshold
    )


def validated_risk_threshold(risk_threshold: float | None) -> float | None:
    """Return an enabled finite threshold, rejecting non-finite API inputs."""
    if risk_threshold is None:
        return None
    if isinstance(risk_threshold, bool):
        raise ValueError("risk_threshold must be a finite number or None.")
    try:
        threshold = float(risk_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("risk_threshold must be a finite number or None.") from error
    if not math.isfinite(threshold):
        raise ValueError("risk_threshold must be a finite number or None.")
    return threshold


def explorer_metrics(
    images: Sequence[dict[str, Any]],
    selected_tags: set[str],
    match_mode: str,
    truth_criterion: str = "majority",
    risk_threshold: float | None = None,
) -> dict[str, int | float | None | list[dict[str, Any]]]:
    """Classify images using category matching OR an optional score threshold."""
    risk_threshold = validated_risk_threshold(risk_threshold)
    classified_images = []
    counts = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }
    for image in images:
        predicted_needs_review = predicts_review(
            image, selected_tags, match_mode, risk_threshold
        )
        human_needs_review = human_truth(image, truth_criterion)
        classification = (
            "true_positive"
            if predicted_needs_review and human_needs_review
            else "false_positive"
            if predicted_needs_review
            else "false_negative"
            if human_needs_review
            else "true_negative"
        )
        counts[classification] += 1
        classified_images.append(
            {
                **image,
                "predicted_needs_review": predicted_needs_review,
                "human_needs_review": human_needs_review,
                "classification": classification,
            }
        )

    precision_denominator = counts["true_positive"] + counts["false_positive"]
    recall_denominator = counts["true_positive"] + counts["false_negative"]
    return {
        **counts,
        "precision": counts["true_positive"] / precision_denominator
        if precision_denominator
        else None,
        "recall": counts["true_positive"] / recall_denominator
        if recall_denominator
        else None,
        "images": classified_images,
    }


def explorer_suggestions(
    images: Sequence[dict[str, Any]],
    selected_tags: set[str],
    match_mode: str,
    truth_criterion: str = "majority",
    risk_threshold: float | None = None,
) -> list[dict[str, int | str]]:
    """Return category-only marginal changes while holding the threshold fixed."""
    risk_threshold = validated_risk_threshold(risk_threshold)
    baseline = explorer_metrics(
        images, selected_tags, match_mode, truth_criterion, risk_threshold
    )
    all_tags = sorted({tag for image in images for tag in image["content_tags"]})
    candidates = (
        ((tag, selected_tags | {tag}) for tag in all_tags if tag not in selected_tags)
        if match_mode == "any"
        else ((tag, selected_tags - {tag}) for tag in sorted(selected_tags))
    )
    suggestions = []
    for tag, candidate_tags in candidates:
        candidate = explorer_metrics(
            images, candidate_tags, match_mode, truth_criterion, risk_threshold
        )
        caught_change = candidate["true_positive"] - baseline["true_positive"]
        excess_change = candidate["false_positive"] - baseline["false_positive"]
        if caught_change > 0:
            suggestions.append(
                {
                    "tag": tag,
                    "additional_caught": caught_change,
                    "additional_excess": excess_change,
                }
            )
    return sorted(
        suggestions,
        key=lambda suggestion: (
            -suggestion["additional_caught"],
            suggestion["additional_excess"],
            suggestion["tag"],
        ),
    )


def explorer_zero_result_explanation(
    images: Sequence[dict[str, Any]],
    selected_tags: set[str],
    match_mode: str,
    truth_criterion: str = "majority",
    risk_threshold: float | None = None,
) -> str | None:
    """Explain why the active explorer selection has no caught images."""
    risk_threshold = validated_risk_threshold(risk_threshold)
    matching_images = [
        image
        for image in images
        if predicts_review(image, selected_tags, match_mode, risk_threshold)
    ]
    if not matching_images and risk_threshold is not None:
        if not selected_tags:
            return (
                "No images meet the active risk threshold, and no categories are "
                "selected, so no images are predicted for review."
            )
        return (
            "No image matches the selected categories or meets the active risk "
            "threshold, so no images are predicted for review."
        )
    if not selected_tags and risk_threshold is None:
        return "No categories selected, so no images are predicted for review."
    if not matching_images and match_mode == "all":
        return "No image has every selected category, so no images are predicted for review."
    if matching_images and not any(
        human_truth(image, truth_criterion) for image in matching_images
    ):
        return (
            "Matching images exist, but none meet the selected human truth criterion."
        )
    return None


from html_explorer import format_html_explorer


def analyze_rows(
    rows: list[dict[str, str]], fieldnames: Sequence[str] | None
) -> dict[str, Any]:
    """Calculate image, reviewer, and content-tag review rates from CSV rows."""
    if not fieldnames:
        raise ValueError("CSV file must include a header row.")

    missing_columns = REQUIRED_COLUMNS - set(fieldnames)
    if missing_columns:
        raise ValueError(
            f"CSV file is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    reviewer_columns = [
        fieldname
        for fieldname in fieldnames
        if fieldname.startswith(REVIEWER_COLUMN_PREFIX)
    ]
    if not reviewer_columns:
        raise ValueError(
            "CSV file must include at least one reviewer column beginning with 'review'."
        )

    reviewer_scores: dict[str, dict[str, list[float]]] = {
        reviewer: {"needs_review": [], "does_not_need_review": []}
        for reviewer in reviewer_columns
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
            raise ValueError(
                f"Row {row_number} has an invalid mgRiskScore: {row['mgRiskScore']!r}"
            ) from error
        if not math.isfinite(risk_score):
            raise ValueError(
                f"Row {row_number} has an invalid mgRiskScore: {row['mgRiskScore']!r}"
            )

        decisions = {
            reviewer: needs_review(row.get(reviewer)) for reviewer in reviewer_columns
        }
        positive_decisions = sum(decisions.values())
        image_needs_review = positive_decisions > len(reviewer_columns) / 2
        total_images_needing_review += image_needs_review
        tags = sorted(
            {tag.strip() for tag in row["mgContentTags"].split(",") if tag.strip()}
        )
        images.append(
            {
                "id": row["id"],
                "title": row.get("title", ""),
                "risk_score": risk_score,
                "needs_review_count": positive_decisions,
                "reviewer_count": len(reviewer_columns),
                "needs_review_percentage": percentage(
                    positive_decisions, len(reviewer_columns)
                ),
                "content_tags": tags,
                "human_needs_review": image_needs_review,
            }
        )

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
            category: score_summary(scores) for category, scores in categories.items()
        }
        for reviewer, categories in reviewer_scores.items()
    }
    global_image_review_rate = total_images_needing_review / len(rows) if rows else 0.0
    tag_metrics = {}
    for tag, counts in sorted(tag_counts.items()):
        image_counts = tag_image_counts[tag]
        tag_metrics[tag] = {
            "total_times_marked_for_review": counts["needs_review"],
            "total_responses": counts["total_responses"],
            "reviewer_marked_for_review_percentage": percentage(
                counts["needs_review"], counts["total_responses"]
            ),
            "images_with_tag_needing_review_count": image_counts[
                "images_needing_review"
            ],
            "images_with_tag_total_count": image_counts["image_count"],
            **tag_bayesian_metrics(
                image_counts["images_needing_review"],
                image_counts["image_count"],
                global_image_review_rate,
            ),
        }
    content_tags = dict(
        sorted(
            tag_metrics.items(),
            key=lambda item: (
                -item[1]["baseline_smoothed_review_rate_percentage"],
                -item[1]["estimated_review_rate_credible_range_min"],
                -item[1]["probability_review_rate_exceeds_baseline_percentage"],
                item[0],
            ),
        )
    )

    return {
        "images": images,
        "reviewers": reviewers,
        "all_reviewers": {
            category: score_summary(scores)
            for category, scores in all_reviewer_scores.items()
        },
        "content_tag_analysis": {
            "global_image_review_rate_percentage": round(
                100 * global_image_review_rate, 2
            ),
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
    parser = argparse.ArgumentParser(
        description="Analyze human review decisions from a CSV export."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="CSV file containing image metadata and review columns.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "markdown", "html"),
        default="json",
        help="Report format (default: json).",
    )
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

    if args.output_format == "markdown":
        print(format_markdown_report(report), end="")
    elif args.output_format == "html":
        print(format_html_explorer(report), end="")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
