import pytest

from analyze_review_responses import analyze_rows, needs_review, tag_bayesian_metrics


def test_needs_review_accepts_only_literal_y_values():
    assert needs_review("y")
    assert needs_review("Y")
    assert not needs_review("Y ")
    assert not needs_review("maybe")
    assert not needs_review("")
    assert not needs_review(None)


def test_analyze_rows_calculates_image_reviewer_and_tag_metrics():
    report = analyze_rows(
        [
            {
                "id": "first",
                "title": "First image",
                "mgRiskScore": "10",
                "mgContentTags": "people_visible, text_language_english, people_visible",
                "review_alex": "Y",
                "review_blair": "N",
            },
            {
                "id": "second",
                "title": "Second image",
                "mgRiskScore": "30",
                "mgContentTags": "text_language_english",
                "review_alex": "N",
                "review_blair": "y",
            },
        ],
        ["id", "title", "mgRiskScore", "mgContentTags", "review_alex", "review_blair"],
    )

    assert report["images"] == [
        {
            "id": "first",
            "title": "First image",
            "risk_score": 10.0,
            "needs_review_count": 1,
            "reviewer_count": 2,
            "needs_review_percentage": 50.0,
        },
        {
            "id": "second",
            "title": "Second image",
            "risk_score": 30.0,
            "needs_review_count": 1,
            "reviewer_count": 2,
            "needs_review_percentage": 50.0,
        },
    ]
    assert report["reviewers"] == {
        "review_alex": {
            "needs_review": {"average": 10.0, "median": 10.0, "min": 10.0, "max": 10.0},
            "does_not_need_review": {"average": 30.0, "median": 30.0, "min": 30.0, "max": 30.0},
        },
        "review_blair": {
            "needs_review": {"average": 30.0, "median": 30.0, "min": 30.0, "max": 30.0},
            "does_not_need_review": {"average": 10.0, "median": 10.0, "min": 10.0, "max": 10.0},
        },
    }
    assert report["all_reviewers"] == {
        "needs_review": {"average": 20.0, "median": 20.0, "min": 10.0, "max": 30.0},
        "does_not_need_review": {"average": 20.0, "median": 20.0, "min": 10.0, "max": 30.0},
    }
    assert report["content_tag_analysis"] == {
        "global_image_review_rate_percentage": 0.0,
        "prior_strength_images": 5.0,
    }
    assert report["content_tag_definitions"]["prior_strength_images"].startswith(
        "Number of image-equivalents"
    )
    assert report["content_tags"]["people_visible"] == {
        "total_times_marked_for_review": 1,
        "total_responses": 2,
        "reviewer_marked_for_review_percentage": 50.0,
        "images_with_tag_needing_review_count": 0,
        "images_with_tag_total_count": 1,
        "images_majority_agree_needs_review_percentage": 0.0,
        "baseline_smoothed_review_rate_percentage": 0.0,
        "estimated_review_rate_credible_range_min": 0.0,
        "estimated_review_rate_credible_range_max": 0.0,
        "probability_review_rate_exceeds_baseline_percentage": 100.0,
    }
    assert report["content_tags"]["text_language_english"]["images_with_tag_total_count"] == 2
    assert report["content_tags"]["text_language_english"]["total_responses"] == 4


def test_tag_bayesian_metrics_shrinks_sparse_tags_toward_global_rate():
    metrics = tag_bayesian_metrics(2, 2, 0.3)

    assert metrics["images_majority_agree_needs_review_percentage"] == 100.0
    assert metrics["baseline_smoothed_review_rate_percentage"] == 50.0
    assert metrics["estimated_review_rate_credible_range_min"] < 50.0
    assert metrics["estimated_review_rate_credible_range_max"] > 50.0
    assert metrics["probability_review_rate_exceeds_baseline_percentage"] > 80.0


def test_analyze_rows_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns: mgContentTags"):
        analyze_rows([], ["id", "mgRiskScore", "review_alex"])
