import subprocess
import sys

import pytest

from analyze_review_responses import (
    analyze_rows,
    dcr_image_urls,
    explorer_metrics,
    explorer_suggestions,
    explorer_zero_result_explanation,
    format_html_explorer,
    format_markdown_report,
    human_truth,
    needs_review,
    predicts_review,
    tag_bayesian_metrics,
)


@pytest.mark.parametrize("value", ["y", "Y", " Y ", "\ty\n"])
def test_needs_review_accepts_whitespace_padded_y_values(value):
    assert needs_review(value)


@pytest.mark.parametrize("value", ["maybe", "?", "", " ", "n", "N", "yes", None])
def test_needs_review_rejects_non_y_values(value):
    assert not needs_review(value)


def test_analyze_rows_counts_maybe_as_no_positive_vote():
    report = analyze_rows(
        [
            {
                "id": "uncertain",
                "mgRiskScore": "10",
                "mgContentTags": "tag",
                "review_alex": "maybe",
                "review_blair": "?",
            }
        ],
        ["id", "mgRiskScore", "mgContentTags", "review_alex", "review_blair"],
    )

    assert report["images"][0]["needs_review_count"] == 0
    assert report["images"][0]["human_needs_review"] is False


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
            "content_tags": ["people_visible", "text_language_english"],
            "human_needs_review": False,
        },
        {
            "id": "second",
            "title": "Second image",
            "risk_score": 30.0,
            "needs_review_count": 1,
            "reviewer_count": 2,
            "needs_review_percentage": 50.0,
            "content_tags": ["text_language_english"],
            "human_needs_review": False,
        },
    ]
    assert report["reviewers"] == {
        "review_alex": {
            "needs_review": {"average": 10.0, "median": 10.0, "min": 10.0, "max": 10.0},
            "does_not_need_review": {
                "average": 30.0,
                "median": 30.0,
                "min": 30.0,
                "max": 30.0,
            },
        },
        "review_blair": {
            "needs_review": {"average": 30.0, "median": 30.0, "min": 30.0, "max": 30.0},
            "does_not_need_review": {
                "average": 10.0,
                "median": 10.0,
                "min": 10.0,
                "max": 10.0,
            },
        },
    }
    assert report["all_reviewers"] == {
        "needs_review": {"average": 20.0, "median": 20.0, "min": 10.0, "max": 30.0},
        "does_not_need_review": {
            "average": 20.0,
            "median": 20.0,
            "min": 10.0,
            "max": 30.0,
        },
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
    assert (
        report["content_tags"]["text_language_english"]["images_with_tag_total_count"]
        == 2
    )
    assert report["content_tags"]["text_language_english"]["total_responses"] == 4


def test_tag_bayesian_metrics_shrinks_sparse_tags_toward_global_rate():
    metrics = tag_bayesian_metrics(2, 2, 0.3)

    assert metrics["images_majority_agree_needs_review_percentage"] == 100.0
    assert metrics["baseline_smoothed_review_rate_percentage"] == 50.0
    assert metrics["estimated_review_rate_credible_range_min"] < 50.0
    assert metrics["estimated_review_rate_credible_range_max"] > 50.0
    assert metrics["probability_review_rate_exceeds_baseline_percentage"] > 80.0


def test_format_markdown_report_includes_shareable_report_sections():
    report = analyze_rows(
        [
            {
                "id": "image|one",
                "title": "A title\nwith a line break",
                "mgRiskScore": "20",
                "mgContentTags": "people_visible",
                "review_alex": "Y",
            }
        ],
        ["id", "title", "mgRiskScore", "mgContentTags", "review_alex"],
    )

    markdown = format_markdown_report(report)

    assert "# Human Review Analysis" in markdown
    assert "## Reviewer Risk Scores" in markdown
    assert "## Content Tag Review Signals" in markdown
    assert "## Per-Image Review Decisions" in markdown
    assert "image\\|one" in markdown
    assert "A title with a line break" in markdown
    assert "| people_visible |" in markdown


def test_content_tags_are_sorted_by_estimated_review_workload():
    report = analyze_rows(
        [
            {
                "id": "high-one",
                "mgRiskScore": "10",
                "mgContentTags": "high",
                "review_alex": "Y",
                "review_blair": "Y",
                "review_casey": "Y",
            },
            {
                "id": "high-two",
                "mgRiskScore": "20",
                "mgContentTags": "high",
                "review_alex": "Y",
                "review_blair": "Y",
                "review_casey": "Y",
            },
            {
                "id": "medium",
                "mgRiskScore": "30",
                "mgContentTags": "medium",
                "review_alex": "Y",
                "review_blair": "Y",
                "review_casey": "Y",
            },
            {
                "id": "low",
                "mgRiskScore": "40",
                "mgContentTags": "low",
                "review_alex": "N",
                "review_blair": "N",
                "review_casey": "N",
            },
        ],
        [
            "id",
            "mgRiskScore",
            "mgContentTags",
            "review_alex",
            "review_blair",
            "review_casey",
        ],
    )

    assert list(report["content_tags"]) == ["high", "medium", "low"]


def test_analyze_rows_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns: mgContentTags"):
        analyze_rows([], ["id", "mgRiskScore", "review_alex"])


@pytest.mark.parametrize("risk_score", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_analyze_rows_rejects_nonfinite_risk_scores(risk_score):
    with pytest.raises(ValueError, match=r"Row 2 has an invalid mgRiskScore"):
        analyze_rows(
            [
                {
                    "id": "image",
                    "mgRiskScore": risk_score,
                    "mgContentTags": "tag",
                    "review_alex": "Y",
                }
            ],
            ["id", "mgRiskScore", "mgContentTags", "review_alex"],
        )


def test_explorer_metrics_handles_none_any_all_overlap_and_untagged_images():
    images = [
        {"id": "both-positive", "content_tags": ["a", "b"], "human_needs_review": True},
        {"id": "a-negative", "content_tags": ["a"], "human_needs_review": False},
        {"id": "b-positive", "content_tags": ["b"], "human_needs_review": True},
        {"id": "untagged-negative", "content_tags": [], "human_needs_review": False},
    ]

    no_selection = explorer_metrics(images, set(), "any")
    assert no_selection["true_negative"] == 2
    assert no_selection["false_negative"] == 2
    assert no_selection["precision"] is None
    assert explorer_metrics(images, {"a", "b"}, "any")["true_positive"] == 2
    assert explorer_metrics(images, {"a", "b"}, "any")["false_positive"] == 1
    all_metrics = explorer_metrics(images, {"a", "b"}, "all")
    assert all_metrics["true_positive"] == 1
    assert all_metrics["false_negative"] == 1
    assert all_metrics["true_negative"] == 2
    assert all_metrics["precision"] == 1.0
    assert all_metrics["recall"] == 0.5


def test_explorer_metrics_uses_none_for_zero_precision_and_recall_denominators():
    metrics = explorer_metrics(
        [{"id": "negative", "content_tags": [], "human_needs_review": False}],
        set(),
        "any",
    )

    assert metrics["precision"] is None
    assert metrics["recall"] is None


def test_explorer_risk_threshold_is_inclusive_and_composes_with_categories():
    images = [
        {
            "id": "at-threshold-positive",
            "content_tags": [],
            "risk_score": 5.0,
            "needs_review_count": 1,
            "reviewer_count": 1,
        },
        {
            "id": "above-threshold-negative",
            "content_tags": [],
            "risk_score": 6.0,
            "needs_review_count": 0,
            "reviewer_count": 1,
        },
        {
            "id": "below-threshold-positive",
            "content_tags": [],
            "risk_score": 4.0,
            "needs_review_count": 1,
            "reviewer_count": 1,
        },
        {
            "id": "category-only-positive",
            "content_tags": ["a"],
            "risk_score": 1.0,
            "needs_review_count": 1,
            "reviewer_count": 1,
        },
    ]

    assert predicts_review(images[0], set(), "any", 5.0)
    assert not predicts_review(images[2], set(), "any", 5.0)
    assert predicts_review(images[3], {"a"}, "any", 5.0)
    metrics = explorer_metrics(images, {"a"}, "any", risk_threshold=5.0)
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert explorer_metrics(images, {"a"}, "any") == explorer_metrics(
        images, {"a"}, "any", risk_threshold=None
    )


@pytest.mark.parametrize("risk_threshold", [float("nan"), float("inf"), -float("inf")])
def test_explorer_rejects_nonfinite_risk_thresholds(risk_threshold):
    image = {"content_tags": [], "risk_score": 1.0, "human_needs_review": False}

    with pytest.raises(ValueError, match="risk_threshold must be a finite number"):
        predicts_review(image, set(), "any", risk_threshold)
    with pytest.raises(ValueError, match="risk_threshold must be a finite number"):
        explorer_metrics([image], set(), "any", risk_threshold=risk_threshold)


def test_explorer_truth_criteria_recompute_truth_from_normalized_votes():
    images = [
        {
            "id": "one-of-three",
            "content_tags": ["a"],
            "needs_review_count": 1,
            "reviewer_count": 3,
            "human_needs_review": False,
        },
        {
            "id": "two-of-three",
            "content_tags": ["a"],
            "needs_review_count": 2,
            "reviewer_count": 3,
            "human_needs_review": True,
        },
        {
            "id": "three-of-three",
            "content_tags": ["b"],
            "needs_review_count": 3,
            "reviewer_count": 3,
            "human_needs_review": True,
        },
        {
            "id": "two-of-four",
            "content_tags": ["a"],
            "needs_review_count": 2,
            "reviewer_count": 4,
            "human_needs_review": False,
        },
        {
            "id": "one-of-one",
            "content_tags": ["b"],
            "needs_review_count": 1,
            "reviewer_count": 1,
            "human_needs_review": True,
        },
    ]

    assert [human_truth(image, "any") for image in images] == [
        True,
        True,
        True,
        True,
        True,
    ]
    assert [human_truth(image, "half") for image in images] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert [human_truth(image, "majority") for image in images] == [
        False,
        True,
        True,
        False,
        True,
    ]
    assert [human_truth(image, "all") for image in images] == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert explorer_metrics(images, {"a"}, "any", "all")["true_positive"] == 0
    assert explorer_metrics(images, {"a"}, "any", "any")["true_positive"] == 3
    assert explorer_metrics(images, {"a"}, "any")["true_positive"] == 1
    assert explorer_metrics(images, {"a"}, "any", "majority")["true_positive"] == 1
    assert explorer_metrics(images, {"a"}, "any", "half")["true_positive"] == 2


def test_explorer_zero_explanations_and_suggestions_follow_classifier_semantics():
    images = [
        {
            "id": "a-positive",
            "content_tags": ["a"],
            "needs_review_count": 2,
            "reviewer_count": 2,
        },
        {
            "id": "b-positive",
            "content_tags": ["b"],
            "needs_review_count": 2,
            "reviewer_count": 2,
        },
        {
            "id": "d-negative",
            "content_tags": ["d"],
            "needs_review_count": 0,
            "reviewer_count": 2,
        },
        {
            "id": "a-c-positive",
            "content_tags": ["a", "c"],
            "needs_review_count": 2,
            "reviewer_count": 2,
        },
    ]

    assert explorer_zero_result_explanation(images, set(), "any") == (
        "No categories selected, so no images are predicted for review."
    )
    assert explorer_zero_result_explanation(images, {"a", "b"}, "all") == (
        "No image has every selected category, so no images are predicted for review."
    )
    assert explorer_zero_result_explanation(images, {"d"}, "any") == (
        "Matching images exist, but none meet the selected human truth criterion."
    )
    assert explorer_suggestions(images, {"a"}, "any") == [
        {"tag": "b", "additional_caught": 1, "additional_excess": 0}
    ]
    assert explorer_suggestions(images, {"a", "c"}, "all") == [
        {"tag": "c", "additional_caught": 1, "additional_excess": 0}
    ]


def test_explorer_suggestions_and_zero_explanation_keep_threshold_active():
    images = [
        {
            "id": "threshold-positive",
            "content_tags": ["a"],
            "risk_score": 10.0,
            "needs_review_count": 1,
            "reviewer_count": 1,
        },
        {
            "id": "category-positive",
            "content_tags": ["b"],
            "risk_score": 1.0,
            "needs_review_count": 1,
            "reviewer_count": 1,
        },
    ]

    assert explorer_suggestions(images, {"a"}, "any", risk_threshold=5.0) == [
        {"tag": "b", "additional_caught": 1, "additional_excess": 0}
    ]
    assert explorer_zero_result_explanation(
        images, set(), "any", risk_threshold=20.0
    ) == (
        "No images meet the active risk threshold, and no categories are selected, "
        "so no images are predicted for review."
    )
    assert (
        explorer_zero_result_explanation(images, set(), "any", risk_threshold=5.0)
        is None
    )


def test_all_mode_empty_selection_is_an_initialization_exception():
    images = [
        {
            "id": "positive",
            "content_tags": ["a"],
            "needs_review_count": 2,
            "reviewer_count": 2,
        }
    ]

    assert explorer_metrics(images, set(), "all")["true_positive"] == 0
    assert explorer_metrics(images, {"a"}, "all")["true_positive"] == 1


def test_html_explorer_uses_safe_base64_payload_and_renders_expected_controls():
    report = analyze_rows(
        [
            {
                "id": "id</script>",
                "title": "<&\u2028\u2029",
                "mgRiskScore": "1",
                "mgContentTags": "tag</script>",
                "review_a": " y ",
            }
        ],
        ["id", "title", "mgRiskScore", "mgContentTags", "review_a"],
    )

    html = format_html_explorer(report)

    assert "Human Review Tag Explorer" in html
    assert "What this explorer answers" in html
    assert "proposed routing rule" in html
    assert "threshold disabled, the rule flags nothing" in html
    assert "Select all" in html
    assert "tag</script>" not in html
    assert "id</script>" not in html
    assert "atob('" in html
    assert "classificationText].join(' ')" in html
    assert "classificationText])" in html
    assert "Human truth criterion" in html
    assert "At least half flagged" in html
    assert "At least one selected category" in html
    assert "Every selected category" in html
    assert 'id="risk-threshold"' in html
    assert 'id="clear-threshold"' in html
    assert "Number.isFinite(value)" in html
    assert "category match OR score threshold" in html
    assert "Category suggestions hold the active risk threshold fixed" in html
    assert "Votes</dt>" in html
    assert "Red: missed (FN)" in html
    assert "Yellow: excess (FP)" in html
    assert "Suggested category changes" in html
    assert 'data-sort="risk"' in html
    assert 'data-sort="votes"' in html
    assert 'data-sort="classification"' in html
    assert 'aria-sort="none"' in html
    assert "Sort the table:" in html
    assert "Risk score ↕" in html
    assert "Select one category to start the ALL category rule" in html
    assert "DCR_RECORD_URL_BASE" in html
    assert "encodeURIComponent(id)" in html
    assert "createRecordCell(image.id)" in html
    assert "recordCells=new WeakMap()" in html
    assert "recordCellFor(image)" in html
    assert "Thumbnail unavailable" in html
    assert "link.target='_blank'" in html
    assert "link.rel='noopener noreferrer'" in html
    assert "image.loading='lazy'" in html


def test_html_explorer_sorts_votes_by_normalized_rate_with_id_tiebreaker():
    playwright_sync_api = pytest.importorskip("playwright.sync_api")
    report = {
        "images": [
            {
                "id": "one-of-ten",
                "title": "",
                "content_tags": [],
                "risk_score": 0,
                "needs_review_count": 1,
                "reviewer_count": 10,
            },
            {
                "id": "two-of-two",
                "title": "",
                "content_tags": [],
                "risk_score": 0,
                "needs_review_count": 2,
                "reviewer_count": 2,
            },
            {
                "id": "alpha-half",
                "title": "",
                "content_tags": [],
                "risk_score": 0,
                "needs_review_count": 1,
                "reviewer_count": 2,
            },
            {
                "id": "beta-half",
                "title": "",
                "content_tags": [],
                "risk_score": 0,
                "needs_review_count": 2,
                "reviewer_count": 4,
            },
        ],
        "content_tags": [],
    }

    with playwright_sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except playwright_sync_api.Error as error:
            pytest.skip(f"Playwright Chromium is unavailable: {error}")
        try:
            page = browser.new_page()
            page.route("https://dcr-test.lib.unc.edu/**", lambda route: route.abort())
            page.set_content(format_html_explorer(report), wait_until="load")
            assert page.locator("img.thumbnail").evaluate_all(
                "(images) => images.every((image) => image.loading === 'lazy')"
            )

            page.locator('[data-sort="votes"]').click()
            assert page.locator("tbody tr td:nth-child(2)").all_text_contents() == [
                "one-of-ten",
                "alpha-half",
                "beta-half",
                "two-of-two",
            ]

            page.locator('[data-sort="votes"]').click()
            assert page.locator("tbody tr td:nth-child(2)").all_text_contents() == [
                "two-of-two",
                "beta-half",
                "alpha-half",
                "one-of-ten",
            ]
        finally:
            browser.close()


def test_html_explorer_threshold_smoke_blocks_dcr_and_can_be_cleared():
    playwright_sync_api = pytest.importorskip("playwright.sync_api")
    report = analyze_rows(
        [
            {
                "id": "score-only",
                "mgRiskScore": "10",
                "mgContentTags": "",
                "review_a": "Y",
            },
            {
                "id": "category-only",
                "mgRiskScore": "1",
                "mgContentTags": "category",
                "review_a": "N",
            },
        ],
        ["id", "mgRiskScore", "mgContentTags", "review_a"],
    )
    dcr_requests = []

    with playwright_sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except playwright_sync_api.Error as error:
            pytest.skip(f"Playwright Chromium is unavailable: {error}")
        try:
            page = browser.new_page()
            page.route(
                "https://dcr-test.lib.unc.edu/**",
                lambda route: (dcr_requests.append(route.request.url), route.abort()),
            )
            page.set_content(format_html_explorer(report), wait_until="load")
            page.locator("#risk-threshold").fill("10")
            assert (
                page.locator("tbody tr", has_text="score-only").get_attribute("class")
                == "true_positive"
            )
            assert page.locator("#zero-result").is_hidden()
            page.locator("#clear-threshold").click()
            assert (
                page.locator("tbody tr", has_text="score-only").get_attribute("class")
                == "false_negative"
            )
        finally:
            browser.close()

    assert dcr_requests


def test_html_explorer_can_select_inclusive_half_truth_criterion():
    playwright_sync_api = pytest.importorskip("playwright.sync_api")
    report = analyze_rows(
        [
            {
                "id": "two-of-four",
                "mgRiskScore": "1",
                "mgContentTags": "category",
                "review_a": "Y",
                "review_b": "Y",
                "review_c": "N",
                "review_d": "N",
            }
        ],
        [
            "id",
            "mgRiskScore",
            "mgContentTags",
            "review_a",
            "review_b",
            "review_c",
            "review_d",
        ],
    )

    with playwright_sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except playwright_sync_api.Error as error:
            pytest.skip(f"Playwright Chromium is unavailable: {error}")
        try:
            page = browser.new_page()
            page.route("https://dcr-test.lib.unc.edu/**", lambda route: route.abort())
            page.set_content(format_html_explorer(report), wait_until="load")
            page.locator(".tag input").check()

            assert (
                page.locator("tbody tr", has_text="two-of-four").get_attribute("class")
                == "false_positive"
            )
            page.locator("#truth").select_option("half")
            assert (
                page.locator("tbody tr", has_text="two-of-four").get_attribute("class")
                == "true_positive"
            )
        finally:
            browser.close()


def test_dcr_image_urls_encodes_an_id_as_one_path_component():
    assert dcr_image_urls("record /?&#") == {
        "record": "https://dcr-test.lib.unc.edu/record/record%20%2F%3F%26%23",
        "thumbnail": (
            "https://dcr-test.lib.unc.edu/services/api/thumb/"
            "record%20%2F%3F%26%23/small"
        ),
    }


def test_dcr_image_urls_builds_exact_urls_for_a_normal_id():
    assert dcr_image_urls("abc-123") == {
        "record": "https://dcr-test.lib.unc.edu/record/abc-123",
        "thumbnail": "https://dcr-test.lib.unc.edu/services/api/thumb/abc-123/small",
    }


def test_dcr_image_urls_avoids_malformed_urls_for_empty_ids():
    assert dcr_image_urls("") is None
    assert dcr_image_urls(None) is None
    assert dcr_image_urls(".") is None
    assert dcr_image_urls("..") is None


def test_cli_renders_html_format(tmp_path):
    csv_file = tmp_path / "reviews.csv"
    csv_file.write_text(
        "id,mgRiskScore,mgContentTags,review_a\nimage,1,tag,Y\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            "analyze_review_responses.py",
            str(csv_file),
            "--format",
            "html",
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.startswith("<!doctype html>")
    assert "Human Review Tag Explorer" in result.stdout


@pytest.mark.parametrize("format_args", [[], ["--format", "json"]])
def test_cli_renders_json_by_default_and_when_explicit(tmp_path, format_args):
    csv_file = tmp_path / "reviews.csv"
    csv_file.write_text(
        "id,mgRiskScore,mgContentTags,review_a\nimage,1,tag,Y\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            "analyze_review_responses.py",
            str(csv_file),
            *format_args,
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.startswith("{")
