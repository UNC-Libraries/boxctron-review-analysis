"""Analyze human review decisions exported in a CSV file.

Usage:
    python analyze_review_responses.py path/to/reviews.csv [--format json|markdown|html]
"""

import argparse
import base64
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
    if not selected_tags:
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


def format_html_explorer(report: dict[str, Any]) -> str:
    """Render a self-contained, file://-safe tag-selection explorer."""
    # Base64 makes JSON data inert inside the script element, including hostile-looking text.
    payload = base64.b64encode(
        json.dumps(
            {"images": report["images"], "content_tags": list(report["content_tags"])}
        ).encode()
    ).decode()
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Review Tag Explorer</title>
<style>
 body{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px;color:#1f2937} button,input,select{font:inherit} .controls{display:flex;gap:.75rem;align-items:center;flex-wrap:wrap}.tags{columns:3 14rem;margin:1rem 0}.tag{display:block;break-inside:avoid;margin:.25rem}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.75rem;margin:1rem 0}.metric{background:#f3f4f6;padding:.75rem;border-radius:.4rem}.metric b{display:block;font-size:1.4rem} table{width:100%;border-collapse:collapse}th,td{border:1px solid #d1d5db;padding:.45rem;text-align:left;vertical-align:top}th button{border:0;background:none;padding:0;color:inherit;font-weight:inherit;cursor:pointer;text-decoration:underline}.sort-direction{display:inline-block;min-width:1em}.true_positive{background:#dcfce7}.false_positive{background:#fef3c7}.false_negative{background:#fee2e2}.true_negative{background:#eff6ff}.legend{display:flex;gap:.5rem;flex-wrap:wrap}.legend span{padding:.3rem .5rem;border-radius:.3rem}.notice{padding:.75rem;background:#f3f4f6;border-left:4px solid #6b7280}.thumbnail{display:block;max-width:120px;max-height:120px;margin-bottom:.3rem}.thumbnail-unavailable{display:block;margin-bottom:.3rem;font-style:italic}.hidden{display:none}
</style></head><body>
<h1>Human Review Tag Explorer</h1>
<section aria-labelledby="purpose-heading"><h2 id="purpose-heading">What this explorer answers</h2><p>It tests a proposed routing rule against human review decisions: if images with selected categories or an optional risk-score threshold were flagged for review, how many human-flagged images would that rule catch, miss, or flag unnecessarily?</p><p>Category ANY/ALL applies only to selected categories. A finite risk threshold independently flags every image with a risk score at or above it, including untagged images; the combined rule is category match OR score threshold. The human truth criterion is the benchmark: any reviewer, at least half, a strict majority (the default), or all reviewers must have marked an image for review. With no categories selected and the threshold disabled, the rule flags nothing, so every human-positive image is missed.</p></section>
<section aria-labelledby="definitions-heading"><h2 id="definitions-heading">Definitions</h2><dl><dt>Caught (TP)</dt><dd>The combined category-or-threshold rule flags an image and it meets the selected human truth criterion.</dd><dt>Excess (FP)</dt><dd>The combined rule flags an image but it does not meet the selected human truth criterion.</dd><dt>Missed (FN)</dt><dd>The combined rule does not flag an image that meets the selected human truth criterion.</dd><dt>Correct bypass (TN)</dt><dd>The combined rule does not flag an image and it does not meet the selected human truth criterion.</dd><dt>Precision</dt><dd>Caught divided by images flagged by the combined rule.</dd><dt>Recall</dt><dd>Caught divided by human-positive images.</dd><dt>Votes</dt><dd>Normalized positive reviewer votes / total reviewers; only trimmed, case-insensitive <code>y</code> is positive.</dd></dl></section>
<section aria-labelledby="legend-heading"><h2 id="legend-heading">Classification colors</h2><div class="legend"><span class="true_positive">Green: caught (TP)</span><span class="false_positive">Yellow: excess (FP)</span><span class="false_negative">Red: missed (FN)</span><span class="true_negative">Blue: correct bypass (TN)</span></div></section>
<div class="controls"><label>Search categories <input id="tag-search" type="search"></label><button id="select-all" type="button">Select all</button><button id="clear" type="button">Clear</button><fieldset><legend>Category matching</legend><label><input type="radio" name="mode" value="any" checked> ANY: At least one selected category</label><label><input type="radio" name="mode" value="all"> ALL: Every selected category</label></fieldset><label>Risk score threshold <input id="risk-threshold" type="number" step="any" inputmode="decimal" aria-describedby="risk-threshold-help"></label><button id="clear-threshold" type="button">Disable threshold</button><span id="risk-threshold-help">Optional; finite score ≥ threshold flags independently of categories.</span><label>Human truth criterion <select id="truth"><option value="any">Any reviewer flagged</option><option value="half">At least half flagged</option><option value="majority" selected>Strict majority flagged</option><option value="all">All reviewers flagged</option></select></label></div>
<p><strong>ALL mode:</strong> selecting the first category starts the category rule; after that, adding categories only restricts category matches. The risk threshold remains independent.</p>
<div id="tags" class="tags"></div><label>Filter images <input id="image-search" type="search"></label>
<p id="zero-result" class="notice" aria-live="polite"></p><section class="metrics" id="metrics"></section>
<section aria-labelledby="suggestions-heading"><h2 id="suggestions-heading">Suggested category changes</h2><p>Exploratory marginal comparisons, not validated recommendations.</p><div id="suggestions" aria-live="polite"></div></section>
 <p><strong>Sort the table:</strong> select the underlined Risk score, Votes, or Classification heading once for ascending order and again to reverse it.</p>
 <table><thead><tr><th>Record</th><th>ID</th><th>Title</th><th>Tags</th><th aria-sort="none"><button type="button" data-sort="risk">Risk score ↕<span class="sort-direction" aria-hidden="true"></span></button></th><th aria-sort="none"><button type="button" data-sort="votes">Votes ↕<span class="sort-direction" aria-hidden="true"></span></button></th><th aria-sort="none"><button type="button" data-sort="classification">Classification ↕<span class="sort-direction" aria-hidden="true"></span></button></th></tr></thead><tbody id="images"></tbody></table>
<script>
const data=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('"""
        + payload
        + """'),c=>c.charCodeAt(0))));
 const DCR_RECORD_URL_BASE='https://dcr-test.lib.unc.edu/record/', DCR_THUMBNAIL_URL_BASE='https://dcr-test.lib.unc.edu/services/api/thumb/';
 const tags=document.querySelector('#tags'), metrics=document.querySelector('#metrics'), imageBody=document.querySelector('#images'), suggestions=document.querySelector('#suggestions'), zeroResult=document.querySelector('#zero-result'), recordCells=new WeakMap(), sortState={key:null,direction:'ascending'};
const selected=()=>new Set([...document.querySelectorAll('.tag input:checked')].map(input=>input.value));
 function truth(image, criterion){return criterion==='any'?image.needs_review_count>0:criterion==='all'?image.needs_review_count===image.reviewer_count:criterion==='half'?image.needs_review_count>=image.reviewer_count/2:image.needs_review_count>image.reviewer_count/2;}
 function categoryPredict(image, chosen, mode){if(!chosen.size)return false;const have=new Set(image.content_tags);return mode==='any'?[...chosen].some(tag=>have.has(tag)):[...chosen].every(tag=>have.has(tag));}
 function activeThreshold(){const input=document.querySelector('#risk-threshold'),value=input.valueAsNumber;return input.value.trim()!==''&&Number.isFinite(value)?value:null;}
 function predict(image, chosen, mode, threshold){return categoryPredict(image,chosen,mode)||(threshold!==null&&image.risk_score>=threshold);}
 function classify(image, chosen, mode, criterion, threshold){const predicted=predict(image,chosen,mode,threshold), human=truth(image,criterion);return predicted?(human?'true_positive':'false_positive'):(human?'false_negative':'true_negative');}
 function classificationLabel(classification){return classification.replaceAll('_',' ');}
 function recordUrls(id){if(typeof id!=='string'||id.length===0||id==='.'||id==='..')return null;const encodedId=encodeURIComponent(id);return {record:DCR_RECORD_URL_BASE+encodedId,thumbnail:DCR_THUMBNAIL_URL_BASE+encodedId+'/small'};}
 function appendThumbnailUnavailable(container){const unavailable=document.createElement('span');unavailable.className='thumbnail-unavailable';unavailable.textContent='Thumbnail unavailable';container.append(unavailable);}
 function createRecordCell(id){const cell=document.createElement('td'), urls=recordUrls(id);if(!urls){appendThumbnailUnavailable(cell);const unavailable=document.createElement('span');unavailable.textContent='Record unavailable';cell.append(unavailable);return cell}const image=document.createElement('img');image.className='thumbnail';image.src=urls.thumbnail;image.alt='Thumbnail for record '+id;image.addEventListener('error',()=>{image.remove();appendThumbnailUnavailable(cell)},{once:true});const link=document.createElement('a');link.href=urls.record;link.target='_blank';link.rel='noopener noreferrer';link.textContent=id;link.setAttribute('aria-label','Open DCR record '+id+' in a new tab');cell.append(image,link);return cell;}
 function recordCellFor(image){let cell=recordCells.get(image);if(!cell){cell=createRecordCell(image.id);recordCells.set(image,cell)}return cell;}
 function countsFor(chosen, mode, criterion, threshold){const counts={true_positive:0,false_positive:0,false_negative:0,true_negative:0};for(const image of data.images)counts[classify(image,chosen,mode,criterion,threshold)]++;return counts}
 function renderSuggestions(chosen, mode, criterion, threshold, baseline){suggestions.replaceChildren();const intro=document.createElement('p');if(mode==='all'){intro.textContent=chosen.size?'Removing a selected restriction can increase caught; adding categories cannot.':'Select one category to start the ALL category rule; after that, adding categories only restricts category matches.';suggestions.append(intro);if(!chosen.size)return;for(const tag of [...chosen].sort()){const candidate=new Set(chosen);candidate.delete(tag);const counts=countsFor(candidate,mode,criterion,threshold), caught=counts.true_positive-baseline.true_positive;if(caught<=0)continue;const item=document.createElement('div');item.textContent=`Remove ${tag}: ${caught>0?'+':''}${caught} caught, ${counts.false_positive-baseline.false_positive>=0?'+':''}${counts.false_positive-baseline.false_positive} excess.`;suggestions.append(item)}}else{intro.textContent='Unchecked categories that increase caught, sorted by additional caught then lower additional excess.';suggestions.append(intro);const items=[];for(const tag of data.content_tags){if(chosen.has(tag))continue;const candidate=new Set(chosen);candidate.add(tag);const counts=countsFor(candidate,mode,criterion,threshold), caught=counts.true_positive-baseline.true_positive;if(caught>0)items.push({tag,caught,excess:counts.false_positive-baseline.false_positive})}items.sort((a,b)=>b.caught-a.caught||a.excess-b.excess||a.tag.localeCompare(b.tag));if(!items.length){const none=document.createElement('p');none.textContent='No unchecked category increases caught for this selection and human truth criterion.';suggestions.append(none)}for(const item of items){const line=document.createElement('div');line.textContent=`Add ${item.tag}: +${item.caught} caught, ${item.excess>=0?'+':''}${item.excess} excess.`;suggestions.append(line)}}const thresholdNote=document.createElement('p');thresholdNote.textContent=threshold===null?'Category suggestions assume the risk threshold is disabled.':'Category suggestions hold the active risk threshold fixed; changing the threshold is separate.';suggestions.append(thresholdNote);}
function compareRows(left,right){if(!sortState.key)return left.index-right.index;let result=0;if(sortState.key==='risk')result=left.image.risk_score-right.image.risk_score;else if(sortState.key==='votes')result=left.image.needs_review_count-right.image.needs_review_count||left.image.reviewer_count-right.image.reviewer_count;else result=left.classificationText.localeCompare(right.classificationText)||String(left.image.id).localeCompare(String(right.image.id))||left.index-right.index;return sortState.direction==='ascending'?result:-result;}
function updateSortControls(){document.querySelectorAll('[data-sort]').forEach(button=>{const active=button.dataset.sort===sortState.key,direction=sortState.direction;button.closest('th').setAttribute('aria-sort',active?direction:'none');button.querySelector('.sort-direction').textContent=active?(direction==='ascending'?'↑':'↓'):'';button.setAttribute('aria-label',active?`${button.dataset.sort==='risk'?'Risk score':button.dataset.sort==='votes'?'Votes':'Classification'}, sorted ${direction}; activate to sort ${direction==='ascending'?'descending':'ascending'}`:`Sort by ${button.dataset.sort==='risk'?'Risk score':button.dataset.sort==='votes'?'Votes':'Classification'}, ascending`)});}
 function render(){const chosen=selected(), mode=document.querySelector('[name=mode]:checked').value, criterion=document.querySelector('#truth').value, threshold=activeThreshold(), query=document.querySelector('#image-search').value.toLowerCase(), counts=countsFor(chosen,mode,criterion,threshold); const rows=[];
    data.images.forEach((image,index)=>{const classification=classify(image,chosen,mode,criterion,threshold), classificationText=classificationLabel(classification);const searchable=[image.id,image.title,...image.content_tags,classificationText].join(' ').toLowerCase();if(searchable.includes(query))rows.push({image,index,classification,classificationText})});rows.sort(compareRows);const renderedRows=rows.map(({image,classification,classificationText})=>{const row=document.createElement('tr');row.className=classification;row.append(recordCellFor(image));for(const value of [image.id,image.title,image.content_tags.join(', '),image.risk_score,image.needs_review_count+'/'+image.reviewer_count,classificationText]){const cell=document.createElement('td');cell.textContent=value;row.append(cell)}return row});
  const precision=counts.true_positive+counts.false_positive, recall=counts.true_positive+counts.false_negative;
   const hasPredictions=data.images.some(image=>predict(image,chosen,mode,threshold));let explanation='';if(!hasPredictions&&threshold!==null)explanation=chosen.size?'No image matches the selected categories or meets the active risk threshold, so no images are predicted for review.':'No images meet the active risk threshold, and no categories are selected, so no images are predicted for review.';else if(!chosen.size)explanation='No categories selected, so no images are predicted for review.';else if(mode==='all'&&threshold===null&&!data.images.some(image=>categoryPredict(image,chosen,mode)))explanation='No image has every selected category, so no images are predicted for review.';else if(counts.true_positive===0&&hasPredictions)explanation='Matching category or score-threshold images exist, but none meet the selected human truth criterion.';zeroResult.textContent=explanation;zeroResult.classList.toggle('hidden',!explanation);
   metrics.replaceChildren(...[['Caught (TP)',counts.true_positive],['Excess (FP)',counts.false_positive],['Missed (FN)',counts.false_negative],['Correct bypass (TN)',counts.true_negative],['Precision',precision?`${(100*counts.true_positive/precision).toFixed(1)}%`:'N/A (no predicted reviews)'],['Recall',recall?`${(100*counts.true_positive/recall).toFixed(1)}%`:'N/A (no human-positive images)']].map(([name,value])=>{const box=document.createElement('div');box.className='metric';const label=document.createElement('span');label.textContent=name;const result=document.createElement('b');result.textContent=value;box.append(label,result);return box}));imageBody.replaceChildren(...renderedRows);updateSortControls();renderSuggestions(chosen,mode,criterion,threshold,counts)}
function renderTags(){tags.replaceChildren(...data.content_tags.map(tag=>{const label=document.createElement('label');label.className='tag';label.dataset.tag=tag.toLowerCase();const input=document.createElement('input');input.type='checkbox';input.value=tag;input.addEventListener('change',render);label.append(input,document.createTextNode(' '+tag));return label}));render()}
document.querySelector('#tag-search').addEventListener('input',event=>document.querySelectorAll('.tag').forEach(label=>label.classList.toggle('hidden',!label.dataset.tag.includes(event.target.value.toLowerCase()))));
 document.querySelector('#image-search').addEventListener('input',render);document.querySelector('#risk-threshold').addEventListener('input',render);document.querySelectorAll('[name=mode]').forEach(input=>input.addEventListener('change',render));document.querySelector('#truth').addEventListener('change',render);
document.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{sortState.direction=sortState.key===button.dataset.sort&&sortState.direction==='ascending'?'descending':'ascending';sortState.key=button.dataset.sort;render()}));
 document.querySelector('#select-all').addEventListener('click',()=>{document.querySelectorAll('.tag input').forEach(input=>input.checked=true);render()});document.querySelector('#clear').addEventListener('click',()=>{document.querySelectorAll('.tag input').forEach(input=>input.checked=false);render()});document.querySelector('#clear-threshold').addEventListener('click',()=>{document.querySelector('#risk-threshold').value='';render()});renderTags();
</script></body></html>
"""
    )


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
