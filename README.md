# Boxctron Review Analysis

Analyze human review decisions collected for images processed by boxctron-describes. The tool reads a CSV export and writes a JSON report to standard output.

## Setup

Create a project-local virtual environment and install the development dependencies:

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -e ".[dev]"
```

The analyzer uses SciPy for Bayesian credible ranges and evidence calculations. `pytest` is installed only for running tests.

## Usage

```bash
venv/bin/python analyze_review_responses.py /path/to/reviews.csv
```

The default output is JSON. Generate a shareable Markdown report with:

```bash
venv/bin/python analyze_review_responses.py /path/to/reviews.csv --format markdown
```

Generate a self-contained, dependency-free HTML tag explorer (which works when opened using `file://`) with:

```bash
venv/bin/python analyze_review_responses.py /path/to/reviews.csv --format html > review-explorer.html
```

Redirect either report format to a file if desired:

```bash
venv/bin/python analyze_review_responses.py /path/to/reviews.csv --format markdown > review-analysis.md
```

## Input CSV

The CSV must contain these columns:

- `id`: image identifier.
- `mgRiskScore`: finite numeric model risk score for the image.
- `mgContentTags`: comma-separated positive content tags.
- One or more reviewer columns whose names begin with `review`.

An individual reviewer marks an image as needing review when their value, after surrounding whitespace is removed, is `y` case-insensitively. Every other value, including `maybe`, `?`, a blank response, or `n`, is treated as not needing review. An image needs review only when a strict majority of reviewers mark it; ties do not need review.

## Report

The JSON report has these sections:

- `images`: percentage of reviewers marking each image for review, stable sorted/deduplicated `content_tags`, and strict-majority `human_needs_review`.
- `reviewers`: risk-score summaries grouped by each reviewer's decision.
- `all_reviewers`: the same risk-score summaries with every reviewer decision pooled together.
- `content_tag_analysis`: global image-level baseline and Bayesian prior strength.
- `content_tag_definitions`: definitions for the tag metrics in the next section.
- `content_tags`: reviewer-level and image-level review metrics for each content tag, ordered by estimated review workload.

Tag uncertainty metrics use a five-image-equivalent prior based on the global image-level review rate. Increase `TAG_PRIOR_STRENGTH_IMAGES` to make sparse-tag estimates more conservative, or decrease it to let tag-specific data have more influence sooner.

The HTML explorer evaluates a proposed routing rule against human review decisions. **Category matching** predicts review with **ANY** (at least one selected category) or **ALL** (every selected category); in ALL mode, selecting the first category starts the category rule, then further categories only restrict category matches. An optional finite **risk score threshold** independently flags every image whose score is greater than or equal to it, including untagged images: the combined prediction is category match **OR** threshold. Blank, invalid, and disabled thresholds are off. Category suggestions hold the active threshold fixed and do not recommend threshold changes. **Human truth criterion** treats an image as human-positive when any reviewer flagged it, at least half flagged it, a strict majority flagged it (the default and report semantics), or all reviewers flagged it. The inclusive half criterion uses positive votes ≥ reviewer count / 2, so a 2-of-4 tie is positive. Votes are normalized positive reviewer votes / total reviewers. With no categories selected and the threshold disabled, the proposed rule flags nothing, so every human-positive image is missed.

It labels predicted/human outcomes as caught (true positive, green), excess (false positive, yellow), missed (false negative, red), and correct bypass (true negative, blue). Precision is caught divided by predicted reviews and is shown as N/A if no image is predicted for review; recall is caught divided by human-positive images and is shown as N/A if there are no human-positive images. The suggested category changes are exploratory one-category marginal comparisons, not validated recommendations.

The HTML table starts in CSV order; select Risk score, Votes, or Classification to sort rendered rows ascending, then select it again to reverse the order.

Each HTML row includes a DCR thumbnail and an ID-labeled record link. These browser requests require DCR VPN/authentication; the explorer makes no server-side requests and remains usable from `file://`. If a thumbnail cannot load, it is removed and `Thumbnail unavailable` is shown while the record link remains available. Rows with an empty, `.` or `..` ID show unavailable states rather than a malformed URL.

## Tests

```bash
venv/bin/pytest
```

The HTML explorer smoke test uses Playwright Chromium; after installing development dependencies, install its browser once with `venv/bin/playwright install chromium`.
