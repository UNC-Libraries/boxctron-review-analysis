# Boxctron Review Analysis

Analyze human review decisions collected for images processed by boxctron-describes. The tool reads a CSV export and writes a JSON report to standard output.

## Setup

Create a project-local virtual environment and install the development dependencies:

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -e ".[dev]"
```

The analyzer has no runtime dependencies outside the Python standard library. `pytest` is installed only for running tests.

## Usage

```bash
venv/bin/python analyze_review_responses.py /path/to/reviews.csv
```

Redirect the JSON report to a file if desired:

```bash
venv/bin/python analyze_review_responses.py /path/to/reviews.csv > review-analysis.json
```

## Input CSV

The CSV must contain these columns:

- `id`: image identifier.
- `mgRiskScore`: model risk score for the image.
- `mgContentTags`: comma-separated positive content tags.
- One or more reviewer columns whose names begin with `review`.

An individual reviewer marks an image as needing review only with the literal values `y` or `Y`. Every other value, including a blank response, is treated as not needing review.

## Report

The JSON report has these sections:

- `images`: percentage of reviewers marking each image for review.
- `reviewers`: risk-score summaries grouped by each reviewer's decision.
- `all_reviewers`: the same risk-score summaries with every reviewer decision pooled together.
- `content_tag_analysis`: global image-level baseline and Bayesian prior strength.
- `content_tag_definitions`: definitions for the tag metrics in the next section.
- `content_tags`: reviewer-level and image-level review metrics for each content tag.

Tag uncertainty metrics use a five-image-equivalent prior based on the global image-level review rate. Increase `TAG_PRIOR_STRENGTH_IMAGES` to make sparse-tag estimates more conservative, or decrease it to let tag-specific data have more influence sooner.

## Tests

```bash
venv/bin/pytest
```
