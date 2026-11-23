# Dataset Guidelines & Quality Standards

This document describes the data contract, format expectations, and data hygiene standards for the **Automated Content Tagging for Blogs** project.

---

## 📌 1. Why Dataset Inspection is Important

In Natural Language Processing (NLP) and multi-label text classification, predictive quality is bounded by data quality ("*garbage in, garbage out*"). Conducting thorough inspection before modeling is essential for the following reasons:

1. **Vocabulary & Tag Explosion**: User-submitted or scraped tags often contain thousands of distinct variants (e.g., typos, pluralizations, combined hashtags). Without inspection, rare tags (appearing once or twice) degrade model generalization and skew multi-label binarization.
2. **Text Length Discrepancies**: Modern Transformer models (such as Sentence Transformers) operate with fixed token context windows (typically 256 to 512 tokens). Articles that exceed these lengths require dedicated text-chunking strategies; articles that are near-empty provide zero signal.
3. **Data Leakage & Duplicate Bias**: Exact or near-duplicate articles between training and test sets artificially inflate evaluation metrics (Micro-F1 / Precision@3). Detecting duplicates early prevents optimistic evaluation bias.
4. **Missing or Corrupted Metadata**: Blank titles or missing body content lead to runtime pipeline failures. Automated inspection flags both null values (`None`, `NaN`, `pd.NA`) and blank/whitespace-only records upfront.

---

## 📋 2. Expected Dataset Structure

Each dataset record represents a published or draft blog post. Datasets are organized under the following subdirectories:
- `data/raw/`: Original, unprocessed datasets as downloaded from open-access sources.
- `data/processed/`: Standardized, cleaned, and normalized datasets.
- `data/splits/`: Immutable train, validation, and test partition splits (`train.csv`, `val.csv`, `test.csv`).

---

## 🏷️ 3. Column Schema

### Required Columns
| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `post_id` | `str` or `int` | Unique identifier for the blog post or article. | `"post_1042"` or `1042` |
| `title` | `str` | Headline or title of the blog post. | `"Getting Started with Pandas DataFrames"` |
| `body` | `str` | Full textual content of the article (markdown, plaintext, or raw HTML). | `"Pandas is an open-source data analysis library..."` |
| `tags` | `list` / `str` | Existing tags assigned to the article. | `["python", "pandas", "data-science"]` |

### Optional Columns
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `published_date` | `str` (ISO 8601) | Date and time the article was published (e.g., `"2024-03-15T10:00:00Z"`). *Not assumed to exist in all datasets.* |
| `author` | `str` | Name or handle of the author. |
| `url` | `str` | Source link of the original blog post. |
| `claps` / `likes` | `int` | Engagement metrics if available. |

---

## 💾 4. Supported File Formats

The ingestion engine supports the following tabular and semi-structured formats:

1. **CSV (`.csv`)**:
   - Standard comma-separated values file.
   - Text fields containing commas or newlines must be quoted using standard RFC 4180 rules.
2. **JSON (`.json`)**:
   - A JSON array of post objects: `[{"post_id": "...", "title": "...", "body": "...", "tags": [...]}, ...]`.
   - Or a top-level dictionary with an articles/posts list key: `{"data": [...]}` or `{"articles": [...]}`.
3. **JSON Lines (`.jsonl`)**:
   - One complete, valid JSON object per line.
   - Records are currently loaded into memory as a Pandas DataFrame using `pandas.read_json(path, lines=True)`.

---

## 🏷️ 5. How Tags Should Be Represented

To support diverse open-access datasets, the ingestion module (`src/data/inspect_dataset.py`) parses tags formatted in any of the following conventions:

1. **JSON Array / Native Python List** *(Recommended)*:
   ```json
   ["machine-learning", "python", "nlp"]
   ```
2. **Stringified JSON Array** (common in CSV exports):
   ```text
   "[\"machine-learning\", \"python\", \"nlp\"]"
   ```
3. **Consistently Delimited Strings**:
   - Comma-delimited: `"machine-learning, python, nlp"`
   - Pipe-delimited: `"machine-learning|python|nlp"`
   - Semicolon-delimited: `"machine-learning; python; nlp"`
4. **StackExchange-Style XML Brackets**:
   ```text
   "<machine-learning><python><nlp>"
   ```

Missing tag entries (`None`, `np.nan`, `pd.NA`, `pd.NaT`, empty strings, or stringified nulls) are safely resolved to empty lists `[]` and are never treated as valid tag labels.
