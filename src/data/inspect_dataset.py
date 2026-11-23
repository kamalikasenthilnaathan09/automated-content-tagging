"""Dataset inspection and exploratory quality auditing module.

This module provides reusable functions for loading, validating, and profiling
blog article datasets for multi-label content tagging tasks.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: list[str] = ["post_id", "title", "body", "tags"]
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".csv", ".json", ".jsonl")


def is_missing_value(val: Any) -> bool:
    """Safely determine whether a scalar value represents a missing/null entry.

    Correctly identifies `None`, `np.nan`, `pd.NA`, `pd.NaT`, and float NaN
    without raising truth-value errors on sequences or arrays.

    Parameters
    ----------
    val : Any
        Value to inspect.

    Returns
    -------
    bool
        True if the value is a missing/null indicator, False otherwise.
    """
    if val is None or val is pd.NA or val is pd.NaT:
        return True

    # Collections (lists, arrays, dicts) are not scalar missing values
    if isinstance(val, (list, tuple, set, dict, np.ndarray, pd.Series)):
        return False

    try:
        result = pd.isna(val)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return False
    except Exception:
        return False


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a dataset from CSV, JSON, or JSONL into a Pandas DataFrame.

    Note: JSONL files are currently loaded directly into memory as a Pandas
    DataFrame using `pd.read_json(..., lines=True)`.

    Parameters
    ----------
    file_path : str or Path
        Path to the target dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is unsupported, the file is empty, or parsing fails.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported extensions are: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif ext == ".json":
            try:
                df = pd.read_json(path)
            except ValueError:
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                if isinstance(content, list):
                    df = pd.DataFrame(content)
                elif isinstance(content, dict):
                    for key in ("data", "articles", "posts", "items"):
                        if key in content and isinstance(content[key], list):
                            df = pd.DataFrame(content[key])
                            break
                    else:
                        df = pd.DataFrame([content])
                else:
                    raise ValueError(f"Unrecognized JSON structure in {path}")
        else:
            raise ValueError(f"Unhandled file extension: {ext}")
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"The dataset file is empty: {path}") from err
    except Exception as err:
        if isinstance(err, (ValueError, FileNotFoundError)):
            raise
        raise ValueError(f"Failed to parse dataset file '{path}': {err}") from err

    if df.empty:
        raise ValueError(f"Dataset file '{path}' was loaded but contains no records.")

    return df


def validate_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Check whether the required columns exist in the DataFrame.

    Required columns: 'post_id', 'title', 'body', 'tags'.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Returns
    -------
    dict[str, Any]
        Dictionary with validation results:
        - 'is_valid': bool, True if all required columns are present.
        - 'required_columns': list[str], the expected column names.
        - 'present_columns': list[str], columns currently in the DataFrame.
        - 'missing_columns': list[str], expected columns not present.
    """
    present_cols = [str(col) for col in df.columns]
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in present_cols]

    return {
        "is_valid": len(missing_cols) == 0,
        "required_columns": REQUIRED_COLUMNS.copy(),
        "present_columns": present_cols,
        "missing_columns": missing_cols,
    }


def inspect_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate null counts and blank (empty/whitespace-only) counts per column.

    Distinguishes standard Pandas nulls (`None`, `NaN`, `pd.NA`, `pd.NaT`) from
    non-null entries that contain only whitespace or empty strings.
    Does not modify the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'total_rows': int, total number of records.
        - 'columns': dict[str, dict[str, Any]], per-column breakdown with:
            - 'null_count': int
            - 'null_percentage': float
            - 'blank_count': int (empty or whitespace-only)
            - 'blank_percentage': float
            - 'missing_count': int (null_count + blank_count)
            - 'missing_percentage': float
    """
    total_rows = len(df)
    columns_summary: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]

        # 1. Count actual Pandas nulls safely
        null_count = int(series.apply(is_missing_value).sum())

        # 2. Count non-null blanks (empty strings or whitespace-only strings)
        blank_count = 0
        for val in series:
            if is_missing_value(val):
                continue
            if isinstance(val, str):
                if not val.strip():
                    blank_count += 1
            elif isinstance(val, (list, tuple, set, np.ndarray)):
                if len(val) == 0:
                    blank_count += 1

        total_missing = null_count + blank_count

        null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        blank_pct = round((blank_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        missing_pct = round((total_missing / total_rows * 100), 2) if total_rows > 0 else 0.0

        columns_summary[str(col)] = {
            "null_count": null_count,
            "null_percentage": null_pct,
            "blank_count": blank_count,
            "blank_percentage": blank_pct,
            "missing_count": total_missing,
            "missing_percentage": missing_pct,
        }

    return {
        "total_rows": total_rows,
        "columns": columns_summary,
    }


def parse_raw_tags(val: Any) -> list[str]:
    """Parse a tag entry into a clean list of string tags.

    Handles:
    - `None`, `np.nan`, `pd.NA`, `pd.NaT`, and empty strings -> `[]`
    - Native Python list/tuple/set/numpy array
    - Stringified JSON or Python list (e.g. '["python", "ai"]')
    - Delimited strings (comma, pipe, semicolon)
    - StackOverflow/XML style brackets (<tag1><tag2>)

    Missing values are never stringified into tags like "NA", "NaT", or "None".

    Parameters
    ----------
    val : Any
        Raw tags value from a record.

    Returns
    -------
    list[str]
        List of stripped, non-empty tag strings.
    """
    if is_missing_value(val):
        return []

    # Sequence of tags
    if isinstance(val, (list, tuple, set, np.ndarray)):
        parsed_tags: list[str] = []
        for item in val:
            if is_missing_value(item):
                continue
            item_str = str(item).strip()
            if item_str and item_str.lower() not in ("nan", "none", "null", "nat", "<na>"):
                parsed_tags.append(item_str)
        return parsed_tags

    if not isinstance(val, str):
        val = str(val)

    val = val.strip()
    if not val:
        return []

    # Check for stringified null representations
    if val.lower() in ("nan", "none", "null", "nat", "<na>", "[]", "()"):
        return []

    # 1. Try parsing JSON array or Python literal
    if (val.startswith("[") and val.endswith("]")) or (val.startswith("(") and val.endswith(")")):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parse_raw_tags(parsed)
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple, set)):
                return parse_raw_tags(list(parsed))
        except Exception:
            pass

    # 2. Check for StackOverflow / XML style '<tag1><tag2>'
    if val.startswith("<") and val.endswith(">"):
        import re

        extracted = re.findall(r"<([^>]+)>", val)
        if extracted:
            valid_extracted: list[str] = []
            for t in extracted:
                t_clean = t.strip()
                if t_clean and t_clean.lower() not in ("nan", "none", "null", "nat", "na"):
                    valid_extracted.append(t_clean)
            return valid_extracted

    # 3. Delimited strings (pipe, semicolon, comma)
    for delimiter in ("|", ";", ","):
        if delimiter in val:
            tokens = [t.strip() for t in val.split(delimiter)]
            return [t for t in tokens if t and t.lower() not in ("nan", "none", "null", "nat", "<na>")]

    # Single tag fallback
    return [val]


def analyse_tags(df: pd.DataFrame) -> dict[str, Any]:
    """Parse existing tags and compute frequency distribution and statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'tags' column.

    Returns
    -------
    dict[str, Any]
        Dictionary with tag metrics:
        - 'total_tag_occurrences': int
        - 'unique_tags_count': int
        - 'tag_frequencies': dict[str, int] (sorted descending by count)
        - 'avg_tags_per_post': float
        - 'min_tags_per_post': int
        - 'max_tags_per_post': int
        - 'posts_with_zero_tags': int
    """
    if "tags" not in df.columns:
        return {
            "error": "Column 'tags' not found in DataFrame.",
            "total_tag_occurrences": 0,
            "unique_tags_count": 0,
            "tag_frequencies": {},
            "avg_tags_per_post": 0.0,
            "min_tags_per_post": 0,
            "max_tags_per_post": 0,
            "posts_with_zero_tags": len(df),
        }

    tag_counts: dict[str, int] = {}
    tags_per_post: list[int] = []
    total_occurrences = 0

    for raw_tag_entry in df["tags"]:
        parsed = parse_raw_tags(raw_tag_entry)
        count = len(parsed)
        tags_per_post.append(count)
        total_occurrences += count

        for tag in parsed:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    sorted_frequencies = dict(
        sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
    )

    zero_tag_posts = sum(1 for c in tags_per_post if c == 0)
    avg_per_post = round(float(np.mean(tags_per_post)), 2) if tags_per_post else 0.0
    min_per_post = int(np.min(tags_per_post)) if tags_per_post else 0
    max_per_post = int(np.max(tags_per_post)) if tags_per_post else 0

    return {
        "total_tag_occurrences": total_occurrences,
        "unique_tags_count": len(sorted_frequencies),
        "tag_frequencies": sorted_frequencies,
        "avg_tags_per_post": avg_per_post,
        "min_tags_per_post": min_per_post,
        "max_tags_per_post": max_per_post,
        "posts_with_zero_tags": zero_tag_posts,
    }


def _compute_stats(series: pd.Series, valid_count: int, excluded_count: int) -> dict[str, Any]:
    """Helper to compute summary statistics on a clean, non-empty numeric Series."""
    if series.empty or valid_count == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "valid_records": 0,
            "excluded_empty_records": excluded_count,
        }

    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
        "valid_records": valid_count,
        "excluded_empty_records": excluded_count,
    }


def analyse_article_lengths(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate character and word counts for title and body columns.

    Empty and whitespace-only records are excluded from length calculations so
    they do not distort min, mean, median, or standard deviation. The count of
    excluded records is tracked and reported as `excluded_empty_records`.
    Does not modify the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing 'title' and/or 'body'.

    Returns
    -------
    dict[str, Any]
        Dictionary containing character and word count statistics:
        - 'title_char_stats': dict[str, Any]
        - 'title_word_stats': dict[str, Any]
        - 'body_char_stats': dict[str, Any]
        - 'body_word_stats': dict[str, Any]
    """
    results: dict[str, Any] = {}

    def _extract_valid_texts(col_name: str) -> tuple[pd.Series, int, int]:
        total_rows = len(df)
        valid_items: list[str] = []
        for val in df[col_name]:
            if is_missing_value(val):
                continue
            text = str(val).strip()
            if text:
                valid_items.append(text)

        valid_count = len(valid_items)
        excluded_count = total_rows - valid_count
        return pd.Series(valid_items, dtype=object), valid_count, excluded_count

    # Title analysis
    if "title" in df.columns:
        valid_titles, v_count, ex_count = _extract_valid_texts("title")
        if v_count > 0:
            title_chars = valid_titles.apply(len)
            title_words = valid_titles.apply(lambda s: len(s.split()))
            results["title_char_stats"] = _compute_stats(title_chars, v_count, ex_count)
            results["title_word_stats"] = _compute_stats(title_words, v_count, ex_count)
        else:
            empty_stat = _compute_stats(pd.Series(dtype=float), 0, ex_count)
            results["title_char_stats"] = empty_stat
            results["title_word_stats"] = empty_stat
    else:
        results["title_char_stats"] = {"error": "Column 'title' missing"}
        results["title_word_stats"] = {"error": "Column 'title' missing"}

    # Body analysis
    if "body" in df.columns:
        valid_bodies, v_count, ex_count = _extract_valid_texts("body")
        if v_count > 0:
            body_chars = valid_bodies.apply(len)
            body_words = valid_bodies.apply(lambda s: len(s.split()))
            results["body_char_stats"] = _compute_stats(body_chars, v_count, ex_count)
            results["body_word_stats"] = _compute_stats(body_words, v_count, ex_count)
        else:
            empty_stat = _compute_stats(pd.Series(dtype=float), 0, ex_count)
            results["body_char_stats"] = empty_stat
            results["body_word_stats"] = empty_stat
    else:
        results["body_char_stats"] = {"error": "Column 'body' missing"}
        results["body_word_stats"] = {"error": "Column 'body' missing"}

    return results


def detect_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Identify duplicate IDs, exact duplicate titles, and exact duplicate bodies.

    Blank or whitespace-only titles and bodies are excluded from duplicate
    calculations to prevent empty fields from being counted as duplicate content.
    Does not delete or modify any records.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.

    Returns
    -------
    dict[str, Any]
        Dictionary reporting duplicate counts and percentages.
    """
    total_rows = len(df)

    def _calc_column_duplicates(col_name: str) -> dict[str, Any]:
        if col_name not in df.columns:
            return {
                "available": False,
                "missing_columns": [col_name],
                "duplicate_count": 0,
                "duplicate_percentage": 0.0,
            }

        # Filter out missing and whitespace-only values
        valid_mask = df[col_name].apply(lambda v: not is_missing_value(v) and bool(str(v).strip()))
        valid_entries = df.loc[valid_mask, col_name].astype(str).str.strip()

        count = int(valid_entries.duplicated(keep="first").sum())
        pct = round((count / total_rows * 100), 2) if total_rows > 0 else 0.0

        return {
            "available": True,
            "duplicate_count": count,
            "duplicate_percentage": pct,
            "valid_records_evaluated": int(valid_mask.sum()),
        }

    # Title + Body joint duplicates (both must be non-empty)
    joint_info: dict[str, Any] = {}
    if "title" in df.columns and "body" in df.columns:
        valid_joint_mask = (
            df["title"].apply(lambda v: not is_missing_value(v) and bool(str(v).strip()))
            & df["body"].apply(lambda v: not is_missing_value(v) and bool(str(v).strip()))
        )
        if valid_joint_mask.any():
            clean_pairs = pd.DataFrame(
                {
                    "title": df.loc[valid_joint_mask, "title"].astype(str).str.strip(),
                    "body": df.loc[valid_joint_mask, "body"].astype(str).str.strip(),
                }
            )
            joint_count = int(clean_pairs.duplicated(keep="first").sum())
        else:
            joint_count = 0

        joint_pct = round((joint_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        joint_info = {
            "available": True,
            "duplicate_count": joint_count,
            "duplicate_percentage": joint_pct,
            "valid_records_evaluated": int(valid_joint_mask.sum()),
        }
    else:
        joint_info = {
            "available": False,
            "missing_columns": [c for c in ["title", "body"] if c not in df.columns],
            "duplicate_count": 0,
            "duplicate_percentage": 0.0,
        }

    return {
        "total_rows": total_rows,
        "duplicate_post_id": _calc_column_duplicates("post_id"),
        "duplicate_title": _calc_column_duplicates("title"),
        "duplicate_body": _calc_column_duplicates("body"),
        "duplicate_title_and_body": joint_info,
    }


def generate_report(df: pd.DataFrame) -> str:
    """Combine all dataset inspection results into a human-readable text report.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to profile.

    Returns
    -------
    str
        Formatted string containing the complete inspection report.
    """
    lines: list[str] = [
        "=" * 72,
        " DATASET INSPECTION & QUALITY AUDIT REPORT",
        "=" * 72,
        f"Total Records (Rows): {len(df):,}",
        f"Total Columns: {len(df.columns)}",
        f"Columns Present: {', '.join(df.columns)}",
        "",
        "--- 1. COLUMN SCHEMA VALIDATION ---",
    ]

    col_val = validate_columns(df)
    if col_val["is_valid"]:
        lines.append("  [PASS] All required columns ('post_id', 'title', 'body', 'tags') are present.")
    else:
        lines.append(
            f"  [FAIL] Missing required columns: {', '.join(col_val['missing_columns'])}"
        )

    lines.extend(["", "--- 2. MISSING & BLANK VALUES AUDIT ---"])
    missing_info = inspect_missing_values(df)
    for col_name, stats in missing_info["columns"].items():
        lines.append(
            f"  - {col_name:12s}: Total Unpopulated: {stats['missing_count']:5d} ({stats['missing_percentage']:5.2f}%) "
            f"[Nulls: {stats['null_count']}, Blanks/Whitespace: {stats['blank_count']}]"
        )

    lines.extend(["", "--- 3. DUPLICATE RECORDS AUDIT (Excluding Blanks) ---"])
    dup_info = detect_duplicates(df)
    for key, label in [
        ("duplicate_post_id", "Duplicate post_id"),
        ("duplicate_title", "Duplicate titles"),
        ("duplicate_body", "Duplicate bodies"),
        ("duplicate_title_and_body", "Duplicate (title + body)"),
    ]:
        d = dup_info[key]
        if d.get("available", False):
            lines.append(
                f"  - {label:26s}: {d['duplicate_count']:5d} duplicate(s) "
                f"({d['duplicate_percentage']:5.2f}% of total) from {d.get('valid_records_evaluated', 0)} non-blank records"
            )
        else:
            lines.append(f"  - {label:26s}: N/A (columns missing)")

    lines.extend(["", "--- 4. ARTICLE LENGTH STATISTICS (Non-Blank Records Only) ---"])
    len_info = analyse_article_lengths(df)

    def _fmt_stats(stat_dict: dict[str, Any]) -> str:
        if "error" in stat_dict:
            return f"Error: {stat_dict['error']}"
        return (
            f"mean={stat_dict['mean']:.1f}, median={stat_dict['median']:.1f}, "
            f"min={int(stat_dict['min'])}, max={int(stat_dict['max'])}, std={stat_dict['std']:.1f} "
            f"(Evaluated: {stat_dict.get('valid_records', 0)}, Excluded Blank/Null: {stat_dict.get('excluded_empty_records', 0)})"
        )

    lines.append(f"  - Title Words : {_fmt_stats(len_info['title_word_stats'])}")
    lines.append(f"  - Title Chars : {_fmt_stats(len_info['title_char_stats'])}")
    lines.append(f"  - Body Words  : {_fmt_stats(len_info['body_word_stats'])}")
    lines.append(f"  - Body Chars  : {_fmt_stats(len_info['body_char_stats'])}")

    lines.extend(["", "--- 5. TAGS DISTRIBUTION & FREQUENCIES ---"])
    tag_info = analyse_tags(df)
    if "error" in tag_info:
        lines.append(f"  - {tag_info['error']}")
    else:
        lines.append(f"  - Unique Tags Count       : {tag_info['unique_tags_count']:,}")
        lines.append(f"  - Total Tag Occurrences   : {tag_info['total_tag_occurrences']:,}")
        lines.append(f"  - Average Tags per Post   : {tag_info['avg_tags_per_post']}")
        lines.append(
            f"  - Range of Tags per Post  : {tag_info['min_tags_per_post']} to {tag_info['max_tags_per_post']}"
        )
        lines.append(f"  - Posts with Zero Tags    : {tag_info['posts_with_zero_tags']:,}")

        top_tags = list(tag_info["tag_frequencies"].items())[:10]
        if top_tags:
            lines.append("  - Top 10 Most Frequent Tags:")
            for rank, (tag, freq) in enumerate(top_tags, start=1):
                lines.append(f"      {rank:2d}. {tag:20s}: {freq:5d} occurrences")

    lines.extend(["=" * 72, ""])
    return "\n".join(lines)
