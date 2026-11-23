"""Unit tests for dataset inspection module (src/data/inspect_dataset.py).

Uses temporary in-memory data fixtures to verify inspection functions without
creating or persisting training data.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inspect_dataset import (
    analyse_article_lengths,
    analyse_tags,
    detect_duplicates,
    generate_report,
    inspect_missing_values,
    is_missing_value,
    load_dataset,
    parse_raw_tags,
    validate_columns,
)


class TestInspectDataset(unittest.TestCase):
    """Test suite verifying modular inspection functions."""

    def setUp(self) -> None:
        """Create sample in-memory data for testing."""
        self.sample_records = [
            {
                "post_id": "1",
                "title": "Introduction to Python Programming",
                "body": "Python is a versatile programming language used in web and data science.",
                "tags": ["python", "programming", "beginner"],
            },
            {
                "post_id": "2",
                "title": "Machine Learning Fundamentals",
                "body": "Machine learning enables computers to learn patterns from historical data.",
                "tags": '["machine-learning", "python", "ai"]',
            },
            {
                "post_id": "3",
                "title": "Deep Learning with Neural Networks",
                "body": "Neural networks are composed of layers of artificial neurons.",
                "tags": "deep-learning, neural-networks, ai",
            },
            {
                "post_id": "4",
                "title": "Data Visualization with Seaborn",
                "body": "Seaborn provides a high-level interface for statistical graphs.",
                "tags": "python|data-visualization",
            },
            {
                "post_id": "5",
                "title": "Untagged Article Example",
                "body": "This post does not have any tags assigned yet.",
                "tags": None,
            },
        ]
        self.sample_df = pd.DataFrame(self.sample_records)

    # -------------------------------------------------------------------------
    # 1. load_dataset Tests
    # -------------------------------------------------------------------------

    def test_load_dataset_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.csv"
            self.sample_df.to_csv(file_path, index=False)
            loaded_df = load_dataset(file_path)
            self.assertEqual(len(loaded_df), 5)
            self.assertIn("title", loaded_df.columns)

    def test_load_dataset_json_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.sample_records, f)
            loaded_df = load_dataset(file_path)
            self.assertEqual(len(loaded_df), 5)

    def test_load_dataset_json_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({"data": self.sample_records}, f)
            loaded_df = load_dataset(file_path)
            self.assertEqual(len(loaded_df), 5)

    def test_load_dataset_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for rec in self.sample_records:
                    f.write(json.dumps(rec) + "\n")
            loaded_df = load_dataset(file_path)
            self.assertEqual(len(loaded_df), 5)

    def test_load_dataset_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_dataset("non_existent_file.csv")

    def test_load_dataset_unsupported_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.parquet"
            file_path.touch()
            with self.assertRaises(ValueError) as ctx:
                load_dataset(file_path)
            self.assertIn("Unsupported file extension", str(ctx.exception))

    def test_load_dataset_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "empty.csv"
            file_path.touch()
            with self.assertRaises(ValueError):
                load_dataset(file_path)

    # -------------------------------------------------------------------------
    # 2. validate_columns Tests
    # -------------------------------------------------------------------------

    def test_validate_columns_success(self) -> None:
        result = validate_columns(self.sample_df)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["missing_columns"], [])

    def test_validate_columns_missing(self) -> None:
        incomplete_df = self.sample_df.drop(columns=["tags", "body"])
        result = validate_columns(incomplete_df)
        self.assertFalse(result["is_valid"])
        self.assertIn("tags", result["missing_columns"])
        self.assertIn("body", result["missing_columns"])

    # -------------------------------------------------------------------------
    # 3. Missing and Blank Values Tests
    # -------------------------------------------------------------------------

    def test_is_missing_value(self) -> None:
        self.assertTrue(is_missing_value(None))
        self.assertTrue(is_missing_value(np.nan))
        self.assertTrue(is_missing_value(pd.NA))
        self.assertTrue(is_missing_value(pd.NaT))
        self.assertFalse(is_missing_value("hello"))
        self.assertFalse(is_missing_value(0))
        self.assertFalse(is_missing_value(["python"]))
        self.assertFalse(is_missing_value([]))

    def test_inspect_missing_and_blank_values(self) -> None:
        df = pd.DataFrame(
            {
                "post_id": ["1", "2", "3", "4", "5"],
                "title": ["Title A", "", "   ", None, "Title E"],
                "body": ["Body A", "Body B", pd.NA, "   ", "Body E"],
                "tags": [["tag1"], pd.NaT, [], np.nan, "tag2"],
            }
        )
        result = inspect_missing_values(df)
        self.assertEqual(result["total_rows"], 5)

        # Title: 1 null (None), 2 blanks ("", "   ") -> total missing 3
        title_stats = result["columns"]["title"]
        self.assertEqual(title_stats["null_count"], 1)
        self.assertEqual(title_stats["blank_count"], 2)
        self.assertEqual(title_stats["missing_count"], 3)
        self.assertEqual(title_stats["missing_percentage"], 60.0)

        # Body: 1 null (pd.NA), 1 blank ("   ") -> total missing 2
        body_stats = result["columns"]["body"]
        self.assertEqual(body_stats["null_count"], 1)
        self.assertEqual(body_stats["blank_count"], 1)
        self.assertEqual(body_stats["missing_count"], 2)

        # Tags: 2 nulls (pd.NaT, np.nan), 1 blank ([]) -> total missing 3
        tags_stats = result["columns"]["tags"]
        self.assertEqual(tags_stats["null_count"], 2)
        self.assertEqual(tags_stats["blank_count"], 1)
        self.assertEqual(tags_stats["missing_count"], 3)

    # -------------------------------------------------------------------------
    # 4. Tag Parsing and Missing Tag Types Tests
    # -------------------------------------------------------------------------

    def test_parse_raw_tags_missing_values(self) -> None:
        """Verify None, np.nan, pd.NA, pd.NaT, and stringified nulls return []."""
        self.assertEqual(parse_raw_tags(None), [])
        self.assertEqual(parse_raw_tags(np.nan), [])
        self.assertEqual(parse_raw_tags(pd.NA), [])
        self.assertEqual(parse_raw_tags(pd.NaT), [])
        self.assertEqual(parse_raw_tags(""), [])
        self.assertEqual(parse_raw_tags("   "), [])
        self.assertEqual(parse_raw_tags("<NA>"), [])
        self.assertEqual(parse_raw_tags("NaT"), [])
        self.assertEqual(parse_raw_tags("nan"), [])
        self.assertEqual(parse_raw_tags("None"), [])
        self.assertEqual(parse_raw_tags("[]"), [])

    def test_parse_raw_tags_list_containing_missing(self) -> None:
        """Verify embedded nulls within tag lists are pruned without stringifying."""
        tags_list = [None, pd.NA, "python", np.nan, "machine-learning", pd.NaT, "  "]
        self.assertEqual(parse_raw_tags(tags_list), ["python", "machine-learning"])

    def test_parse_raw_tags_formats(self) -> None:
        self.assertEqual(parse_raw_tags(["python", "ai"]), ["python", "ai"])
        self.assertEqual(parse_raw_tags('["python", "ai"]'), ["python", "ai"])
        self.assertEqual(parse_raw_tags("python, ai, ml"), ["python", "ai", "ml"])
        self.assertEqual(parse_raw_tags("python|ai|ml"), ["python", "ai", "ml"])
        self.assertEqual(parse_raw_tags("python; ai; ml"), ["python", "ai", "ml"])
        self.assertEqual(parse_raw_tags("<python><ai>"), ["python", "ai"])
        self.assertEqual(parse_raw_tags("single-tag"), ["single-tag"])

    def test_analyse_tags_with_various_missing(self) -> None:
        df = pd.DataFrame(
            {
                "tags": [
                    ["python", "nlp"],
                    pd.NA,
                    pd.NaT,
                    np.nan,
                    None,
                    "",
                    "   ",
                    "<NA>",
                    "python",
                ]
            }
        )
        stats = analyse_tags(df)
        self.assertEqual(stats["posts_with_zero_tags"], 7)
        self.assertEqual(stats["unique_tags_count"], 2)  # python, nlp
        self.assertEqual(stats["tag_frequencies"]["python"], 2)
        self.assertEqual(stats["tag_frequencies"]["nlp"], 1)
        self.assertNotIn("NA", stats["tag_frequencies"])
        self.assertNotIn("NaT", stats["tag_frequencies"])
        self.assertNotIn("nan", stats["tag_frequencies"])

    # -------------------------------------------------------------------------
    # 5. Article Lengths (Excluding Blanks/Nulls) Tests
    # -------------------------------------------------------------------------

    def test_analyse_article_lengths_excludes_blanks(self) -> None:
        df = pd.DataFrame(
            {
                # 2 valid titles, 3 blank/null
                "title": ["Short Title", "A Somewhat Longer Title Here", "", "   ", None],
                # 2 valid bodies (10 words and 20 words), 3 blank/null
                "body": [
                    "one two three four five six seven eight nine ten",
                    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
                    pd.NA,
                    "   ",
                    None,
                ],
            }
        )
        stats = analyse_article_lengths(df)

        body_words = stats["body_word_stats"]
        # Valid records must be 2, excluded must be 3
        self.assertEqual(body_words["valid_records"], 2)
        self.assertEqual(body_words["excluded_empty_records"], 3)
        # Min words must be 10, NOT 0!
        self.assertEqual(body_words["min"], 10.0)
        self.assertEqual(body_words["max"], 20.0)
        self.assertEqual(body_words["mean"], 15.0)
        self.assertEqual(body_words["median"], 15.0)

        title_words = stats["title_word_stats"]
        self.assertEqual(title_words["valid_records"], 2)
        self.assertEqual(title_words["excluded_empty_records"], 3)
        self.assertEqual(title_words["min"], 2.0)  # "Short Title" -> 2 words

    # -------------------------------------------------------------------------
    # 6. Duplicate Detection (Excluding Blanks) Tests
    # -------------------------------------------------------------------------

    def test_detect_duplicates_ignores_blanks(self) -> None:
        df = pd.DataFrame(
            {
                "post_id": ["1", "2", "3", "4", "5", "6"],
                # "Machine Learning" is duplicated once.
                # Three records are blank or whitespace only.
                "title": ["Machine Learning", "Machine Learning", "", "   ", None, "Deep Learning"],
                "body": ["Content A", "Content A", "   ", None, "", "Content B"],
            }
        )
        dups = detect_duplicates(df)

        # Duplicate title should be 1 ("Machine Learning" appears twice -> 1 duplicate)
        # The blank/whitespace titles MUST NOT be counted as duplicates!
        self.assertEqual(dups["duplicate_title"]["duplicate_count"], 1)
        self.assertEqual(dups["duplicate_title"]["valid_records_evaluated"], 3)

        # Duplicate body should be 1 ("Content A" appears twice -> 1 duplicate)
        self.assertEqual(dups["duplicate_body"]["duplicate_count"], 1)
        self.assertEqual(dups["duplicate_body"]["valid_records_evaluated"], 3)

        # Joint duplicate (title + body) should be 1
        self.assertEqual(dups["duplicate_title_and_body"]["duplicate_count"], 1)
        self.assertEqual(dups["duplicate_title_and_body"]["valid_records_evaluated"], 3)

    # -------------------------------------------------------------------------
    # 7. DataFrame Immutability Test
    # -------------------------------------------------------------------------

    def test_dataframe_immutability(self) -> None:
        """Confirm that all inspection functions preserve the input DataFrame unchanged."""
        df_original = self.sample_df.copy(deep=True)

        validate_columns(self.sample_df)
        inspect_missing_values(self.sample_df)
        analyse_tags(self.sample_df)
        analyse_article_lengths(self.sample_df)
        detect_duplicates(self.sample_df)
        generate_report(self.sample_df)

        pd.testing.assert_frame_equal(self.sample_df, df_original)

    # -------------------------------------------------------------------------
    # 8. Report Generation Test
    # -------------------------------------------------------------------------

    def test_generate_report(self) -> None:
        report = generate_report(self.sample_df)
        self.assertIsInstance(report, str)
        self.assertIn("DATASET INSPECTION & QUALITY AUDIT REPORT", report)
        self.assertIn("COLUMN SCHEMA VALIDATION", report)
        self.assertIn("MISSING & BLANK VALUES AUDIT", report)
        self.assertIn("DUPLICATE RECORDS AUDIT", report)
        self.assertIn("ARTICLE LENGTH STATISTICS", report)
        self.assertIn("TAGS DISTRIBUTION & FREQUENCIES", report)
        self.assertIn("Top 10 Most Frequent Tags", report)


if __name__ == "__main__":
    unittest.main()
