"""Unit tests for the pure helper functions in app.py.

Importing app.py runs the full Streamlit script (it's not structured as a
library), so these tests exercise the standalone data-processing helpers
that don't depend on Streamlit's runtime session state.
"""
import pandas as pd
import pytest

import app


def test_minmax_100_scales_between_0_and_100():
    result = app._minmax_100(pd.Series([10, 20, 30]))
    assert result.tolist() == [0.0, 50.0, 100.0]


def test_minmax_100_flat_series_returns_50():
    result = app._minmax_100(pd.Series([5, 5, 5]))
    assert result.tolist() == [50.0, 50.0, 50.0]


def test_minmax_100_empty_series_returns_empty():
    result = app._minmax_100(pd.Series([], dtype=float))
    assert result.empty


def test_tokenize_feedback_lowercases_and_drops_short_tokens():
    assert app.tokenize_feedback("Great Fun, a bit Slow at 5 PM!") == ["great", "fun", "bit", "slow"]


def test_tokenize_feedback_handles_non_string_input():
    # str(None) -> "None", which still tokenizes as a 4-letter word.
    assert app.tokenize_feedback(None) == ["none"]


def test_detect_emotion_matches_lexicon_keyword():
    assert app.detect_emotion("The event was really well organized and smooth") == "Trust"


def test_detect_emotion_defaults_to_neutral_with_no_keyword_hits():
    assert app.detect_emotion("It was okay I guess") == "Neutral"


def test_detect_emotion_empty_text_is_neutral():
    assert app.detect_emotion("") == "Neutral"


def test_detect_emotion_breaks_ties_alphabetically():
    # "bad" (Anger) and "hard" (Fear) each match exactly one keyword.
    assert app.detect_emotion("bad and hard") == "Anger"


def test_confidence_score_is_bounded_0_to_100():
    score = app.confidence_score("excellent excellent excellent excellent excellent", 0.9, "Positive")
    assert 0.0 <= score <= 100.0


def test_confidence_score_low_for_empty_feedback():
    assert app.confidence_score("", 0.0, "Neutral") == 0.0


def test_detect_themes_matches_known_theme():
    assert "Logistics" in app.detect_themes("the venue was crowded")


def test_detect_themes_defaults_to_general():
    assert app.detect_themes("nothing relevant here") == ["General"]


def test_build_benchmark_frame_empty_for_missing_dimension():
    df = pd.DataFrame({"State": ["Karnataka"]})
    result = app.build_benchmark_frame(df, "College", {"participants": 1, "rating": 1, "sentiment": 1, "revenue": 1})
    assert result.empty


def test_build_benchmark_frame_ranks_higher_score_first():
    df = pd.DataFrame(
        {
            "Student Name": ["a", "b", "c", "d"],
            "College": ["X", "X", "Y", "Y"],
            "Rating": [5, 5, 1, 1],
            "Sentiment": ["Positive", "Positive", "Negative", "Negative"],
            "Amount Paid": [100, 100, 10, 10],
        }
    )
    weights = {"participants": 1.0, "rating": 1.0, "sentiment": 1.0, "revenue": 1.0}
    bench = app.build_benchmark_frame(df, "College", weights)
    assert bench.iloc[0]["College"] == "X"
    assert bench.iloc[0]["Rank"] == 1
    assert bench.iloc[0]["Benchmark_Score"] > bench.iloc[1]["Benchmark_Score"]


def test_build_benchmark_frame_zero_revenue_does_not_divide_by_zero():
    df = pd.DataFrame(
        {
            "Student Name": ["a", "b"],
            "College": ["X", "Y"],
            "Rating": [5, 4],
            "Sentiment": ["Positive", "Neutral"],
            "Amount Paid": [0, 0],
        }
    )
    weights = {"participants": 1.0, "rating": 1.0, "sentiment": 1.0, "revenue": 1.0}
    bench = app.build_benchmark_frame(df, "College", weights)
    assert (bench["Revenue_Share"] == 0.0).all()
    assert not bench["Revenue_Share"].isna().any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
