"""Unit tests for the pure helper functions in app.py.

Importing app.py runs the full Streamlit script (it's not structured as a
library), so these tests exercise the standalone data-processing helpers
that don't depend on Streamlit's runtime session state.
"""
import pandas as pd
import pytest

import app


def test_chart_layout_uppercases_title_and_uses_default_height():
    layout = app.chart_layout("revenue by event")
    assert layout["title"]["text"] == "REVENUE BY EVENT"
    assert layout["height"] == 400


def test_chart_layout_respects_custom_height():
    layout = app.chart_layout("Custom", height=520)
    assert layout["height"] == 520


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


def test_compute_risk_score_ranks_worse_metrics_higher():
    negative_rate = pd.Series([0.0, 50.0, 100.0])
    avg_rating = pd.Series([5.0, 3.0, 1.0])
    avg_confidence = pd.Series([100.0, 50.0, 0.0])
    scores = app.compute_risk_score(negative_rate, avg_rating, avg_confidence)
    assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2]
    assert scores.iloc[0] == 0.0
    assert scores.iloc[2] == 100.0


def test_compute_risk_score_weights_negative_rate_most_heavily():
    # Worst negative rate alone should outweigh worst rating alone at these weights (0.5 vs 0.35).
    worst_negative_only = app.compute_risk_score(
        pd.Series([0.0, 100.0]), pd.Series([5.0, 5.0]), pd.Series([100.0, 100.0])
    )
    worst_rating_only = app.compute_risk_score(
        pd.Series([0.0, 0.0]), pd.Series([5.0, 1.0]), pd.Series([100.0, 100.0])
    )
    assert worst_negative_only.iloc[1] > worst_rating_only.iloc[1]


def test_classify_sentiment_positive_above_threshold():
    assert app.classify_sentiment(0.06) == "Positive"


def test_classify_sentiment_negative_below_threshold():
    assert app.classify_sentiment(-0.06) == "Negative"


def test_classify_sentiment_neutral_within_band():
    assert app.classify_sentiment(0.0) == "Neutral"
    assert app.classify_sentiment(0.05) == "Neutral"
    assert app.classify_sentiment(-0.05) == "Neutral"


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


def test_build_benchmark_frame_all_zero_weights_does_not_divide_by_zero():
    df = pd.DataFrame(
        {
            "Student Name": ["a", "b", "c", "d"],
            "College": ["X", "X", "Y", "Y"],
            "Rating": [5, 5, 1, 1],
            "Sentiment": ["Positive", "Positive", "Negative", "Negative"],
            "Amount Paid": [100, 100, 10, 10],
        }
    )
    weights = {"participants": 0.0, "rating": 0.0, "sentiment": 0.0, "revenue": 0.0}
    bench = app.build_benchmark_frame(df, "College", weights)
    assert (bench["Benchmark_Score"] == 0.0).all()


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


def test_build_benchmark_frame_all_nan_dimension_returns_empty():
    # groupby drops NaN keys, so an all-NaN dimension column yields zero groups
    # even though the dimension column itself exists.
    df = pd.DataFrame(
        {
            "Student Name": ["a", "b"],
            "College": [None, None],
            "Rating": [5, 4],
            "Sentiment": ["Positive", "Neutral"],
            "Amount Paid": [10, 20],
        }
    )
    weights = {"participants": 1.0, "rating": 1.0, "sentiment": 1.0, "revenue": 1.0}
    bench = app.build_benchmark_frame(df, "College", weights)
    assert bench.empty


def test_oracle_reports_top_event_state_and_negative_sentiment_alert():
    df = pd.DataFrame(
        {
            "Event Name": ["A", "A", "B"],
            "State": ["Karnataka", "Karnataka", "Kerala"],
            "Sentiment": ["Positive", "Negative", "Negative"],
            "Rating": [5, 2, 3],
            "Amount Paid": [100, 50, 200],
            "Event Type": ["Individual", "Group", "Individual"],
            "College": ["X", "X", "Y"],
        }
    )
    insights = app.oracle(df)
    assert any("**A**" in line and "dominates" in line for line in insights)
    assert any("**Karnataka**" in line and "leads" in line for line in insights)
    # 2/3 negative is above the 15% alert threshold.
    assert any("Post-mortem recommended" in line for line in insights)


def test_oracle_single_state_does_not_recommend_expanding_to_itself():
    df = pd.DataFrame(
        {
            "Event Name": ["A", "A", "B"],
            "State": ["Karnataka", "Karnataka", "Karnataka"],
            "Sentiment": ["Positive", "Positive", "Neutral"],
            "Rating": [5, 4, 5],
            "Amount Paid": [100, 100, 100],
            "Event Type": ["Individual", "Individual", "Individual"],
            "College": ["X", "X", "X"],
        }
    )
    insights = app.oracle(df)
    assert any("only state represented" in line for line in insights)
    assert not any("Expand outreach" in line for line in insights)


def test_oracle_single_event_does_not_report_gap_against_itself():
    df = pd.DataFrame(
        {
            "Event Name": ["A", "A", "A"],
            "State": ["Karnataka", "Kerala", "Karnataka"],
            "Sentiment": ["Positive", "Positive", "Neutral"],
            "Rating": [5, 4, 5],
            "Amount Paid": [100, 100, 100],
            "Event Type": ["Individual", "Individual", "Individual"],
            "College": ["X", "X", "X"],
        }
    )
    insights = app.oracle(df)
    assert any("Only one event in this slice" in line for line in insights)
    assert not any("Rating gap" in line for line in insights)


def test_oracle_healthy_sentiment_message_below_threshold():
    df = pd.DataFrame(
        {
            "Event Name": ["A", "A", "A"],
            "State": ["Karnataka", "Karnataka", "Karnataka"],
            "Sentiment": ["Positive", "Positive", "Neutral"],
            "Rating": [5, 4, 5],
            "Amount Paid": [100, 100, 100],
            "Event Type": ["Individual", "Individual", "Individual"],
            "College": ["X", "X", "X"],
        }
    )
    insights = app.oracle(df)
    assert any("within healthy range" in line for line in insights)


def test_oracle_empty_data_returns_single_message():
    assert app.oracle(pd.DataFrame()) == ["No data matches current filters."]


def test_get_wc_returns_none_when_only_stopwords():
    assert app.get_wc(["the", "and", "is"], "GnBu") is None


def test_get_wc_returns_none_for_empty_input():
    assert app.get_wc([], "GnBu") is None
    assert app.get_wc(["", None], "GnBu") is None


def test_get_wc_excludes_stopwords_and_short_tokens():
    wc = app.get_wc(["The event was fantastic and the food was great"], "GnBu")
    assert wc is not None
    words = wc.words_
    assert "event" in words
    assert "fantastic" in words
    assert "food" in words
    assert "great" in words
    assert "the" not in words
    assert "and" not in words
    assert "was" not in words


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
