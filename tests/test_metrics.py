"""
Unit tests for core/metrics.py.
"""

import pytest
from core.metrics import calculate_cer, calculate_wer, evaluate_text_accuracy, levenshtein_distance


def test_levenshtein_distance_exact_match():
    assert levenshtein_distance(list("hello"), list("hello")) == 0
    assert levenshtein_distance(["hello", "world"], ["hello", "world"]) == 0


def test_levenshtein_distance_edits():
    assert levenshtein_distance(list("kitten"), list("sitting")) == 3
    assert levenshtein_distance(["a", "b", "c"], ["a", "d", "c"]) == 1


def test_cer_calculation():
    # Exact match
    assert calculate_cer("aiTransform OCR", "aiTransform OCR") == 0.0

    # 1 char difference in 10 chars
    cer = calculate_cer("0123456789", "012345678X")
    assert round(cer, 2) == 0.10

    # Empty cases
    assert calculate_cer("", "") == 0.0
    assert calculate_cer("", "text") == 1.0


def test_wer_calculation():
    # 1 word substitution out of 4
    wer = calculate_wer("Artificial Intelligence transforms enterprise", "Artificial Systems transforms enterprise")
    assert round(wer, 2) == 0.25

    # Exact match
    assert calculate_wer("Digital transformation 2026", "Digital transformation 2026") == 0.0


def test_evaluate_text_accuracy():
    res = evaluate_text_accuracy("Hello World", "Hello World")
    assert res["cer"] == 0.0
    assert res["wer"] == 0.0
    assert res["char_accuracy_pct"] == 100.0
    assert res["word_accuracy_pct"] == 100.0
