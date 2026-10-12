"""
Metrics calculation module for OCR evaluation.
Computes Character Error Rate (CER), Word Error Rate (WER),
and character/word accuracy using Levenshtein distance.
"""

from typing import Dict, Any, List


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """
    Computes standard Levenshtein distance between two sequences (characters or words).
    Time complexity: O(len(seq1) * len(seq2)) with O(min(len, len)) space.
    """
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)
    if len(seq2) == 0:
        return len(seq1)

    previous_row = list(range(len(seq2) + 1))
    for i, item1 in enumerate(seq1):
        current_row = [i + 1]
        for j, item2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (item1 != item2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Computes Character Error Rate (CER):
    CER = Levenshtein_Distance(reference, hypothesis) / len(reference)
    Normalized to range [0.0, 1.0].
    """
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    distance = levenshtein_distance(ref_chars, hyp_chars)
    return min(1.0, distance / len(ref_chars))


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Computes Word Error Rate (WER):
    WER = Levenshtein_Distance(reference_words, hypothesis_words) / len(reference_words)
    Normalized to range [0.0, 1.0].
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    distance = levenshtein_distance(ref_words, hyp_words)
    return min(1.0, distance / len(ref_words))


def evaluate_text_accuracy(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Calculates comprehensive accuracy metrics between ground truth and OCR prediction.
    """
    cer = calculate_cer(reference, hypothesis)
    wer = calculate_wer(reference, hypothesis)

    char_accuracy = max(0.0, 1.0 - cer) * 100.0
    word_accuracy = max(0.0, 1.0 - wer) * 100.0

    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "char_accuracy_pct": round(char_accuracy, 2),
        "word_accuracy_pct": round(word_accuracy, 2),
        "ground_truth_chars": len(reference),
        "extracted_chars": len(hypothesis),
        "ground_truth_words": len(ref_words),
        "extracted_words": len(hyp_words),
    }
