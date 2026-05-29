"""
Generate Information Certainty labels for ver_camf by combining news text and GT.
Steps:
1) Build keyword vocab from corpus (filtered seeds that actually appear).
2) Score each news: high tokens *2 + medium *1 + low *0, adjust with GT trend.
3) Use train scores to compute dynamic thresholds (33rd/66th percentiles).
4) Label each sample: low/medium/high. Save to a new dataset directory.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

INPUT_ROOT = Path("dataset/FNSPID/ver_camf")
OUTPUT_ROOT = Path("dataset/FNSPID/ver_camf_info_certainty")

# Seed keyword candidates; only those that appear in corpus will be kept.
SEED_HIGH = {
    # original
    "confirm",
    "confirmed",
    "definite",
    "certain",
    "guaranteed",
    "assured",
    "will",
    "solid",
    "strong",
    "surge",
    "boost",
    "secured",
    # added high-confidence / positive signals from corpus
    "positive",
    "confidence",
    "growth",
    "bullish",
    "momentum",
    "gains",
    "recovery",
    "stabilization",
    "upward",
    "lead",
}
SEED_MED = {
    # original
    "likely",
    "expected",
    "projected",
    "forecasted",
    "could",
    "may",
    "should",
    "potential",
    "anticipated",
    # added medium/neutral
    "optimism",
    "signals",
}
SEED_LOW = {
    # original
    "uncertain",
    "uncertainty",
    "possibly",
    "unlikely",
    "might",
    "potentially",
    "rumored",
    "tentative",
    "risk",
    "volatility",
    # added low/negative
    "decline",
    "declines",
    "negative",
    "bearish",
    "pressure",
    "risks",
    "temporary",
    "mixed",
    "concerns",
}

POSITIVE_TERMS = {"up", "rise", "rises", "gains", "gain", "increase", "surge", "bullish", "strong"}
NEGATIVE_TERMS = {"down", "drop", "drops", "decline", "declines", "fall", "falls", "bearish", "weak"}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def load_split(path: Path) -> List[Dict]:
    with path.open() as f:
        return json.load(f)


def build_vocab(records: List[Dict]) -> Tuple[set, set, set]:
    counter = Counter()
    for item in records:
        text = str(item.get("news", ""))
        counter.update(tokenize(text))

    high = {tok for tok in SEED_HIGH if counter[tok] > 0}
    med = {tok for tok in SEED_MED if counter[tok] > 0}
    low = {tok for tok in SEED_LOW if counter[tok] > 0}
    return high, med, low


def parse_gt(gt_str: str) -> List[float]:
    return [float(x.strip()) for x in gt_str.split(",")]


def adjust_with_gt(score: int, news_tokens: Counter, gt_str: str) -> int:
    gt_vals = parse_gt(gt_str)
    if len(gt_vals) < 2:
        return score
    pct_change = (gt_vals[-1] - gt_vals[0]) / (abs(gt_vals[0]) + 1e-8)

    pos_flag = any(t in news_tokens for t in POSITIVE_TERMS)
    neg_flag = any(t in news_tokens for t in NEGATIVE_TERMS)

    if pct_change > 0.01 and pos_flag:
        score += 1
    if pct_change < -0.01 and neg_flag:
        score += 1
    if pct_change > 0.01 and neg_flag:
        score -= 1
    if pct_change < -0.01 and pos_flag:
        score -= 1
    if abs(pct_change) < 0.002 and score > 0:
        score -= 1
    return score


def score_text(text: str, gt: str, high_vocab: set, med_vocab: set, low_vocab: set) -> int:
    tokens = tokenize(text)
    token_counts = Counter(tokens)
    score = 0
    score += sum(token_counts[tok] * 2 for tok in high_vocab)
    score += sum(token_counts[tok] * 1 for tok in med_vocab)
    # low tokens contribute 0 by design
    score = adjust_with_gt(score, token_counts, gt)
    return score


def classify(score: float, q_low: float, q_high: float) -> str:
    if score <= q_low:
        return "Low Confidence"
    if score <= q_high:
        return "Medium Confidence"
    return "High Confidence"


def main() -> None:
    all_records = []
    splits = {}
    for split in ["train", "vali", "test"]:
        split_path = INPUT_ROOT / f"{split}.json"
        records = load_split(split_path)
        splits[split] = records
        all_records.extend(records)

    high_vocab, med_vocab, low_vocab = build_vocab(all_records)
    print(f"Vocab sizes -> high:{len(high_vocab)} med:{len(med_vocab)} low:{len(low_vocab)}")

    train_scores = [
        score_text(item.get("news", ""), item["ground_truth"], high_vocab, med_vocab, low_vocab)
        for item in splits["train"]
    ]
    q_low = float(np.percentile(train_scores, 33))
    q_high = float(np.percentile(train_scores, 66))
    print(f"Quantiles (train) -> q_low={q_low:.3f}, q_high={q_high:.3f}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        enriched = []
        label_counter = Counter()
        for item in records:
            score = score_text(item.get("news", ""), item["ground_truth"], high_vocab, med_vocab, low_vocab)
            label = classify(score, q_low, q_high)
            new_item = dict(item)
            new_item["info_certainty_score"] = score
            new_item["info_certainty_label"] = label
            enriched.append(new_item)
            label_counter[label] += 1

        out_path = OUTPUT_ROOT / f"{split}.json"
        out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2))
        print(f"[{split}] saved {len(enriched)} -> {out_path} | label dist: {dict(label_counter)}")


if __name__ == "__main__":
    main()
