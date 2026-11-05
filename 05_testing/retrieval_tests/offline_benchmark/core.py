"""Core scoring functions for the offline feasibility experiment."""

from collections import Counter
from math import log


def bm25_scores(query_tokens, corpus_tokens, k1=1.5, b=0.75):
    """BM25 with Lucene's positive-IDF form; corpus elements are token lists."""
    if not corpus_tokens:
        return []
    average_length = sum(map(len, corpus_tokens)) / len(corpus_tokens)
    if average_length == 0:
        return [0.0] * len(corpus_tokens)
    counts = [Counter(tokens) for tokens in corpus_tokens]
    df = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    n = len(corpus_tokens)
    scores = []
    for tokens, frequencies in zip(corpus_tokens, counts):
        value = 0.0
        for term in query_tokens:
            tf = frequencies[term]
            if not tf:
                continue
            idf = log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denominator = tf + k1 * (1 - b + b * len(tokens) / average_length)
            value += idf * tf * (k1 + 1) / denominator
        scores.append(value)
    return scores


def reciprocal_rank_fusion(ranked_lists, c=60):
    scores = Counter()
    for ranked in ranked_lists:
        seen = set()
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id not in seen:
                scores[doc_id] += 1 / (c + rank)
                seen.add(doc_id)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def all_evidence_covered(retrieved_ids, required_groups):
    if not required_groups:
        return None
    retrieved = set(retrieved_ids)
    return all(any(doc_id in retrieved for doc_id in group) for group in required_groups)
