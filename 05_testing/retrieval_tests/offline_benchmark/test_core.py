"""Independent checks for the offline retrieval experiment."""

import math
import random
import unittest

import numpy as np
from rank_bm25 import BM25Okapi

from core import all_evidence_covered, bm25_scores, reciprocal_rank_fusion


class RetrievalCoreTests(unittest.TestCase):
    def test_lecture_bm25_example_and_unmatched_term(self):
        corpus = [
            ['won'] + ['a'] * 7,
            ['b'] * 12,
            ['won'] + ['c'] * 13,
        ]
        scores = bm25_scores(['won'], corpus, k1=1.5, b=0.75)
        self.assertAlmostEqual(scores[0], 0.54, places=2)
        self.assertEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], 0.43, places=2)

    def test_rrf_uses_ranks_and_combines_overlapping_results(self):
        result = reciprocal_rank_fusion([['A', 'B', 'C'], ['B', 'C', 'A']], c=60)
        self.assertEqual([doc for doc, _ in result], ['B', 'A', 'C'])
        scores = dict(result)
        self.assertAlmostEqual(scores['B'], 1 / 62 + 1 / 61, places=12)

    def test_multiple_required_evidence_groups(self):
        required = [['brief'], ['project_state']]
        self.assertFalse(all_evidence_covered(['brief', 'other'], required))
        self.assertTrue(all_evidence_covered(['other', 'project_state', 'brief'], required))

    def test_empty_answer_set_is_not_marked_as_evidence_covered(self):
        self.assertIsNone(all_evidence_covered(['anything'], []))

    def test_bm25_against_independent_library_with_matched_idf(self):
        rng = random.Random(37)
        vocabulary = ['alpha', 'beta', 'T04', 'sensor', 'report', 'gamma']
        for _ in range(30):
            corpus = [[rng.choice(vocabulary) for _ in range(rng.randint(1, 12))]
                      for _ in range(rng.randint(2, 9))]
            query = [rng.choice(vocabulary) for _ in range(rng.randint(1, 3))]
            reference = BM25Okapi(corpus, k1=1.5, b=0.75)
            # rank_bm25's default IDF differs; align only that definition before comparison.
            reference.idf = {
                term: math.log(1 + (len(corpus) - sum(term in doc for doc in corpus) + 0.5)
                               / (sum(term in doc for doc in corpus) + 0.5))
                for term in vocabulary
            }
            self.assertTrue(np.allclose(bm25_scores(query, corpus), reference.get_scores(query), atol=1e-12))


if __name__ == '__main__':
    unittest.main()
