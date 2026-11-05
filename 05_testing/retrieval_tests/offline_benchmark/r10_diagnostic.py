"""Post-hoc diagnostic for a single failed two-source comparison question."""

import argparse
import json
from pathlib import Path

import jieba

from core import all_evidence_covered, bm25_scores, reciprocal_rank_fusion
from real_source_benchmark import HERE, load_sections, resolve_gold
from run_benchmark import Embedder, QUERY_PREFIX, lexical_tokens, sha256, sort_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'r10_diagnostic_results.json')
    args = parser.parse_args()
    jieba.setLogLevel(40)
    cases = json.loads((HERE / 'real_source_cases.json').read_text(encoding='utf-8'))
    case = next(x for x in cases if x['id'] == 'R10')
    sections, sources = load_sections(include_source_id=False)
    resolve_gold([case], sections)
    subqueries = [
        ('B03', 'B03 文档关于前序任务完成后才启动后续任务的排期原则是什么？'),
        ('A06', 'A06 文档认为硬件未到时，软件能否利用模拟接口并行？'),
    ]
    queries = [case['query']] + [q for _, q in subqueries]
    embedder = Embedder(args.model_dir)
    vectors = embedder.encode([s['text'] for s in sections])
    qvectors = embedder.encode([QUERY_PREFIX + q for q in queries])
    tokenized = [lexical_tokens(s['text']) for s in sections]
    ids = [s['id'] for s in sections]

    def rankings(query, vector, source=None):
        eligible = {s['id'] for s in sections if source is None or s['source'] == source}
        scores = bm25_scores(lexical_tokens(query), tokenized)
        lex = sort_scores({doc_id: value for doc_id, value in zip(ids, scores) if value > 0}, eligible)
        dense = sort_scores(dict(zip(ids, vectors @ vector)), eligible)
        hybrid = reciprocal_rank_fusion([[x[0] for x in lex[:20]], [x[0] for x in dense[:20]]], c=60)
        return {'bm25': lex, 'dense': dense, 'hybrid_rrf': hybrid}

    baseline = rankings(queries[0], qvectors[0])
    unscoped = [rankings(query, qvectors[i + 1]) for i, (_, query) in enumerate(subqueries)]
    scoped = [rankings(query, qvectors[i + 1], source) for i, (source, query) in enumerate(subqueries)]
    details = []
    for method in ('bm25', 'dense', 'hybrid_rrf'):
        for condition, lists in [('single_query_top6', [baseline]),
                                 ('two_queries_unscoped_top3_each', unscoped),
                                 ('two_queries_source_scoped_top3_each', scoped)]:
            per_query = [[doc_id for doc_id, _ in row[method][:(6 if len(lists) == 1 else 3)]] for row in lists]
            combined = list(dict.fromkeys(doc_id for row in per_query for doc_id in row))
            details.append({'method': method, 'condition': condition, 'per_query_ids': per_query,
                            'combined_ids': combined,
                            'all_evidence_covered': all_evidence_covered(combined, case['required_groups'])})
    result = {'metadata': {'description': 'Post-hoc, one case; different query count and conditional source filters.',
                           'case_id': 'R10', 'case_groups': case['required_groups'],
                           'baseline_query': case['query'], 'subqueries': subqueries,
                           'source_sha256': {s['id']: s['sha256'] for s in sources},
                           'script_sha256': sha256(HERE / 'r10_diagnostic.py'),
                           'retrieval_script_sha256': sha256(HERE / 'run_benchmark.py'),
                           'source_script_sha256': sha256(HERE / 'real_source_benchmark.py'),
                           'model_sha256': sha256(args.model_dir / 'model_quantized.onnx'),
                           'warning': 'Post-hoc subqueries use knowledge of the target passages; not a held-out evaluation. Source metadata filter is not access control; no XIPU AI execution.'},
              'details': details}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for item in details:
        print(item['method'], item['condition'], 'all evidence:', item['all_evidence_covered'], item['combined_ids'])


if __name__ == '__main__':
    main()
