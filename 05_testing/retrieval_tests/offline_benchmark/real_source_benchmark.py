"""Compare local retrieval on existing EngiMate team documents, not XIPU AI."""

import argparse
import json
import re
import statistics
from pathlib import Path

import jieba

from core import all_evidence_covered, bm25_scores, reciprocal_rank_fusion
from run_benchmark import Embedder, QUERY_PREFIX, lexical_tokens, sha256, sort_scores, write_record_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SOURCES = {
    'B01': '02_knowledge_base/02——knowledge_base/B01_需求与配置.md',
    'B02': '02_knowledge_base/02——knowledge_base/B02_任务拆解.md',
    'B03': '02_knowledge_base/02——knowledge_base/B03_分工与排期.md',
    'B04': '02_knowledge_base/02——knowledge_base/B04_会议与风险.md',
    'B05': '02_knowledge_base/02——knowledge_base/B05_输出模板.md',
    'A03': '01_official_sources/source_usage_rules/A03_资料使用与引用规则.md',
    'A06': '01_official_sources/general_engineering_methods/A06_通用工程方法_来源摘编.md',
}


def load_sections(include_source_id=False):
    sections, sources = [], []
    for source_id, relative in SOURCES.items():
        path = ROOT / relative
        text = path.read_text(encoding='utf-8-sig')
        headings = list(re.finditer(r'^##\s+(.+?)\s*$', text, flags=re.MULTILINE))
        assert headings, relative
        for i, heading in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            content = text[heading.start():end].strip()
            if include_source_id:
                content = f'来源文件编号：{source_id}\n' + content
            sections.append({'id': f'{source_id}-{i + 1:02}', 'source': source_id,
                             'heading': heading.group(1), 'text': content})
        sources.append({'id': source_id, 'path': relative, 'sha256': sha256(path)})
    return sections, sources


def resolve_gold(cases, sections):
    for case in cases:
        groups = []
        for group in case['groups']:
            candidates = []
            for alternative in group:
                matches = [section['id'] for section in sections if section['source'] == alternative['source']
                           and alternative['anchor'] in section['text']]
                assert matches, (case['id'], alternative)
                candidates.extend(matches)
            groups.append(sorted(set(candidates)))
        case['required_groups'] = groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'real_source_results.json')
    parser.add_argument('--include-source-id', action='store_true',
                        help='Post-hoc comparison: prepend the real source ID to each section.')
    args = parser.parse_args()
    jieba.setLogLevel(40)
    cases = json.loads((HERE / 'real_source_cases.json').read_text(encoding='utf-8'))
    sections, sources = load_sections(include_source_id=args.include_source_id)
    assert len(cases) == 12
    resolve_gold(cases, sections)
    embedder = Embedder(args.model_dir)
    vectors = embedder.encode([section['text'] for section in sections])
    queries = embedder.encode([QUERY_PREFIX + case['query'] for case in cases])
    tokenized = [lexical_tokens(section['text']) for section in sections]
    ids = [section['id'] for section in sections]
    eligible = set(ids)
    results = []
    for case, query_vec in zip(cases, queries):
        values = bm25_scores(lexical_tokens(case['query']), tokenized)
        lexical = sort_scores({doc_id: value for doc_id, value in zip(ids, values) if value > 0}, eligible)
        vector = sort_scores(dict(zip(ids, vectors @ query_vec)), eligible)
        fused = reciprocal_rank_fusion([[x[0] for x in lexical[:20]], [x[0] for x in vector[:20]]], c=60)
        for method, ranking in [('bm25', lexical), ('dense', vector), ('hybrid_rrf', fused)]:
            top = ranking[:5]
            top_ids = [x[0] for x in top]
            gold = {doc_id for group in case['required_groups'] for doc_id in group}
            first = next((i for i, doc_id in enumerate(top_ids, 1) if doc_id in gold), None)
            results.append({'case_id': case['id'], 'method': method, 'required_groups': case['required_groups'],
                            'top5': [{'id': doc_id, 'score': round(score, 8)} for doc_id, score in top],
                            'hit_at_1': bool(top_ids and top_ids[0] in gold),
                            'all_evidence_at_3': all_evidence_covered(top_ids[:3], case['required_groups']),
                            'reciprocal_rank_at_5': 1 / first if first else 0.0})
    summary = []
    for method in ('bm25', 'dense', 'hybrid_rrf'):
        subset = [r for r in results if r['method'] == method]
        summary.append({'method': method, 'cases': len(subset),
                        'hit_at_1': statistics.mean(r['hit_at_1'] for r in subset),
                        'all_evidence_at_3': statistics.mean(r['all_evidence_at_3'] for r in subset),
                        'mrr_at_5': statistics.mean(r['reciprocal_rank_at_5'] for r in subset)})
    output = {'metadata': {'description': 'Local test on repository texts; no XIPU AI execution.',
                           'source_count': len(sources), 'section_count': len(sections), 'case_count': len(cases),
                           'source_cases_sha256': sha256(HERE / 'real_source_cases.json'),
                           'script_sha256': sha256(HERE / 'real_source_benchmark.py'),
                           'core_sha256': sha256(HERE / 'core.py'),
                           'model_sha256': sha256(args.model_dir / 'model_quantized.onnx'),
                           'chunking': 'Split at Markdown level-2 headings; sections are not platform chunks.',
                           'include_source_id': args.include_source_id,
                           'warning': 'Retrieval of a team statement does not prove that it is correct.'},
              'sources': sources,
              'sections': [{k: section[k] for k in ('id', 'source', 'heading')} for section in sections],
              'cases': cases, 'summary': summary, 'details': results}
    write_record_json(args.output, output)
    print(f"Compared {len(cases)} questions against {len(sections)} sections from {len(sources)} files.")
    for row in summary:
        print(row)
    print(f'Wrote {len(results)} individual results to {args.output}')


if __name__ == '__main__':
    main()
