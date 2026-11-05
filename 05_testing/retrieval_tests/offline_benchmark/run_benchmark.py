"""Run a synthetic offline BM25, dense-vector and RRF retrieval comparison.

This does not connect to XIPU AI or generate answers. See README.md before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import jieba
import numpy as np
import onnxruntime as ort
import tokenizers
from tokenizers import Tokenizer

from core import all_evidence_covered, bm25_scores, reciprocal_rank_fusion

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / 'fixtures'
MODEL_SHA256 = '15b717c382bcb518ba457b93ea6850ede7f4f1cd8937454aa06972366cd19bcc'
TOKENIZER_SHA256 = '48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26'
QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：'
FIXTURE_FILES = [
    ('env-brief', 'ENV', 'A04_虚构课程要求.md'),
    ('env-state', 'ENV', 'A04_虚构项目状态.md'),
    ('soft-state', 'SOFT', 'A04_虚构软件项目状态.md'),
    ('orion-a', 'ORION', 'A04_冲突通知甲.md'),
    ('orion-b', 'ORION', 'A04_冲突通知乙.md'),
    ('nova-204', 'NOVA', 'A08_虚构_Nova_SEN204.md'),
    ('nova-240', 'NOVA', 'A08_虚构_Nova_SEN240.md'),
    ('vega-204', 'VEGA', 'A08_虚构_Vega_SEN204.md'),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_record_json(path, value):
    """Keep each experimental record on one line for useful Git diffs."""
    lines = ['{']
    items = list(value.items())
    for index, (key, item) in enumerate(items):
        suffix = ',' if index < len(items) - 1 else ''
        if isinstance(item, list):
            lines.append(f'  {json.dumps(key)}: [')
            for row_index, row in enumerate(item):
                comma = ',' if row_index < len(item) - 1 else ''
                lines.append('    ' + json.dumps(row, ensure_ascii=False, separators=(',', ':')) + comma)
            lines.append('  ]' + suffix)
        else:
            lines.append(f'  {json.dumps(key)}: ' + json.dumps(item, ensure_ascii=False, separators=(',', ':')) + suffix)
    lines.append('}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def load_fixtures():
    docs = []
    for doc_id, project, filename in FIXTURE_FILES:
        path = FIXTURES / filename
        docs.append({'id': doc_id, 'project': project, 'text': path.read_text(encoding='utf-8-sig'),
                     'source': 'fixtures/' + filename, 'sha256': sha256(path)})
    return docs


def make_noise(seed, count):
    """Artificial, inspectable distractors; not a natural large-document corpus."""
    rng = random.Random(seed)
    projects = ['ENV', 'NOVA', 'VEGA', 'ORION', 'SOFT']
    activities = ['检查外壳颜色', '讨论界面布局', '整理采购备件', '检查接口标识',
                  '准备演示图片', '安排会议室', '整理软件操作说明', '检查材料包装']
    docs = []
    for i in range(count):
        project = projects[i % len(projects)]
        activity = rng.choice(activities)
        task_num = 100 + rng.randrange(800)
        device_num = 300 + rng.randrange(500)  # never 204 or 240
        pseudo_voltage = rng.choice(['2.1', '2.8', '3.6', '4.2', '4.8'])
        intro = {
            'ENV': '虚构室内环境监测项目进度与组会记录',
            'NOVA': '虚构 Nova 项目器件与活动记录',
            'VEGA': '虚构 Vega 项目器件与活动记录',
            'ORION': '虚构 Project Orion 工作事项记录',
            'SOFT': '虚构预约网页项目协作记录',
        }[project]
        text = (f'{intro}。本段仅供测试。工作编号 T{task_num}，'
                f'讨论{activity}；模拟器件编号 SEN-{device_num}，'
                f'虚构模拟电压 {pseudo_voltage} V，当前资料不包含评分结论。')
        docs.append({'id': f'noise-{seed}-{i:03}', 'project': project, 'text': text,
                     'source': 'synthetic-generated',
                     'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    return docs


TOKEN_PATTERN = re.compile(r'[a-z]+(?:[-_][a-z0-9]+)+|[a-z]+[0-9]+|[a-z]+|[0-9]+|[\u4e00-\u9fff]+', re.I)


def lexical_tokens(text):
    tokens = []
    for match in TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r'[\u4e00-\u9fff]+', match):
            tokens.extend(part for part in jieba.cut(match, HMM=False) if part.strip())
        else:
            tokens.append(match)
            if '-' in match or '_' in match:
                tokens.extend(part for part in re.split('[-_]', match) if part)
    return tokens


class Embedder:
    def __init__(self, model_dir):
        model_file = model_dir / 'model_quantized.onnx'
        tokenizer_file = model_dir / 'tokenizer.json'
        if sha256(model_file) != MODEL_SHA256 or sha256(tokenizer_file) != TOKENIZER_SHA256:
            raise ValueError('Model/tokenizer SHA-256 does not match the pinned experiment files.')
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_file), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.input_names = [value.name for value in self.session.get_inputs()]
        self.cache = {}

    def encode(self, texts, batch_size=16):
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            encoded = self.tokenizer.encode_batch(batch)
            # No document in this controlled corpus may be silently truncated.
            if any(len(item.ids) >= 512 for item in encoded):
                raise ValueError('An input reached the 512-token limit; split and re-annotate it.')
            arrays = {
                'input_ids': np.asarray([item.ids for item in encoded], dtype=np.int64),
                'attention_mask': np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                'token_type_ids': np.asarray([item.type_ids for item in encoded], dtype=np.int64),
            }
            hidden = self.session.run(None, {name: arrays[name] for name in self.input_names})[0]
            vectors = hidden[:, 0, :]
            vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            for text, vector in zip(batch, vectors):
                self.cache[text] = vector
        return np.stack([self.cache[text] for text in texts])


def validate_cases(cases, fixtures):
    by_id = {doc['id']: doc for doc in fixtures}
    assert len(cases) == 18 and len(by_id) == 8
    assert len({case['id'] for case in cases}) == len(cases)
    for case in cases:
        for group in case['required_groups']:
            assert group and all(doc_id in by_id for doc_id in group), case['id']
            assert all(by_id[doc_id]['project'] == case['project'] for doc_id in group), case['id']
        for doc_id, anchors in case['anchors'].items():
            if isinstance(anchors, str):
                anchors = [anchors]
            assert doc_id in by_id
            assert all(anchor in by_id[doc_id]['text'] for anchor in anchors), (case['id'], doc_id)
        assert set(case['anchors']) == {doc_id for group in case['required_groups'] for doc_id in group}


def sort_scores(scores, eligible):
    return sorted(((doc_id, float(score)) for doc_id, score in scores.items()
                   if doc_id in eligible), key=lambda pair: (-pair[1], pair[0]))


def run_once(docs, cases, embedder, query_vectors, seed, noise_count, scope):
    doc_vectors = embedder.encode([doc['text'] for doc in docs])
    doc_ids = [doc['id'] for doc in docs]
    tokenized = [lexical_tokens(doc['text']) for doc in docs]
    details = []
    for case, query_vector in zip(cases, query_vectors):
        eligible = {doc['id'] for doc in docs if not scope or doc['project'] == case['project']}
        lex = bm25_scores(lexical_tokens(case['query']), tokenized)
        lexical = sort_scores({doc_id: val for doc_id, val in zip(doc_ids, lex) if val > 0}, eligible)
        vector = sort_scores(dict(zip(doc_ids, doc_vectors @ query_vector)), eligible)
        # Both methods contribute up to 20 candidates; this is an experiment setting, not a platform claim.
        fused = reciprocal_rank_fusion([[x[0] for x in lexical[:20]], [x[0] for x in vector[:20]]], c=60)
        for method, ranked in [('bm25', lexical), ('dense', vector), ('hybrid_rrf', fused)]:
            top = ranked[:5]
            ids = [doc_id for doc_id, _ in top]
            groups = case['required_groups']
            relevant = {doc_id for group in groups for doc_id in group}
            first = next((i for i, doc_id in enumerate(ids, 1) if doc_id in relevant), None)
            details.append({
                'seed': seed, 'noise_count': noise_count, 'doc_count': len(docs),
                'scope_filter': scope, 'method': method, 'case_id': case['id'],
                'category': case['category'], 'project': case['project'],
                'required_groups': groups, 'top5': [{'id': doc_id, 'score': round(score, 8)} for doc_id, score in top],
                'hit_at_1': None if not groups else bool(ids and ids[0] in relevant),
                'all_evidence_at_3': all_evidence_covered(ids[:3], groups),
                'reciprocal_rank_at_5': None if not groups else (1.0 / first if first else 0.0),
            })
    return details


def summarize(details):
    rows = []
    for scope in (True, False):
        for count in (0, 40, 200):
            for method in ('bm25', 'dense', 'hybrid_rrf'):
                subset = [x for x in details if x['scope_filter'] == scope and x['noise_count'] == count
                          and x['method'] == method and x['required_groups']]
                multi = [x for x in subset if len(x['required_groups']) > 1]
                rows.append({'scope_filter': scope, 'noise_count': count, 'method': method,
                             'answerable_runs': len(subset), 'multi_evidence_runs': len(multi),
                             'hit_at_1': statistics.mean(x['hit_at_1'] for x in subset),
                             'all_evidence_at_3': statistics.mean(x['all_evidence_at_3'] for x in subset),
                             'multi_evidence_at_3': statistics.mean(x['all_evidence_at_3'] for x in multi),
                             'mrr_at_5': statistics.mean(x['reciprocal_rank_at_5'] for x in subset)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'observed_results.json')
    args = parser.parse_args()
    started = time.perf_counter()
    cases = json.loads((HERE / 'cases.json').read_text(encoding='utf-8'))
    fixtures = load_fixtures()
    validate_cases(cases, fixtures)
    jieba.setLogLevel(40)
    embedder = Embedder(args.model_dir)
    query_vectors = embedder.encode([QUERY_PREFIX + case['query'] for case in cases])
    details = []
    manifests = []
    for seed in (11, 19, 29):
        noise = make_noise(seed, 200)
        for count in (0, 40, 200):
            docs = fixtures + noise[:count]
            manifests.append({'seed': seed, 'noise_count': count,
                              'documents': [{k: d[k] for k in ('id', 'project', 'source', 'sha256')} for d in docs]})
            for scope in (True, False):
                details.extend(run_once(docs, cases, embedder, query_vectors, seed, count, scope))
            print(f'Completed seed={seed}, synthetic_distractors={count}', flush=True)
    output = {
        'metadata': {'description': 'Local offline synthetic retrieval experiment; no XIPU AI execution.',
                     'seeds': [11, 19, 29], 'noise_counts': [0, 40, 200], 'scope_conditions': [True, False],
                     'case_count': len(cases), 'answerable_case_count': sum(bool(x['required_groups']) for x in cases),
                     'cases_sha256': sha256(HERE / 'cases.json'), 'model_sha256': MODEL_SHA256,
                     'tokenizer_sha256': TOKENIZER_SHA256, 'model': 'Xenova/bge-small-zh-v1.5 model_quantized.onnx',
                     'embedding': 'CPU ONNX, CLS pooling, L2 normalized, query prefix, max 512 tokens',
                     'bm25': 'positive IDF BM25, k1=1.5, b=0.75; jieba and explicit identifier tokens',
                     'fusion': 'RRF c=60, up to 20 results per method',
                     'elapsed_seconds': round(time.perf_counter() - started, 3),
                     'python': sys.version.split()[0], 'numpy': np.__version__,
                     'onnxruntime': ort.__version__, 'tokenizers': tokenizers.__version__,
                     'jieba': jieba.__version__, 'script_sha256': sha256(HERE / 'run_benchmark.py'),
                     'core_sha256': sha256(HERE / 'core.py')},
        'summary': summarize(details), 'manifests': manifests, 'details': details,
    }
    write_record_json(args.output, output)
    print(f'Wrote {len(details)} raw case/method/condition rows to {args.output}', flush=True)


if __name__ == '__main__':
    main()
