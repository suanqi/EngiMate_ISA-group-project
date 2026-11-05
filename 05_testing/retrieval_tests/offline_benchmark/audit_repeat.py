"""Independently audit two saved runs of the offline benchmark."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
first_path, repeat_path = HERE / 'observed_results.json', HERE / 'observed_repeat.json'
first = json.loads(first_path.read_text(encoding='utf-8'))
repeat = json.loads(repeat_path.read_text(encoding='utf-8'))
assert first['metadata']['script_sha256'] == repeat['metadata']['script_sha256']
assert len(first['details']) == len(repeat['details']) == 972
assert first['summary'] == repeat['summary']
assert first['manifests'] == repeat['manifests']
max_delta = 0.0
for x, y in zip(first['details'], repeat['details']):
    for key in ('seed', 'noise_count', 'scope_filter', 'method', 'case_id', 'required_groups',
                'hit_at_1', 'all_evidence_at_3', 'reciprocal_rank_at_5'):
        assert x[key] == y[key], (key, x['case_id'])
    assert [t['id'] for t in x['top5']] == [t['id'] for t in y['top5']], x['case_id']
    for a, b in zip(x['top5'], y['top5']):
        max_delta = max(max_delta, abs(a['score'] - b['score']))
assert max_delta < 0.00001
manifests = {(m['seed'], m['noise_count']): {d['id']: d['project'] for d in m['documents']}
             for m in first['manifests']}
for x in first['details']:
    projects = manifests[x['seed'], x['noise_count']]
    assert all(t['id'] in projects for t in x['top5'])
    if x['scope_filter']:
        assert all(projects[t['id']] == x['project'] for t in x['top5'])
    groups, ids = x['required_groups'], [t['id'] for t in x['top5'][:3]]
    expected = None if not groups else all(any(d in ids for d in group) for group in groups)
    assert x['all_evidence_at_3'] == expected
for summary in first['summary']:
    subset = [x for x in first['details'] if x['scope_filter'] == summary['scope_filter']
              and x['noise_count'] == summary['noise_count'] and x['method'] == summary['method']
              and x['required_groups']]
    multi = [x for x in subset if len(x['required_groups']) > 1]
    assert len(subset) == summary['answerable_runs'] == 48
    assert len(multi) == summary['multi_evidence_runs'] == 9
    assert abs(sum(x['all_evidence_at_3'] for x in subset) / len(subset)
               - summary['all_evidence_at_3']) < 1e-12
    assert abs(sum(x['all_evidence_at_3'] for x in multi) / len(multi)
               - summary['multi_evidence_at_3']) < 1e-12
audit = {
    'first_output_sha256': hashlib.sha256(first_path.read_bytes()).hexdigest(),
    'repeat_output_sha256': hashlib.sha256(repeat_path.read_bytes()).hexdigest(),
    'comparison': 'same source, code, model and 972 rows; same top-5 order and computed metrics',
    'max_absolute_score_difference': max_delta,
    'scope_and_independent_metric_audit': 'passed',
    'warning': 'Repeated deterministic runs are not independent natural-world samples; no XIPU AI execution.',
}
(HERE / 'repeat_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(audit, ensure_ascii=False, indent=2))
