"""Check repeated team-document and post-hoc diagnostic results."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


base, repeated = read('real_source_results.json'), read('real_source_repeat.json')
with_id = read('real_source_with_id.json')
diagnostic, diagnostic_repeated = read('r10_diagnostic_results.json'), read('r10_diagnostic_repeat.json')
assert base == repeated
assert diagnostic == diagnostic_repeated
assert base['metadata']['section_count'] == with_id['metadata']['section_count'] == 41
assert base['metadata']['source_count'] == 7 and base['metadata']['case_count'] == 12
assert base['metadata']['include_source_id'] is False and with_id['metadata']['include_source_id'] is True
assert base['metadata']['script_sha256'] == with_id['metadata']['script_sha256']
assert len(base['details']) == len(with_id['details']) == 36
for report in (base, with_id):
    for row in report['details']:
        ids = [x['id'] for x in row['top5'][:3]]
        assert row['all_evidence_at_3'] == all(any(x in ids for x in group) for group in row['required_groups'])
    for summary in report['summary']:
        rows = [r for r in report['details'] if r['method'] == summary['method']]
        assert len(rows) == summary['cases'] == 12
        assert abs(sum(r['all_evidence_at_3'] for r in rows) / 12 - summary['all_evidence_at_3']) < 1e-12
for row in diagnostic['details']:
    assert row['all_evidence_covered'] is (row['condition'] != 'single_query_top6')

names = ['real_source_results.json', 'real_source_repeat.json', 'real_source_with_id.json',
         'r10_diagnostic_results.json', 'r10_diagnostic_repeat.json']
result = {'repeat_equal': True, 'raw_metric_recalculation': 'passed',
          'r10_diagnostic': 'single query missed at least one required section; both reformulated variants found both',
          'sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names},
          'warning': 'Post-hoc single-case reformulation; no claim of generalization or XIPU AI execution.'}
(HERE / 'real_source_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
