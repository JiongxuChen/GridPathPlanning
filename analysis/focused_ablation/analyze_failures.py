"""Recount paired verifier/repair failures from immutable case artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path


METHODS = ('v2', 'improvement_verify_v1')
SPLITS = ('dev', 'test_id', 'test_ood')
ROUTES = ('self_prompt', 'self_prefix', 'executor_prompt', 'executor_prefix')
DELTAS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(root, split, panel):
    path = root / 'data/v2/shared' / f'{split}_{panel}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def cases(root, method, split, panel):
    directory = root / 'results' / method / 't5_base' / split / panel
    return {path.stem: read(path) for path in directory.glob('sample_*.json')}


def unchecked_endpoint(row, candidate):
    r, c = row['start']
    for action in candidate.split():
        dr, dc = DELTAS.get(action, (0, 0))
        r += dr
        c += dc
    return [r, c] == row['goal']


def verifier_counts(paired, ids):
    answer = {'count': len(ids)}
    for method in METHODS:
        found = [paired[method][sid] for sid in ids]
        answer[method] = {
            'detected_invalid': sum(c['self_verification']['effective_label'].startswith('invalid') for c in found),
            'joint_correct': sum(c['self_verification']['effective_label'] == c['executor_feedback'] for c in found),
            'predicted_valid_ids': [c['sample_id'] for c in found
                                    if c['self_verification']['effective_label'] == 'valid'],
        }
    return answer


def analyze(root, split):
    g = rows(root, split, 'generate')
    v = rows(root, split, 'verify')
    paired = {m: cases(root, m, split, 'cases') for m in METHODS}
    fixed = {m: cases(root, m, split, 'verification_cases') for m in METHODS}
    assert all(set(paired[m]) >= {r['sample_id'] for r in g} for m in METHODS)
    assert all(set(fixed[m]) == {r['sample_id'] for r in v} for m in METHODS)

    for row in g:
        sid = row['sample_id']
        base, improved = (paired[m][sid] for m in METHODS)
        assert base['initial'] == improved['initial']
        assert base['executor_feedback'] == improved['executor_feedback']
        for route in ('executor_prompt', 'executor_prefix'):
            assert base['outcomes'][route] == improved['outcomes'][route]

    bad = [r for r in g if not paired[METHODS[0]][r['sample_id']]['initial_success']]
    by_reason = {}
    for reason in sorted({paired[METHODS[0]][r['sample_id']]['initial_scoring']['execution']['error_reason']
                          for r in bad}):
        selected = [r['sample_id'] for r in bad
                    if paired[METHODS[0]][r['sample_id']]['initial_scoring']['execution']['error_reason'] == reason]
        by_reason[reason] = verifier_counts(paired, selected)

    collision_rows = [r for r in bad
                      if paired[METHODS[0]][r['sample_id']]['initial_scoring']['execution']['error_reason']
                      == 'obstacle_collision']
    collision_ids = [r['sample_id'] for r in collision_rows]
    at_goal = [r['sample_id'] for r in collision_rows
               if unchecked_endpoint(r, paired[METHODS[0]][r['sample_id']]['initial']['text'])]

    repair = {}
    for method in METHODS:
        failures = [paired[method][r['sample_id']] for r in bad]
        repair[method] = {}
        for route in ROUTES:
            outcomes = [c['outcomes'][route] for c in failures]
            called = [(c, o) for c, o in zip(failures, outcomes) if o['detail']['model_calls']]
            repair[method][route] = {
                'initial_failures': len(failures),
                'model_calls': sum(o['detail']['model_calls'] for o in outcomes),
                'repaired': sum(o['success'] for o in outcomes),
                'still_failed_after_call': sum(not o['success'] for _, o in called),
                'repeated_original_after_call': sum(o['prediction']['text'] == c['initial']['text']
                                                    for c, o in called),
                'repaired_ids': [c['sample_id'] for c, o in zip(failures, outcomes) if o['success']],
            }

    fixed_by_reason = {}
    for reason in sorted({r['error_reason'] for r in v if not r['is_valid']}):
        ids = [r['sample_id'] for r in v if r['error_reason'] == reason]
        fixed_by_reason[reason] = verifier_counts(fixed, ids)
    return {
        'split': split,
        'independent_g': len(g), 'independent_v': len(v),
        'natural_initial_failures': len(bad),
        'natural_by_reason': by_reason,
        'natural_collisions': verifier_counts(paired, collision_ids),
        'natural_collisions_unchecked_endpoint_at_goal': verifier_counts(paired, at_goal),
        'fixed_invalid_by_reason': fixed_by_reason,
        'natural_repair': repair,
        'paired_G_and_executor_outcomes_identical': True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {split: analyze(args.root, split) for split in SPLITS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
