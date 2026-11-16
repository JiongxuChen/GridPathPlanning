"""Audit the isolated ablation against the complete structured verifier."""
import argparse
import ast
import json
from pathlib import Path


MODEL = 'src/gridpath/planning/model.py'
METHOD = 'improvement_verify_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cases(root, panel):
    directory = root / f'results/{METHOD}/t5_base/dev/{panel}'
    return {p.stem: read(p) for p in directory.glob('sample_*.json')}


def metrics(root):
    report = read(root / f'results/{METHOD}/summary_dev.json')['models']['t5_base']
    v = report['fixed_verification']['self']
    panels = {}
    for key in ('generated', 'controlled_repair'):
        panel = report['panels'][key]
        panels[key] = {
            'direct_success': panel['direct']['success_count'],
            'self_repairs': {
                route: {
                    k: panel['feedback_routes']['self']['repairs'][route][k]
                    for k in ('final_success_count', 'valid_destruction_count',
                              'repair_success_count', 'logical_repair_model_calls')
                } for route in ('prompt', 'prefix')
            },
        }
    return {
        'fixed_v80': {
            'macro_f1': v['macro_f1'],
            'classification_accuracy': v['classification_accuracy'],
            'localization_count_of_40': v['first_error_localization_count'],
            'joint_count_of_40': v['joint_diagnosis_count'],
            'confusion': v['confusion'],
            'collision_joint_count_of_10': v['by_reason']['obstacle_collision']['joint_diagnosis_count'],
        },
        'natural_collisions': report['real_collision_verification']['all_collisions'],
        'panels': panels,
    }


def source_contract(reference, ablation):
    before = ast.parse((reference / MODEL).read_text(encoding='utf-8'))
    after = ast.parse((ablation / MODEL).read_text(encoding='utf-8'))
    def unchanged_nodes(tree):
        return [ast.dump(node) for node in tree.body
                if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))
                and getattr(node, 'name', None) != 'path_loss']
    assert unchanged_nodes(before) == unchanged_nodes(after), 'Code outside path_loss changed'

    first = read(reference / f'checkpoints/{METHOD}/t5_base/verify/run.json')
    second = read(ablation / f'checkpoints/{METHOD}/t5_base/verify/run.json')
    old_hash = first['sources'][MODEL]
    new_hash = second['sources'][MODEL]
    assert old_hash != new_hash
    for binding in (first, second):
        binding.pop('fingerprint')
        binding['sources'].pop(MODEL)
    assert first == second, 'Training binding changed beyond the loss-module hash'
    return {'only_binding_difference': MODEL, 'reference_hash': old_hash, 'ablation_hash': new_hash}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--ablation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert read(args.reference / f'results/{METHOD}/review_dev.json')['status'] == 'passed'
    assert read(args.ablation / f'results/{METHOD}/review_dev.json')['status'] == 'passed'
    assert not (args.ablation / f'manifests/test_usage_{METHOD}.json').exists()
    assert not (args.ablation / f'results/{METHOD}/summary_test_id.json').exists()

    paired = {root: cases(root, 'cases') for root in (args.reference, args.ablation)}
    fixed = {root: cases(root, 'verification_cases') for root in (args.reference, args.ablation)}
    assert set(paired[args.reference]) == set(paired[args.ablation])
    assert len(paired[args.reference]) == 120
    assert set(fixed[args.reference]) == set(fixed[args.ablation])
    assert len(fixed[args.reference]) == 80
    for sid, old in paired[args.reference].items():
        new = paired[args.ablation][sid]
        assert old['initial'] == new['initial']
        assert old['executor_feedback'] == new['executor_feedback']
        for route in ('executor_prompt', 'executor_prefix'):
            assert old['outcomes'][route] == new['outcomes'][route]
    for sid, old in fixed[args.reference].items():
        new = fixed[args.ablation][sid]
        assert old['executor_feedback'] == new['executor_feedback']

    old_entries = read(args.reference / 'manifests/gpu_budget_total.json')['entries']
    new_entries = read(args.ablation / 'manifests/gpu_budget_total.json')['entries']
    assert new_entries[:len(old_entries)] == old_entries
    extra = new_entries[len(old_entries):]
    assert len(extra) == 3 and all(e['status'] == 'complete' for e in extra)
    result = {
        'contract': source_contract(args.reference, args.ablation),
        'paired_case_count': 120,
        'paired_fixed_v_count': 80,
        'same_G_truth_and_executor_routes': True,
        'ablation_test_sets_not_evaluated': True,
        'additional_gpu_hours': sum(e['wall_seconds'] for e in extra) / 3600,
        'reference': metrics(args.reference),
        'ablation': metrics(args.ablation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
