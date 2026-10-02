import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('-','_'),ROOT/'scripts'/filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_row(root, variant, conv, row, tail=''):
    path = root/variant/'conversations'/conv/'benchmark'/'stamp'/'locomo'/'witnessrag.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n'+tail)
    return path


def test_panel_uses_official_f1_and_qualifies_ids_by_conversation(tmp_path):
    watch = module('watch-locomo-ablation.py')
    write_row(tmp_path,'full','conv00',{'qid':'same','tipo':'multi-hop','f1':0,'f1_locomo':1})
    write_row(tmp_path,'full','conv01',{'qid':'same','tipo':'single-hop','f1':1,'f1_locomo':0})
    write_row(tmp_path,'no-witness','conv00',{'qid':'same','tipo':'multi-hop','f1_locomo':.4},tail='{"qid":')
    result = watch.snapshot(tmp_path)
    assert result['variants']['full']['overall'] == {'n':2,'f1':50}
    paired = result['variants']['no-witness']['paired_with_full']
    assert paired['n'] == 1 and paired['full']['f1'] == 100 and paired['variant']['f1'] == 40
    text = watch.render(result)
    assert '2/1540' in text and '100.00 (1)' in text


def test_panel_rejects_duplicate_complete_ids_and_invalid_scores(tmp_path):
    watch = module('watch-locomo-ablation.py')
    row = {'qid':'q','f1_locomo':1,'tipo':'temporal'}
    path = write_row(tmp_path,'full','conv00',row)
    write_row(tmp_path,'full','conv00',row)
    with pytest.raises(ValueError,match='Duplicate'):
        watch.records(tmp_path/'full')
    path.write_text(json.dumps({**row,'f1_locomo':100})+'\n')
    with pytest.raises(ValueError,match='official F1'):
        watch.records(tmp_path/'full')


def test_all_five_commands_share_budget_and_scientific_flags(tmp_path):
    queue = module('run-locomo-component-ablation.py')
    args = queue.parser().parse_args(['--output',str(tmp_path/'out'),'--cache',str(tmp_path/'cache')])
    commands = queue.commands(args)
    assert list(commands) == ['full','no-witness','no-time-reference','no-time-model','no-reflection']
    from wrag.pilot import parser, make_plan
    for name, cmd in commands.items():
        plan = make_plan(parser().parse_args(cmd[4:]),args.output/name)
        cfg = plan['settings']
        assert cfg['fact_budget'] == 40 and cfg['concurrency'] == 2
        assert cfg['study_ablation'] == name and not cfg['reflection_replan']
        assert cfg['locomo_conversation'] == 'all' and cfg['qa_max_tokens'] == 128


def test_empty_or_partial_results_are_never_marked_complete(tmp_path):
    queue = module('run-locomo-component-ablation.py')
    assert not queue.validated_complete(tmp_path)
    (tmp_path/'status.json').write_text('{"status":"complete"}')
    assert not queue.validated_complete(tmp_path)
