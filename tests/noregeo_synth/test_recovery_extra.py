from experiments.noregeo_synth import recovery_extra as ev
from experiments.noregeo_synth import run_recovery_eval as launcher


def task(stem, qid):
    return {"stem": stem, "row": {"qid": qid}}


def test_selection_excludes_old_and_prioritizes_new_stems():
    tasks = [task(s, i) for s in ["a", "b", "c"] for i in [1, 2, 3]]
    old = [task("a", 1)]
    chosen = ev.select_tasks(tasks, old, 3)
    assert chosen == [task("b", 1), task("c", 1), task("a", 2)]
    assert not ({ev.task_id(t) for t in old} & {ev.task_id(t) for t in chosen})


def record(arm):
    first = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    last = {"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}
    return {
        "task_id": "a:1",
        "arm": arm,
        "status": "completed",
        "correct": True,
        "events": [{"raw": first}],
        "first": first,
        "final_raw": last,
    }


def test_token_dedup_including_partial_pairs():
    single = ev.group_metrics([record("history")], 1)
    paired = ev.group_metrics([record("history"), record("canvas")], 1)
    assert single["actual_tokens"] == 45
    assert paired["actual_tokens"] == 75
    assert paired["arms"]["canvas"]["prompt_tokens"] == 30
    assert paired["paired"]["both"] == 1


def test_launcher_payload_selects_correct_module_and_description():
    old = launcher.payload("commit", "/tmp/out", "/tmp/env")
    new = launcher.payload(
        "commit",
        "/tmp/out",
        "/tmp/env",
        ev.SPEC,
        "experiments.noregeo_synth.recovery_extra",
    )
    assert "recovery_extra" not in old["job"]["script"]
    assert "recovery_extra" in new["job"]["script"]
    assert new["job"]["description"].endswith(ev.SPEC.name)
    assert old["job"]["description"] != new["job"]["description"]
