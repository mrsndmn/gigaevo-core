"""Offline report correctness for model text, failures, and both frame layouts."""

import base64
import json

import pytest

from experiments.noregeo_synth.feedback_report import build_report


def save(root, arm="canvas", status="completed", layout="frames"):
    directory = root / "sample" / arm
    directory.mkdir(parents=True)
    record = {
        "task_id": "midpoint:1",
        "arm": arm,
        "question": "Find the midpoint.",
        "golden_answer": [1, 2],
        "final_answer": {"x": 1, "y": 2},
        "status": status,
        "accepted": status == "completed",
        "error": None,
        "config": {"observation": "image" if arm == "canvas" else "status"},
        "steps": [
            {
                "title": "<script>alert(1)</script>",
                "reasoning": "A < B & C",
                "code": 'print("</script><script>alert(2)</script>")',
            }
        ],
        "checks": {
            "answer_ok": True,
            "code_answer_ok": True,
            "drawn_ok": None,
            "incremental_ok": False,
            "problems": ["<bad>"],
        },
    }
    (directory / "record.json").write_text(json.dumps(record))
    frames = directory / layout
    frames.mkdir()
    (frames / "step_01.png").write_bytes(b"PNG-content")
    return directory


@pytest.mark.parametrize("layout", ["frames", "turn_01"])
def test_embedded_frames_and_escaped_model_output(tmp_path, layout):
    root = tmp_path / "run"
    save(root, layout=layout)
    out = build_report(root, tmp_path / "out.html", ["<human note>"])
    html = out.read_text()
    assert "data:image/png;base64," + base64.b64encode(b"PNG-content").decode() in html
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;human note&gt;" in html
    assert "Прогон отсутствует" in html
    assert 'href="http' not in html and 'src="http' not in html
    assert "Не проверено" in html
    assert "Не пройдено" in html


def test_failed_and_missing_artifacts_remain_visible(tmp_path):
    root = tmp_path / "run"
    directory = save(root, status="error")
    path = directory / "record.json"
    r = json.loads(path.read_text())
    r.update(checks=None, final_answer=None, error="RuntimeError: <endpoint failed>")
    path.write_text(json.dumps(r))
    (directory / "frames" / "step_01.png").unlink()
    html = build_report(root, tmp_path / "out.html").read_text()
    assert "Ошибка прогона" in html and "Не получен" in html
    assert "Кадр не сохранён" in html
    assert "&lt;endpoint failed&gt;" in html
    assert '<span class="badge good">' not in html


def test_replay_does_not_present_execution_time_as_inference_time(tmp_path):
    root = tmp_path / "run"
    directory = save(root)
    path = directory / "record.json"
    record = json.loads(path.read_text())
    record.update(generation_source="replay", wall_time_s=0.2)
    path.write_text(json.dumps(record))
    html = build_report(root, tmp_path / "out.html").read_text()
    assert "не новая генерация" in html
    assert "Время прогона: 0.2" not in html


def test_empty_run_fails_without_replacing_report(tmp_path):
    out = tmp_path / "out.html"
    out.write_text("existing")
    with pytest.raises(ValueError, match="no feedback records"):
        build_report(tmp_path / "missing", out)
    assert out.read_text() == "existing"
