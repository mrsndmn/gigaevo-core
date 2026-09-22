"""Build a standalone, offline HTML comparison from saved feedback rollouts.

python -m experiments.noregeo_synth.feedback_report --run RUN_DIR --out report.html
No inference or execution of model-written code. Supports original replay-frame
layout and the current persistent-worker layout. Optional --note values are
human observations, shown separately from automatic verification.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from html import escape
import json
from pathlib import Path

LABELS = {"history": "Без изображений", "canvas": "С изображениями"}
STATUS = {
    "completed": "Завершён",
    "execution_failed": "Ошибка исполнения",
    "error": "Ошибка прогона",
    "running": "Не завершён",
}
CHECKS = {
    "answer_ok": "Финальный ответ",
    "code_answer_ok": "Ответ в коде",
    "drawn_ok": "Положение точки ответа",
    "incremental_ok": "Изменения рисунка",
}

CSS = """
:root{color-scheme:light;--ink:#182c3b;--muted:#5e7180;--line:#dce4ea;--paper:#fff;--bg:#f2f5f7;--blue:#1769aa;--green:#126341}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}
main{max-width:1360px;margin:auto;padding:40px 28px 80px}h1{font-size:clamp(28px,4vw,44px);line-height:1.15;max-width:960px;margin:10px 0 18px;letter-spacing:-.035em}h2{font-size:24px;margin:0 0 12px}h3{margin:0 0 12px;font-size:19px}p{margin:8px 0 16px}.eyebrow{font-size:12px;font-weight:750;letter-spacing:.13em;text-transform:uppercase;color:var(--blue)}.muted,small{color:var(--muted)}.lead{max-width:1000px;font-size:18px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:24px;min-width:0;overflow-wrap:anywhere}.history{border-top:4px solid #83929e}.canvas{border-top:4px solid var(--blue)}.answer{font-size:23px;font-weight:700;margin:14px 0}.badge{display:inline-block;border-radius:6px;padding:3px 9px;background:#eaf0f4;font-size:13px;margin:0 5px 5px 0}.good{color:#145d3b;background:#e4f3ea}.bad{color:#973b24;background:#fff0e9}.unknown{color:#61717e;background:#edf1f4}.note{border-left:4px solid #b47813;background:#fff8e8;padding:20px 24px;border-radius:0 12px 12px 0;margin:28px 0}.note p:last-child{margin-bottom:0}.task{margin-top:48px}.task-title{scroll-margin-top:90px}.nav{position:sticky;top:0;z-index:2;display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:12px 0;background:#f2f5f7f5;border-bottom:1px solid var(--line);margin:22px 0}.nav a{padding:7px 12px;text-decoration:none;color:var(--ink);background:white;border:1px solid var(--line);border-radius:8px;font-size:14px}.nav a:hover{border-color:var(--blue);color:var(--blue)}.step{margin:30px 0 44px;scroll-margin-top:95px}.step-heading{display:flex;align-items:baseline;gap:14px;margin-bottom:12px}.step-heading h3{margin:0}.step-heading span{font-size:13px;color:var(--muted)}figure{margin:16px 0}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:white;cursor:zoom-in}figcaption{color:var(--muted);font-size:13px;margin-top:8px}.reasoning{white-space:pre-wrap;line-height:1.65}.label{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:700;margin:16px 0 6px}details{border-top:1px solid var(--line);padding-top:12px;margin-top:16px}summary{cursor:pointer;font-weight:650;color:var(--blue)}pre{background:#142737;color:#e6f0f7;padding:18px;border-radius:10px;overflow:auto;font:13px/1.6 ui-monospace,SFMono-Regular,monospace;tab-size:4}ul{padding-left:20px}li{margin-bottom:7px}.missing{padding:45px 20px;text-align:center;background:#f7f9fa;color:var(--muted);border:1px dashed #bdcbd5;border-radius:8px}.fact{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #edf1f4;padding:8px 0;font-size:14px}.fact:last-child{border:0}.error{white-space:pre-wrap;color:#973b24}.protocol{margin-top:40px}.dialog{max-width:96vw;max-height:96vh;padding:12px;border:0;border-radius:12px}.dialog::backdrop{background:#10222dcc}.dialog img{display:block;max-width:90vw;max-height:82vh;margin:auto}.dialog button{display:block;margin:0 0 8px auto;padding:8px 16px;cursor:pointer}.footer{margin-top:36px;font-size:13px;color:var(--muted)}
@media(max-width:760px){main{padding:24px 14px 45px}.grid{grid-template-columns:1fr}.card{padding:18px}.nav{position:static}.step-heading{display:block}.lead{font-size:16px}.fact{flex-wrap:wrap}}
@media print{body{background:white}.nav,.dialog{display:none}.card{break-inside:avoid}main{padding:0}details{display:block}}
"""
SCRIPT = """
const dialog = document.getElementById('image-dialog');
document.querySelectorAll('figure img').forEach(img => {
  img.tabIndex = 0;
  const show = () => { dialog.querySelector('img').src = img.src; dialog.querySelector('img').alt = img.alt; dialog.showModal(); };
  img.addEventListener('click', show);
  img.addEventListener('keydown', event => { if(event.key === 'Enter') show(); });
});
dialog.querySelector('button').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if(event.target === dialog) dialog.close(); });
"""


def text(value):
    return escape(str(value), quote=True)


def answer(value):
    if value is None:
        return "Не получен"
    if isinstance(value, dict):
        if "point" in value:
            return answer(value["point"])
        if "x" in value and "y" in value:
            return answer([value["x"], value["y"]])
        return "; ".join(f"{text(k)}: {answer(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(answer(v) for v in value) + ")"
    if isinstance(value, float):
        return text(f"{value:.8g}")
    return text(value)


def badge(value):
    if value is True:
        return '<span class="badge good">Пройдено</span>'
    if value is False:
        return '<span class="badge bad">Не пройдено</span>'
    return '<span class="badge unknown">Не проверено</span>'


def frame_path(directory, index):
    for path in (
        directory / "frames" / f"step_{index:02d}.png",
        directory / f"turn_{index:02d}" / f"step_{index:02d}.png",
    ):
        if path.is_file():
            return path
    return None


def figure(path, caption):
    if path is None:
        return '<div class="missing">Кадр не сохранён</div>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{text(caption)}" '
        f'loading="lazy"><figcaption>{text(caption)} · нажмите для увеличения</figcaption></figure>'
    )


def overview(record):
    arm = record["arm"]
    checks = record.get("checks") or {}
    state = record.get("status", "unknown")
    parts = [
        f'<article class="card {text(arm)}"><h3>{text(LABELS.get(arm, arm))}</h3>',
        f'<span class="badge">{text(STATUS.get(state, state))}</span>',
        f'<div class="answer">{answer(record.get("final_answer"))}</div>',
    ]
    for key, label in CHECKS.items():
        parts.append(
            f'<div class="fact"><span>{label}</span>{badge(checks.get(key))}</div>'
        )
    accepted = (
        record.get("accepted") if state in ("completed", "execution_failed") else None
    )
    parts.append(
        f'<div class="fact"><strong>Все проверки трейса</strong>{badge(accepted)}</div>'
    )
    if record.get("error"):
        parts.append(f'<p class="error">{text(record["error"])}</p>')
    problems = checks.get("problems") or []
    if problems:
        parts.append(
            '<ul class="error">'
            + "".join(f"<li>{text(p)}</li>" for p in problems)
            + "</ul>"
        )
    parts.append("<details><summary>Сведения о прогоне</summary>")
    cfg = record.get("config", {}).get("generation", {})
    parts.append(f"<p>Модель: {text(cfg.get('model', 'не записана'))}</p>")
    parts.append(
        f"<p>Шагов: {len(record.get('steps', []))}. Записанных вызовов: {len(record.get('calls', []))}.</p>"
    )
    if record.get("generation_source"):
        parts.append(
            "<p>Повторное исполнение сохранённых ответов, не новая генерация. Время этого исполнения не отражает задержку модели.</p>"
        )
    elif record.get("wall_time_s") is not None:
        parts.append(f"<p>Время прогона: {text(record['wall_time_s'])} с.</p>")
    if record.get("shared_first_step"):
        parts.append(
            "<p>Первое действие взято из другого режима и не генерировалось повторно. Прямое сравнение времени и числа вызовов некорректно.</p>"
        )
    parts.append("</details></article>")
    return "".join(parts)


def step_card(record, directory, index):
    arm = record["arm"]
    label = LABELS.get(arm, arm)
    steps = record.get("steps", [])
    if index > len(steps):
        return f'<article class="card {text(arm)}"><h3>{text(label)}</h3><div class="missing">Шаг отсутствует · {text(STATUS.get(record.get("status"), "нет данных"))}</div></article>'
    step = steps[index - 1]
    mode = record.get("config", {}).get("observation")
    observation = (
        "До этого действия: условие, история кода и PNG предыдущих шагов."
        if mode == "image" and index > 1
        else "До этого действия: условие и история кода; промежуточные PNG не передаются."
        if index > 1
        else "Первое действие: промежуточной обратной связи ещё нет."
    )
    caption = (
        "Результат исполнения — обратная связь для следующего ответа"
        if mode == "image"
        else "Результат исполнения — модель этот PNG не получает"
    )
    parts = [
        f'<article class="card {text(arm)}"><div class="eyebrow">{text(label)}</div>',
        f"<h3>{text(step.get('title', 'Без названия'))}</h3>",
        f'<p class="muted"><small>{observation}</small></p>',
        '<div class="label">Пояснение модели · оригинал</div>',
        f'<p class="reasoning">{text(step.get("reasoning", "Нет пояснения"))}</p>',
        figure(frame_path(directory, index), caption),
        "<details><summary>Код этого действия</summary>",
        f"<pre><code>{text(step.get('code', ''))}</code></pre></details>",
    ]
    if mode == "image" and index > 1:
        parts.append(
            "<details><summary>Предыдущий кадр — перед этим действием</summary>"
        )
        parts.append(figure(frame_path(directory, index - 1), "Кадр предыдущего шага"))
        parts.append("</details>")
    parts.append("</article>")
    return "".join(parts)


def build_report(run: Path, out: Path, notes=()):
    run, out = Path(run).resolve(), Path(out).resolve()
    groups = defaultdict(dict)
    for path in sorted(run.glob("*/*/record.json")):
        record = json.loads(path.read_text())
        task_id, arm = record["task_id"], record["arm"]
        if arm not in LABELS:
            raise ValueError(f"unsupported feedback arm: {arm}")
        if arm in groups[task_id]:
            raise ValueError(f"duplicate record: {task_id}/{arm}")
        groups[task_id][arm] = (record, path.parent)
    if not groups:
        raise ValueError(f"no feedback records in {run}")
    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>NoReGeo — анализ промежуточной обратной связи</title>",
        f"<style>{CSS}</style></head><body><main>",
        '<div class="eyebrow">NoReGeo / анализ эксперимента</div>',
        "<h1>Помогает ли видеть результат собственных построений?</h1>",
        '<p class="lead">Слева — история кода и статус исполнения. Справа — те же источники плюс рисунки после каждого действия. Сравнивайте пояснения, выполненный код и то, что появилось на канвасе.</p>',
        f'<p class="muted">Задач: {len(groups)} · источник: {text(run.name)} · все изображения встроены в этот файл.</p>',
    ]
    if notes:
        parts.append(
            '<aside class="note"><h3>Наблюдения при просмотре</h3><small>Ручной анализ, отдельно от автоматических проверок.</small>'
        )
        parts.extend(f"<p>{text(note)}</p>" for note in notes)
        parts.append("</aside>")
    for task_number, (task_id, records) in enumerate(sorted(groups.items()), 1):
        exemplar = next(iter(records.values()))[0]
        maximum = max(len(r.get("steps", [])) for r, _ in records.values())
        prefix = f"task-{task_number}"
        parts += [
            f'<section class="task" id="{prefix}"><h2 class="task-title">{text(task_id)}</h2>',
            f'<p class="lead">{text(exemplar.get("question", "Условие не записано"))}</p>',
            f"<p>Эталон: <strong>{answer(exemplar.get('golden_answer'))}</strong>. Используется только при проверке, не передаётся модели.</p>",
            '<div class="grid">',
        ]
        for arm in LABELS:
            parts.append(
                overview(records[arm][0])
                if arm in records
                else f'<article class="card"><h3>{LABELS[arm]}</h3><p>Прогон отсутствует. Это не успешный результат.</p></article>'
            )
        parts.append("</div>")
        first_steps = [
            r.get("steps", [])[0] for r, _ in records.values() if r.get("steps")
        ]
        shared = (
            len(first_steps) == 2
            and first_steps[0] == first_steps[1]
            and any(r.get("shared_first_step") for r, _ in records.values())
        )
        if shared:
            parts.append(
                '<p class="muted">Первое действие общее: оба режима продолжают один и тот же код.</p>'
            )
        parts.append('<nav class="nav" aria-label="Переход к шагу">')
        parts.extend(
            f'<a href="#{prefix}-step-{i}">Шаг {i}</a>' for i in range(1, maximum + 1)
        )
        parts.append(f'<a href="#{prefix}">Результат</a></nav>')
        for i in range(1, maximum + 1):
            parts.append(
                f'<section class="step" id="{prefix}-step-{i}"><div class="step-heading"><h3>Шаг {i}</h3><span>Сопоставление по номеру действия; содержание может различаться</span></div><div class="grid">'
            )
            for arm in LABELS:
                parts.append(
                    step_card(*records[arm], i)
                    if arm in records
                    else '<article class="card"><div class="missing">Нет прогона для сравнения</div></article>'
                )
            parts.append("</div></section>")
        parts.append("</section>")
    parts += [
        '<aside class="note protocol"><h3>Как читать результаты</h3>',
        "<p>Зелёные проверки означают, что выполнены критерии проверяющего кода. Они не подтверждают каждое предложение в пояснении. Для задач на точку проверяется её положение; для других типов эта проверка может быть неприменима.</p>",
        "<p>Исходные координаты остаются в условии у обоих режимов. Один пилот не устанавливает преимущество изображений. В визуальном режиме рисунок последнего действия также передаётся перед отдельным финальным ответом, если прогон доходит до этого запроса.</p></aside>",
        '<p class="footer">Отчёт построен из сохранённых артефактов. Открытие файла не вызывает модель и не исполняет код шагов. Работает без сети; код раскрывается по нажатию.</p>',
        '</main><dialog class="dialog" id="image-dialog"><button type="button">Закрыть · Esc</button><img alt="Увеличенный кадр"></dialog>',
        f"<script>{SCRIPT}</script></body></html>",
    ]
    if out.suffix.lower() != ".html":
        raise ValueError("output must end with .html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--note", action="append", default=[], help="human analysis note; repeatable"
    )
    args = parser.parse_args(argv)
    print(build_report(args.run, args.out, args.note))


if __name__ == "__main__":
    main()
