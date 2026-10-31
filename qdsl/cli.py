"""CLI Kiln: читаемый вывод, цветные статусы, подсказки при ошибках, тихий режим для библиотек."""
import argparse
import codecs
import math
import os
import re
import subprocess
import sys
import threading
import time

from .codegen import generate
from .lexer import QdslSyntaxError
from .parser import parse
from .presets import HARDWARE
from .schema import validate

if os.name == "nt":
    os.system("")  # включает ANSI-цвета в консоли Windows
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except AttributeError:
    pass

USE_COLOR = sys.stdout.isatty()
STYLE = {"ok": "32", "warn": "33", "err": "31", "info": "36", "dim": "2", "bold": "1"}
SYMBOL = {"ok": "✔", "warn": "⚠", "err": "✖", "run": "▶", "stop": "■"}

# Шум библиотек, который не нужен пользователю; при --raw не отключается
QUIET_ENV = {
    "PYTHONWARNINGS": "ignore",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}

# Подсказки по типичным ошибкам: (фрагмент из вывода, что делать)
HINTS = [
    ("winerror 10013", "Нет доступа к huggingface.co (VPN или фаервол). Укажите локальную папку модели."),
    ("couldn't connect to 'https://huggingface.co'", "Нет доступа к huggingface.co. Укажите локальную папку модели."),
    ("out of memory", "Не хватает видеопамяти. Уменьшите context или batch, а accum увеличьте, чтобы сохранить батч."),
    ("can't find a valid checkpoint", "Чекпоинт повреждён. Удалите неполную папку checkpoint-N и запустите снова."),
    ("loss = nan", "Loss стал NaN. Уменьшите lr, например до 1e-4."),
    ("loss = inf", "Loss стал бесконечным. Уменьшите lr."),
    ("filenotfounderror", "Файл или папка не найдены. Проверьте пути к модели и датасету в .qdsl."),
    ("modulenotfounderror", "Не установлена библиотека. Установите: pip install transformers trl peft bitsandbytes accelerate datasets"),
    ("not implemented for 'bfloat16'", "Смешение bf16 и fp16. Проверьте, что target соответствует вашему железу."),
]


def paint(style, text):
    if not USE_COLOR:
        return text
    return f"\033[{STYLE[style]}m{text}\033[0m"


def say(text=""):
    print(text, flush=True)


def fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m} мин {s} с" if m else f"{s} с"


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        return None, [(1, f"не удалось открыть файл: {e.strerror}")], []
    try:
        root = parse(src)
    except QdslSyntaxError as e:
        return None, [(e.line, e.msg)], []
    return validate(root, HARDWARE)


def analyze_tokens(data_path, data_format, model_name, context_limit):
    """Быстрый анализ длины токенов в датасете без зависаний сети и тяжелых импортов."""
    if not os.path.isfile(data_path):
        return None
    texts = []
    try:
        import json
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if data_format == "alpaca":
                        u = item.get("instruction", "") + ("\n" + item["input"] if item.get("input") else "")
                        texts.append(f"{u} {item.get('output', '')}")
                    else:
                        msgs = item.get("messages", [])
                        if isinstance(msgs, list):
                            texts.append(" ".join(m.get("content", "") for m in msgs if isinstance(m, dict)))
                        else:
                            texts.append(str(item))
                except Exception:
                    continue
    except Exception:
        return None

    if not texts:
        return None

    tok = None
    tok_json = os.path.join(model_name, "tokenizer.json") if os.path.isdir(model_name) else None
    if tok_json and os.path.isfile(tok_json):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tok_json)
        except Exception:
            tok = None

    lengths = []
    heuristic = False
    if tok:
        for t in texts:
            try:
                lengths.append(len(tok.encode(t, add_special_tokens=True).ids))
            except Exception:
                lengths.append(max(1, len(t) // 4))
    else:
        heuristic = True
        lengths = [max(1, int(len(t) / 3.6)) for t in texts]

    if not lengths:
        return None

    min_l = min(lengths)
    max_l = max(lengths)
    avg_l = int(sum(lengths) / len(lengths))
    fits = sum(1 for l in lengths if l <= context_limit)
    total = len(lengths)
    pct = int(100 * fits / total) if total else 100
    dropped = total - fits
    note = " (оценка)" if heuristic else ""
    res = f"min {min_l} · avg {avg_l} · max {max_l}{note} · {pct}% в context {context_limit}"
    if dropped > 0:
        res += f" ({dropped} отсеются)"
    return res


def count_examples(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def uses_bf16(cfg):
    return bool(cfg["profile"]["bf16"]) and cfg["model"]["precision"] != "fp16"


def total_steps(cfg, n_examples):
    if not n_examples:
        return None
    t = cfg["train"]
    per_step = t["batch"] * t["accum"] * cfg["gpus"]
    return math.ceil(math.ceil(n_examples / per_step) * t["epochs"])


def print_issues(errs, warns):
    for line, msg in errs:
        say(f"  {paint('err', SYMBOL['err'])} строка {line}: {msg}")
    for line, msg in warns:
        say(f"  {paint('warn', SYMBOL['warn'])} строка {line}: {msg}")


def print_header(cfg, n_examples, title):
    m, t, ev, ex = cfg["model"], cfg["train"], cfg.get("eval"), cfg.get("export")
    steps = total_steps(cfg, n_examples)
    tok_stats = analyze_tokens(cfg["data"]["path"], cfg["data"]["format"], m["name"], m["context"])
    rows = [
        ("модель", f"{m['name']} · {m['precision']} · context {m['context']}"),
        ("данные", f"{cfg['data']['path']} · {n_examples if n_examples else '?'} примеров · {cfg['data']['train_on']}"),
    ]
    if tok_stats:
        rows.append(("токены", tok_stats))
    rows.extend([
        ("железо", f"{cfg['target']} · {cfg['gpus']} GPU · {'bf16' if uses_bf16(cfg) else 'fp16'} · backend {cfg['backend']}"),
        ("батч", f"{t['batch']} × accum {t['accum']} × GPU {cfg['gpus']} = {cfg['effective_batch']}"),
        ("обучение", f"эпох {t['epochs']:g} · lr {t['lr']:g}" + (f" · шагов {steps}" if steps else "")),
    ])
    if ev:
        rows.append(("валидация", f"{ev['path']} · каждые {ev['every']} шагов · остановка после {ev['patience']} без улучшения"))
    if ex:
        rows.append(("экспорт", f"{ex['format']} ({ex['quantization']})"))
    rows.append(("выход", t["output"]))
    say(paint("bold", title))
    for label, value in rows:
        say(f"  {paint('dim', label.ljust(10))}{value}")
    say()


def latest_checkpoint(out_dir):
    if not os.path.isdir(out_dir):
        return None
    steps = []
    for name in os.listdir(out_dir):
        if name.startswith("checkpoint-") and name.split("-", 1)[1].isdigit():
            steps.append((int(name.split("-", 1)[1]), name))
    return max(steps)[1] if steps else None


def run_streamed(cmd, env):
    """Запускает скрипт обучения и печатает его вывод в реальном времени. Возвращает (код, stdout, stderr)."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    captured = {"out": [], "err": []}

    def pump(stream, key, target):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            raw_bytes = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not raw_bytes:
                break
            text = decoder.decode(raw_bytes)
            captured[key].append(text)
            target.write(text)
            target.flush()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, "out", sys.stdout), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, "err", sys.stderr), daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        say()
        say(paint("warn", "Прервано клавишами Ctrl+C. Для безопасной остановки создайте файл STOP в папке вывода."))
        rc = proc.wait()
    for t in threads:
        t.join()
    return rc, "".join(captured["out"]), "".join(captured["err"])


def explain_failure(rc, text):
    lines = [line for line in text.splitlines() if line.strip()]
    reason = next((l for l in reversed(lines) if re.search(r"(Error|Exception):", l)), lines[-1] if lines else "")
    say()
    say(paint("err", f"{SYMBOL['err']} Обучение завершилось с ошибкой (код {rc})"))
    if reason:
        say(f"  причина: {reason.strip()}")
    lowered = text.lower()
    for fragment, hint in HINTS:
        if fragment in lowered:
            say(f"  подсказка: {hint}")
            break
    say(paint("dim", "  полный вывод выше; для диагностики запустите с --raw"))


def cmd_check(path):
    cfg, errs, warns = load(path)
    say(paint("bold", f"Проверка {path}"))
    print_issues(errs, warns)
    if errs or cfg is None:
        say(paint("err", f"  {SYMBOL['err']} ошибок: {len(errs)}"))
        return 1
    tail = f", предупреждений: {len(warns)}" if warns else ""
    say(paint("ok", f"  {SYMBOL['ok']} Конфиг корректен{tail}"))
    return 0


def cmd_plan(path):
    cfg, errs, warns = load(path)
    if errs or cfg is None:
        print_issues(errs, warns)
        return 1
    n = count_examples(cfg["data"]["path"])
    print_header(cfg, n, paint("bold", "План обучения"))
    print_issues([], warns)
    return 0


def cmd_train(path, resume_only=False, raw=False):
    cfg, errs, warns = load(path)
    if errs or cfg is None:
        print_issues(errs, warns)
        say(paint("err", f"{SYMBOL['err']} Конфиг не прошёл проверку, обучение не запущено."))
        return 1

    out = cfg["train"]["output"]
    if resume_only:
        if not latest_checkpoint(out):
            say(paint("err", f"{SYMBOL['err']} В '{out}' нет чекпоинтов для продолжения."))
            return 1
        cfg["train"]["resume"] = "auto"

    os.makedirs(out, exist_ok=True)
    script = os.path.join(out, "generated_train.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(generate(cfg, os.path.basename(path)))

    n = count_examples(cfg["data"]["path"])
    print_issues([], warns)
    print_header(cfg, n, paint("bold", f"{SYMBOL['run']} Kiln · обучение"))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if not raw:
        env.update(QUIET_ENV)

    n_gpu = cfg["gpus"] if cfg["strategy"] == "ddp" else 1
    if n_gpu == 1:
        cmd = [sys.executable, script]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n_gpu}", script]

    start = time.monotonic()
    rc, out_text, err_text = run_streamed(cmd, env)
    elapsed = fmt_duration(time.monotonic() - start)
    text = out_text + err_text

    say()
    if rc == 0 and "получен STOP" in text:
        last = latest_checkpoint(out)
        say(paint("warn", f"{SYMBOL['stop']} Остановлено по STOP за {elapsed}"
                          + (f", сохранён {last}" if last else "")))
        say(paint("dim", f"  продолжить: python -m qdsl resume {path}"))
        return 0
    if rc == 0:
        done = re.search(r"\[kiln\] готово: (\S+)", text)
        where = done.group(1) if done else out
        say(paint("ok", f"{SYMBOL['ok']} Готово: {where} · за {elapsed}"))
        return 0
    explain_failure(rc, text)
    return rc


def cmd_export(path, format_override=None, quant_override=None):
    cfg, errs, warns = load(path)
    if errs or cfg is None:
        print_issues(errs, warns)
        return 1

    out = cfg["train"]["output"]
    final = os.path.join(out, "final")
    if not os.path.exists(final):
        say(paint("err", f"{SYMBOL['err']} Папка финальной модели не найдена: '{final}'. Сначала запустите обучение."))
        return 1

    ex = cfg.get("export") or {}
    fmt = format_override or ex.get("format", "gguf")
    quant = quant_override or ex.get("quantization", "q4_k_m")
    backend = cfg["backend"]
    export_dir = os.path.join(out, "export")
    os.makedirs(export_dir, exist_ok=True)

    say(paint("bold", f"{SYMBOL['run']} Экспорт модели: {fmt} ({quant}) → {export_dir}"))

    script_content = f"""import os
from transformers import AutoTokenizer

OUTPUT = {out!r}
FINAL = {final!r}
EXPORT_DIR = {export_dir!r}
FORMAT = {fmt!r}
QUANT = {quant!r}
BACKEND = {backend!r}

if BACKEND == 'unsloth':
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(FINAL, max_seq_length={cfg['model']['context']})
    if FORMAT == 'gguf':
        print(f'[kiln] экспорт в GGUF ({quant})...')
        model.save_pretrained_gguf(EXPORT_DIR, tokenizer, quantization_method=QUANT)
    elif FORMAT == 'merged_16bit':
        print('[kiln] слияние LoRA в 16bit...')
        model.save_pretrained_merged(EXPORT_DIR, tokenizer, save_method='merged_16bit')
    elif FORMAT == 'merged_4bit':
        print('[kiln] слияние LoRA в 4bit...')
        model.save_pretrained_merged(EXPORT_DIR, tokenizer, save_method='merged_4bit')
else:
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(FINAL)
    base = AutoModelForCausalLM.from_pretrained({cfg['model']['name']!r}, torch_dtype='auto')
    model = PeftModel.from_pretrained(base, FINAL)
    print('[kiln] слияние LoRA весов...')
    merged = model.merge_and_unload()
    merged.save_pretrained(EXPORT_DIR)
    tokenizer.save_pretrained(EXPORT_DIR)

print('[kiln] экспорт успешно завершён:', EXPORT_DIR)
"""
    export_script = os.path.join(out, "run_export.py")
    with open(export_script, "w", encoding="utf-8") as f:
        f.write(script_content)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    rc, out_text, err_text = run_streamed([sys.executable, export_script], env)
    if rc == 0:
        say(paint("ok", f"{SYMBOL['ok']} Экспорт готов: {export_dir}"))
        return 0
    explain_failure(rc, out_text + err_text)
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(prog="qdsl", description="Kiln: дообучение LLM на Unsloth и transformers")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("check", "проверить конфиг"),
                            ("plan", "показать план обучения"),
                            ("train", "запустить обучение"),
                            ("resume", "продолжить с последнего полного чекпоинта")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("file", help="путь к .qdsl файлу")
        if name in ("train", "resume"):
            sp.add_argument("--raw", action="store_true", help="показать весь вывод библиотек")

    sp_ex = sub.add_parser("export", help="экспортировать обученную модель в GGUF или 16bit")
    sp_ex.add_argument("file", help="путь к .qdsl файлу")
    sp_ex.add_argument("--format", choices=["gguf", "merged_16bit", "merged_4bit"], default=None, help="формат экспорта")
    sp_ex.add_argument("--quant", default=None, help="тип квантования для gguf (q4_k_m, q8_0, f16)")

    a = p.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a.file)
    if a.cmd == "plan":
        return cmd_plan(a.file)
    if a.cmd == "train":
        return cmd_train(a.file, raw=a.raw)
    if a.cmd == "resume":
        return cmd_train(a.file, resume_only=True, raw=a.raw)
    if a.cmd == "export":
        return cmd_export(a.file, format_override=a.format, quant_override=a.quant)


if __name__ == "__main__":
    sys.exit(main())
