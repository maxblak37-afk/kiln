"""Команды QDSL: check, plan, train, resume."""
import argparse
import os
import subprocess
import sys

from .codegen import generate
from .lexer import QdslSyntaxError
from .parser import parse
from .presets import HARDWARE
from .schema import validate


def load(path: str):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    try:
        root = parse(src)
    except QdslSyntaxError as e:
        return None, [(e.line, e.msg)], []
    return validate(root, HARDWARE)


def show(errs, warns):
    for line, msg in errs:
        print(f"ошибка, строка {line}: {msg}")
    for line, msg in warns:
        print(f"предупреждение, строка {line}: {msg}")


def cmd_check(path: str) -> int:
    cfg, errs, warns = load(path)
    show(errs, warns)
    if errs or cfg is None:
        print(f"найдено ошибок: {len(errs)}")
        return 1
    print("OK: файл корректен")
    return 0


def cmd_plan(path: str) -> int:
    cfg, errs, warns = load(path)
    show(errs, warns)
    if errs or cfg is None:
        return 1
    m, t, p = cfg["model"], cfg["train"], cfg["profile"]
    print(f"Модель:            {m['name']} ({m['precision']}, context {m['context']})")
    print(f"Железо:            {cfg['target']}, {cfg['gpus']} x {p['vram_gb']} ГБ")
    print(f"Стратегия:         {cfg['strategy']}")
    print(f"Батч:              {t['batch']} x accum {t['accum']} x gpus {cfg['gpus']}"
          f" = эффективный {cfg['effective_batch']}")
    print(f"LoRA:              r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}")
    print(f"Эпохи / lr:        {t['epochs']} / {t['lr']}")
    print(f"Чекпоинты:         каждые {t['checkpoint']} шагов, resume={t['resume']}")
    print(f"Выход:             {t['output']}")
    return 0


def cmd_train(path: str, resume_only: bool = False) -> int:
    cfg, errs, warns = load(path)
    show(errs, warns)
    if errs or cfg is None:
        return 1
    out = cfg["train"]["output"]
    if resume_only:
        has_ckpt = os.path.isdir(out) and any(d.startswith("checkpoint-") for d in os.listdir(out))
        if not has_ckpt:
            print(f"ошибка: в '{out}' нет чекпоинтов для продолжения")
            return 1
        cfg["train"]["resume"] = "auto"
    os.makedirs(out, exist_ok=True)
    script = os.path.join(out, "generated_train.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(generate(cfg, os.path.basename(path)))
    print(f"Сгенерирован скрипт: {script}")

    n = cfg["gpus"] if cfg["strategy"] == "ddp" else 1
    if n == 1:
        cmd = [sys.executable, script]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n}", script]
    print("Запуск:", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="qdsl", description="QDSL: дообучение LLM на Unsloth")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("check", "проверить файл"),
                            ("plan", "показать итоговую конфигурацию"),
                            ("train", "запустить обучение"),
                            ("resume", "продолжить с последнего чекпоинта")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("file", help="путь к .qdsl файлу")
    a = p.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a.file)
    if a.cmd == "plan":
        return cmd_plan(a.file)
    if a.cmd == "train":
        return cmd_train(a.file)
    return cmd_train(a.file, resume_only=True)


if __name__ == "__main__":
    sys.exit(main())
