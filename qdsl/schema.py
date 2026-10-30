"""Схема QDSL: ключи, типы, диапазоны, значения по умолчанию и семантические проверки."""
import difflib

from .parser import Block
from .presets import params_b, vram_need

REQUIRED = "__required__"

ROOT_KEYS = {"target": ("str", None, REQUIRED)}

SECTIONS = {
    "parallel": {
        "strategy": ("enum", ["single", "ddp"], None),
        "gpus": ("int", (1, 64), None),
    },
    "model": {
        "precision": ("enum", ["4bit", "16bit", "fp16", "bf16"], "4bit"),
        "context": ("int", (64, 131072), 1024),
    },
    "dataset": {
        "format": ("enum", ["chatml", "alpaca"], "chatml"),
        "train_on": ("enum", ["all", "responses_only"], "responses_only"),
    },
    "adapter": {
        "r": ("int", (1, 512), 16),
        "alpha": ("int", (1, 1024), None),
    },
    "train": {
        "epochs": ("float", (0, 1000), 1),
        "lr": ("float", (1e-6, 1e-1), 2e-4),
        "batch": ("int", (1, 4096), None),
        "accum": ("int", (1, 4096), None),
        "output": ("str", None, REQUIRED),
        "checkpoint": ("int", (1, 10_000_000), 100),
        "resume": ("enum", ["auto", "off"], "auto"),
    },
}

BLOCK_ARGS = {"model": 1, "dataset": 1, "adapter": 1, "train": 0, "parallel": 0}


def _check(kind, rng, key, val, line, errs):
    if kind == "str":
        if not isinstance(val, str):
            errs.append((line, f"'{key}': ожидается строка"))
            return False
        return True
    if kind == "enum":
        if val not in rng:
            errs.append((line, f"'{key}': допустимо {' | '.join(rng)}, получено '{val}'"))
            return False
        return True
    is_bool = isinstance(val, bool)
    ok = isinstance(val, int) and not is_bool if kind == "int" else isinstance(val, (int, float)) and not is_bool
    if not ok:
        word = "целое число" if kind == "int" else "число"
        errs.append((line, f"'{key}': ожидается {word}, получено '{val}'"))
        return False
    if rng and not (rng[0] <= val <= rng[1]):
        errs.append((line, f"'{key}': значение {val} вне диапазона {rng[0]}..{rng[1]}"))
        return False
    return True


def _fill(block: Block, section: str, errs, out: dict):
    spec = SECTIONS[section]
    for key, v in block.entries.items():
        if key not in spec:
            near = difflib.get_close_matches(key, list(spec), n=1)
            hint = f", возможно '{near[0]}'" if near else f"; допустимы: {', '.join(spec)}"
            errs.append((v.line, f"[{section}] неизвестный ключ '{key}'{hint}"))
            continue
        kind, rng, _ = spec[key]
        if _check(kind, rng, key, v.val, v.line, errs):
            out[key] = v.val


def _check_args(b: Block, errs):
    want = BLOCK_ARGS[b.name]
    if len(b.args) != want:
        errs.append((b.line, f"блок '{b.name}' ожидает аргументов: {want}, получено: {len(b.args)}"))
        return
    if b.name in ("model", "dataset") and not isinstance(b.args[0], str):
        errs.append((b.line, f"блок '{b.name}': аргумент должен быть строкой"))
    if b.name == "adapter" and b.args[0] != "lora":
        errs.append((b.line, "adapter поддерживает только 'lora'"))


def validate(root: Block, profiles: dict):
    """Возвращает (cfg | None, errors, warnings). Ошибки и предупреждения — списки (строка, текст)."""
    errs, warns = [], []

    top = {}
    for key, v in root.entries.items():
        if key in ROOT_KEYS:
            top[key] = v.val
        else:
            errs.append((v.line, f"неизвестный ключ верхнего уровня '{key}'"))

    blocks = {}
    for b in root.blocks:
        if b.name not in SECTIONS:
            errs.append((b.line, f"неизвестный блок '{b.name}'; допустимы: {', '.join(SECTIONS)}"))
        elif b.name in blocks:
            errs.append((b.line, f"блок '{b.name}' уже объявлен на строке {blocks[b.name].line}"))
        else:
            blocks[b.name] = b
            _check_args(b, errs)

    for name in ("model", "dataset", "train"):
        if name not in blocks:
            errs.append((1, f"отсутствует обязательный блок '{name}'"))
    if "target" not in top:
        errs.append((1, "отсутствует обязательный ключ 'target'"))

    target = top.get("target")
    profile = profiles.get(target) if isinstance(target, str) else None
    if "target" in top and profile is None:
        errs.append((root.entries["target"].line,
                     f"неизвестный target '{target}'; доступны: {', '.join(profiles)}"))

    fields, missing_required = {}, False
    for name, b in blocks.items():
        vals = {}
        _fill(b, name, errs, vals)
        for key, (_, _, default) in SECTIONS[name].items():
            if default == REQUIRED and key not in vals:
                errs.append((b.line, f"[{name}] отсутствует обязательный ключ '{key}'"))
                missing_required = True
        fields[name] = vals

    if profile is None or missing_required or any(n not in blocks for n in ("model", "dataset", "train")):
        return None, sorted(errs), sorted(warns)

    def line(section, key=None):
        b = blocks.get(section)
        if b is None:
            return 1
        if key and key in b.entries:
            return b.entries[key].line
        return b.line

    mdl, tr, par, ad, ds = (fields.get("model", {}), fields["train"], fields.get("parallel", {}),
                            fields.get("adapter", {}), fields["dataset"])

    prec = mdl.get("precision", "4bit")
    if prec == "bf16" and not profile["bf16"]:
        errs.append((line("model", "precision"),
                     f"precision 'bf16' не поддерживается target '{target}': используйте fp16"))

    gpus = par.get("gpus", profile["gpus"])
    if gpus > profile["gpus"]:
        errs.append((line("parallel", "gpus"),
                     f"gpus={gpus}, а в target '{target}' только {profile['gpus']}"))
    strategy = par.get("strategy", "ddp" if gpus > 1 else "single")
    if strategy == "single" and gpus > 1:
        errs.append((line("parallel", "strategy"), "strategy 'single' требует gpus: 1"))
    if strategy == "ddp" and gpus == 1:
        warns.append((line("parallel", "strategy"), "ddp на одной карте работает как single"))

    batch = tr.get("batch", profile["batch"])
    accum = tr.get("accum", profile["accum"])
    r = ad.get("r", 16)
    ctx = mdl.get("context", 1024)

    params = params_b(blocks["model"].args[0])
    if params is None:
        warns.append((line("model"), "размер модели не найден в имени, проверка VRAM пропущена"))
    else:
        need = vram_need(params, prec)
        if need is None:
            warns.append((line("model"), "модель крупнее 70B, оценка VRAM недоступна"))
        elif need > profile["vram_gb"]:
            warns.append((line("model"),
                          f"оценка ~{need} ГБ на карту больше {profile['vram_gb']} ГБ: обучение, скорее всего, не запустится"))
        elif need > 0.8 * profile["vram_gb"]:
            warns.append((line("model"),
                          f"впритык: ~{need} ГБ из {profile['vram_gb']} ГБ на карту"))
    if ctx > 2048 and profile["vram_gb"] <= 8:
        warns.append((line("model", "context"),
                      f"context {ctx} на {profile['vram_gb']} ГБ может не влезть"))

    cfg = {
        "target": target,
        "profile": profile,
        "gpus": gpus,
        "strategy": strategy,
        "model": {"name": blocks["model"].args[0], "precision": prec, "context": ctx},
        "data": {"path": blocks["dataset"].args[0],
                 "format": ds.get("format", "chatml"),
                 "train_on": ds.get("train_on", "responses_only")},
        "lora": {"r": r, "alpha": ad.get("alpha", 2 * r)},
        "train": {"epochs": tr.get("epochs", 1), "lr": tr.get("lr", 2e-4),
                  "batch": batch, "accum": accum, "output": tr["output"],
                  "checkpoint": tr.get("checkpoint", 100), "resume": tr.get("resume", "auto")},
        "effective_batch": batch * accum * gpus,
    }
    return cfg, sorted(errs), sorted(warns)
