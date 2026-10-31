"""Схема Kiln: ключи, типы, диапазоны, пресеты, валидация, значения по умолчанию и семантические проверки."""
import difflib

from .parser import Block
from .presets import params_b, vram_need

REQUIRED = "__required__"

# Пресеты задают набор параметров целиком. Явно указанное в файле всегда важнее пресета.
PRESETS = {
    "quick":    {"epochs": 1, "lr": 2e-4, "r": 8,  "context": 512,  "checkpoint": 50},
    "balanced": {"epochs": 3, "lr": 2e-4, "r": 16, "context": 1024, "checkpoint": 100},
    "quality":  {"epochs": 5, "lr": 1e-4, "r": 32, "context": 2048, "checkpoint": 200},
}

ROOT_KEYS = {
    "target": ("str", None, REQUIRED),
    "backend": ("enum", ["hf", "unsloth"], None),          # по умолчанию берётся из профиля
    "preset": ("enum", list(PRESETS), None),               # quick | balanced | quality
}

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
        "train_on": ("enum", ["all", "responses_only"], None),  # None -> зависит от backend
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
    # Валидация: проверка качества на отдельных примерах. Нужна, чтобы ловить переобучение.
    "eval": {
        "every": ("int", (1, 10_000_000), 50),      # проверять каждые N шагов
        "patience": ("int", (1, 100), 3),           # остановить после N проверок без улучшения
    },
    "export": {
        "format": ("enum", ["gguf", "merged_16bit", "merged_4bit"], "gguf"),
        "quantization": ("enum", ["q4_k_m", "q8_0", "f16", "q5_k_m"], "q4_k_m"),
    },
}

BLOCK_ARGS = {"model": 1, "dataset": 1, "adapter": 1, "train": 0, "parallel": 0, "eval": 1, "export": 0}


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
    if b.name in ("model", "dataset", "eval") and not isinstance(b.args[0], str):
        errs.append((b.line, f"блок '{b.name}': аргумент должен быть строкой"))
    if b.name == "adapter" and b.args[0] != "lora":
        errs.append((b.line, "adapter поддерживает только 'lora'"))


def validate(root: Block, profiles: dict):
    """Возвращает (cfg | None, errors, warnings). Ошибки и предупреждения — списки (строка, текст)."""
    errs, warns = [], []

    top = {}
    for key, v in root.entries.items():
        if key in ROOT_KEYS:
            top[key] = v
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

    target = top["target"].val if "target" in top else None
    profile = profiles.get(target) if isinstance(target, str) else None
    if "target" in top and profile is None:
        errs.append((top["target"].line,
                     f"неизвестный target '{target}'; доступны: {', '.join(profiles)}"))

    backend = profile["backend"] if profile else None
    if "backend" in top:
        raw = top["backend"].val
        if _check("enum", ROOT_KEYS["backend"][1], "backend", raw, top["backend"].line, errs):
            backend = raw

    preset_name = None
    if "preset" in top:
        raw = top["preset"].val
        if _check("enum", ROOT_KEYS["preset"][1], "preset", raw, top["preset"].line, errs):
            preset_name = raw
    pv = PRESETS.get(preset_name, {})

    def dflt(key, schema_default):
        """Значение из пресета, если он задан, иначе значение по умолчанию схемы."""
        return pv.get(key, schema_default)

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
    ev = fields.get("eval")

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

    train_on = ds.get("train_on")
    if train_on is None:
        train_on = "responses_only" if backend == "unsloth" else "all"
    elif train_on == "responses_only" and backend == "hf":
        warns.append((line("dataset", "train_on"),
                      "responses_only на бэкенде hf пока не поддерживается, обучение идёт по всему тексту"))
        train_on = "all"

    batch = tr.get("batch", profile["batch"])
    accum = tr.get("accum", profile["accum"])
    r = ad.get("r", dflt("r", 16))
    ctx = mdl.get("context", dflt("context", 1024))

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

    epochs = tr.get("epochs", dflt("epochs", 1))
    checkpoint = tr.get("checkpoint", dflt("checkpoint", 100))

    eval_cfg = None
    if "eval" in blocks:
        eval_cfg = {
            "path": blocks["eval"].args[0],
            "every": ev.get("every", 50),
            "patience": ev.get("patience", 3),
        }
        # Чекпоинт и проверка должны совпадать по шагам, иначе лучший чекпоинт не выбрать
        checkpoint = eval_cfg["every"]
        if eval_cfg["every"] > 0 and not warns:
            pass

    export_cfg = None
    if "export" in blocks:
        ex = fields.get("export", {})
        fmt = ex.get("format", "gguf")
        quant = ex.get("quantization", "q4_k_m")
        if fmt == "gguf" and backend == "hf":
            warns.append((line("export", "format"),
                          "экспорт в gguf нативно поддерживается в unsloth; на hf бэкенде будет выполнено слияние в 16bit"))
        export_cfg = {
            "format": fmt,
            "quantization": quant,
        }

    cfg = {
        "target": target,
        "backend": backend,
        "preset": preset_name,
        "profile": profile,
        "gpus": gpus,
        "strategy": strategy,
        "model": {"name": blocks["model"].args[0], "precision": prec, "context": ctx},
        "data": {"path": blocks["dataset"].args[0],
                 "format": ds.get("format", "chatml"),
                 "train_on": train_on},
        "eval": eval_cfg,
        "lora": {"r": r, "alpha": ad.get("alpha", 2 * r)},
        "train": {"epochs": epochs,
                  "lr": tr.get("lr", dflt("lr", 2e-4)),
                  "batch": batch, "accum": accum, "output": tr["output"],
                  "checkpoint": checkpoint,
                  "resume": tr.get("resume", "auto")},
        "export": export_cfg,
        "effective_batch": batch * accum * gpus,
    }
    return cfg, sorted(errs), sorted(warns)
