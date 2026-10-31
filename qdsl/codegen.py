"""Генерация самостоятельного train.py из проверенного конфига. Два бэкенда: hf и unsloth."""

BODY_UNSLOTH = r'''
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=CONTEXT, load_in_4bit=LOAD_4BIT, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)
'''

BODY_HF = r'''
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

compute_dtype = torch.bfloat16 if BF16 else torch.float16
local_rank = int(os.environ.get("LOCAL_RANK", 0))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb = None
if LOAD_4BIT:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, dtype=compute_dtype,
    device_map={"": local_rank}, attn_implementation="sdpa")
if LOAD_4BIT:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
# Адаптеры в том же типе, что и вычисления, иначе GradScaler падает на смеси типов
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.to(compute_dtype)
model.print_trainable_parameters()
'''

# responses_only доступен только в бэкенде unsloth; для hf схема заранее ставит 'all'
BODY_COMMON = r'''
from transformers import EarlyStoppingCallback


def to_messages(ex):
    if DATA_FORMAT == "alpaca":
        user = ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")
        return [{"role": "user", "content": user}, {"role": "assistant", "content": ex["output"]}]
    return ex["messages"]


def to_text(ex):
    return {"text": tokenizer.apply_chat_template(to_messages(ex), tokenize=False)}


def load_split(path):
    ds = load_dataset("json", data_files=path, split="train").map(to_text)
    return ds.filter(lambda ex: len(tokenizer(ex["text"])["input_ids"]) <= CONTEXT)


dataset = load_split(DATA)
print("[kiln] примеров для обучения:", len(dataset))

eval_dataset = None
if EVAL_DATA:
    eval_dataset = load_split(EVAL_DATA)
    print("[kiln] примеров для валидации:", len(eval_dataset),
          "| проверка каждые", EVAL_EVERY, "шагов, остановка после", EVAL_PATIENCE, "без улучшения")

STOP_FILE = os.path.join(OUTPUT, "STOP")
CHECKPOINT_FILES = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_model.safetensors"]


def find_valid_checkpoint(out_dir):
    """Последний ПОЛНЫЙ чекпоинт. Неполный (прерванная запись) пропускается, а не ломает resume."""
    if not os.path.isdir(out_dir):
        return None
    found = []
    for name in os.listdir(out_dir):
        full = os.path.join(out_dir, name)
        if name.startswith("checkpoint-") and os.path.isdir(full):
            try:
                found.append((int(name.split("-", 1)[1]), full))
            except ValueError:
                pass
    for _, full in sorted(found, reverse=True):
        missing = [f for f in CHECKPOINT_FILES if not os.path.exists(os.path.join(full, f))]
        if not missing:
            return full
        print(f"[kiln] пропускаю неполный чекпоинт {full} (нет: {', '.join(missing)})")
    return None


class NanGuard(TrainerCallback):
    """Останавливает обучение при NaN/Inf в loss. Чекпоинты до этого уже сохранены."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            raise RuntimeError(f"loss = {loss} на шаге {state.global_step}: обучение остановлено")


class StopFile(TrainerCallback):
    """Безопасная остановка: создайте файл STOP в папке вывода. Текущий шаг завершится,
    чекпоинт сохранится полностью, и обучение выйдет. Ctrl+C во время записи не нужен."""
    def on_step_end(self, args, state, control, **kwargs):
        if os.path.exists(STOP_FILE):
            print("[kiln] получен STOP: сохраняю чекпоинт и останавливаюсь")
            control.should_save = True
            control.should_training_stop = True
        return control


# Если задан eval, сохраняем и проверяем с одинаковым шагом, иначе лучший чекпоинт не выбрать
save_every = EVAL_EVERY if EVAL_DATA else CHECKPOINT

sft_kwargs = dict(
    output_dir=OUTPUT, num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
    logging_steps=10, save_steps=save_every, save_total_limit=3,
    bf16=BF16, fp16=not BF16, optim=OPTIM, report_to="none",
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="text", seed=3407)

callbacks = [NanGuard(), StopFile()]
if EVAL_DATA:
    sft_kwargs.update(
        eval_strategy="steps", eval_steps=EVAL_EVERY,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False)
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=EVAL_PATIENCE))

args = SFTConfig(**sft_kwargs)

trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset,
                     eval_dataset=eval_dataset, args=args, callbacks=callbacks)
if TRAIN_ON == "responses_only":
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

# Старый флаг STOP от прошлого запуска убираем, иначе обучение остановится на первом шаге
if os.path.exists(STOP_FILE):
    os.remove(STOP_FILE)

resume = find_valid_checkpoint(OUTPUT) if RESUME == "auto" else None
if resume:
    print("[kiln] продолжаю с", resume)

trainer.train(resume_from_checkpoint=resume)

if trainer.is_world_process_zero():
    if EVAL_DATA and trainer.state.best_model_checkpoint:
        print("[kiln] лучший чекпоинт по валидации:", trainer.state.best_model_checkpoint,
              "| eval_loss =", round(trainer.state.best_metric, 4))
    final = os.path.join(OUTPUT, "final")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print("[kiln] готово:", final)

    if EXPORT_FORMAT:
        export_dir = os.path.join(OUTPUT, "export")
        os.makedirs(export_dir, exist_ok=True)
        if BACKEND == "unsloth":
            if EXPORT_FORMAT == "gguf":
                print(f"[kiln] экспорт в GGUF ({EXPORT_QUANT})...")
                model.save_pretrained_gguf(export_dir, tokenizer, quantization_method=EXPORT_QUANT)
            elif EXPORT_FORMAT == "merged_16bit":
                print("[kiln] слияние LoRA и сохранение в 16bit...")
                model.save_pretrained_merged(export_dir, tokenizer, save_method="merged_16bit")
            elif EXPORT_FORMAT == "merged_4bit":
                print("[kiln] слияние LoRA и сохранение в 4bit...")
                model.save_pretrained_merged(export_dir, tokenizer, save_method="merged_4bit")
        else:
            print("[kiln] слияние LoRA и сохранение весов...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(export_dir)
            tokenizer.save_pretrained(export_dir)
        print("[kiln] экспорт завершён:", export_dir)
'''


def generate(cfg: dict, source: str) -> str:
    m, d, a, t = cfg["model"], cfg["data"], cfg["lora"], cfg["train"]
    ev = cfg.get("eval")
    ex = cfg.get("export")
    backend = cfg["backend"]
    # bf16 — если железо его поддерживает и пользователь не просил fp16
    use_bf16 = bool(cfg["profile"]["bf16"]) and m["precision"] != "fp16"
    consts = [
        ("MODEL", m["name"]),
        ("CONTEXT", m["context"]),
        ("LOAD_4BIT", m["precision"] == "4bit"),
        ("LORA_R", a["r"]),
        ("LORA_ALPHA", a["alpha"]),
        ("DATA", d["path"]),
        ("DATA_FORMAT", d["format"]),
        ("TRAIN_ON", d["train_on"]),
        ("EVAL_DATA", ev["path"] if ev else None),
        ("EVAL_EVERY", ev["every"] if ev else None),
        ("EVAL_PATIENCE", ev["patience"] if ev else None),
        ("EPOCHS", t["epochs"]),
        ("LR", t["lr"]),
        ("BATCH", t["batch"]),
        ("ACCUM", t["accum"]),
        ("OUTPUT", t["output"]),
        ("CHECKPOINT", t["checkpoint"]),
        ("RESUME", t["resume"]),
        ("BF16", use_bf16),
        ("OPTIM", "adamw_8bit" if backend == "unsloth" else "adamw_torch"),
        ("BACKEND", backend),
        ("EXPORT_FORMAT", ex["format"] if ex else None),
        ("EXPORT_QUANT", ex["quantization"] if ex else None),
    ]
    lines = [
        f'"""Сгенерировано Kiln из {source} (бэкенд: {backend}). Не редактируйте вручную."""',
        "import math",
        "import os",
        "",
    ]
    lines += [f"{name} = {value!r}" for name, value in consts]
    body_head = BODY_UNSLOTH if backend == "unsloth" else BODY_HF
    lines += ["", body_head.strip("\n"), "", BODY_COMMON.strip("\n"), ""]
    return "\n".join(lines)
