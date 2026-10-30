"""Генерация самостоятельного train.py из проверенного конфига."""

BODY = r'''
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


def to_messages(ex):
    if DATA_FORMAT == "alpaca":
        user = ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")
        return [{"role": "user", "content": user}, {"role": "assistant", "content": ex["output"]}]
    return ex["messages"]


def to_text(ex):
    return {"text": tokenizer.apply_chat_template(to_messages(ex), tokenize=False)}


dataset = load_dataset("json", data_files=DATA, split="train").map(to_text)


class NanGuard(TrainerCallback):
    """Останавливает обучение при NaN/Inf в loss. Чекпоинты до этого уже сохранены."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            raise RuntimeError(f"loss = {loss} на шаге {state.global_step}: обучение остановлено")


args = SFTConfig(
    output_dir=OUTPUT, num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
    logging_steps=10, save_steps=CHECKPOINT, save_total_limit=3,
    bf16=BF16, fp16=not BF16, optim="adamw_8bit", report_to="none",
    dataset_text_field="text", seed=3407)

trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset,
                     args=args, callbacks=[NanGuard()])
if TRAIN_ON == "responses_only":
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

resume = None
if RESUME == "auto" and os.path.isdir(OUTPUT):
    resume = get_last_checkpoint(OUTPUT)
    if resume:
        print("[qdsl] продолжаю с", resume)

trainer.train(resume_from_checkpoint=resume)

if trainer.is_world_process_zero():
    final = os.path.join(OUTPUT, "final")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print("[qdsl] готово:", final)
'''


def generate(cfg: dict, source: str) -> str:
    m, d, a, t = cfg["model"], cfg["data"], cfg["lora"], cfg["train"]
    consts = [
        ("MODEL", m["name"]),
        ("CONTEXT", m["context"]),
        ("LOAD_4BIT", m["precision"] == "4bit"),
        ("LORA_R", a["r"]),
        ("LORA_ALPHA", a["alpha"]),
        ("DATA", d["path"]),
        ("DATA_FORMAT", d["format"]),
        ("TRAIN_ON", d["train_on"]),
        ("EPOCHS", t["epochs"]),
        ("LR", t["lr"]),
        ("BATCH", t["batch"]),
        ("ACCUM", t["accum"]),
        ("OUTPUT", t["output"]),
        ("CHECKPOINT", t["checkpoint"]),
        ("RESUME", t["resume"]),
        ("BF16", m["precision"] == "bf16"),
    ]
    lines = [
        f'"""Сгенерировано QDSL из {source}. Не редактируйте вручную: меняйте .qdsl и пересоберите."""',
        "import math",
        "import os",
        "",
    ]
    lines += [f"{name} = {value!r}" for name, value in consts]
    lines += ["", BODY.strip("\n"), ""]
    return "\n".join(lines)
