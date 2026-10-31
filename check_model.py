"""Быстрая проверка: базовая модель + LoRA-адаптер из runs/smoke/final."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"F:\models\Qwen2.5-0.5B-Instruct"
ADAPTER = r"runs\smoke\final"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

questions = [
    "Как называется столица Польши?",
    "Что такое LoRA?",
    "Сколько будет 12 умножить на 3?",
]
for q in questions:
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("Q:", q)
    print("A:", answer.strip())
    print()
