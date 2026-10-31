# Kiln

Декларативный язык (DSL) для простого и безопасного дообучения LLM на Unsloth и Hugging Face Transformers с поддержкой Kaggle, автоматической оценкой VRAM/токенов и экспортом в GGUF / 16bit.

## Установка

```bash
pip install unsloth trl datasets transformers bitsandbytes torch
```

## Команды

```bash
python -m qdsl check examples/smoke.qdsl      # проверка синтаксиса и схемы
python -m qdsl plan  examples/dual_t4.qdsl    # оценка VRAM, батча и анализ распределения токенов
python -m qdsl train examples/dual_t4.qdsl    # генерация скрипта и запуск обучения
python -m qdsl resume examples/dual_t4.qdsl   # продолжение с последнего валидного чекпоинта
python -m qdsl export examples/dual_t4.qdsl   # экспорт в GGUF (q4_k_m, q8_0) или слияние весов
```

Сгенерированный скрипт сохраняется в папку вывода рядом с результатами (`generated_train.py`).

## Пример конфигурации (`.kiln` / `.qdsl`)

```kiln
target: "2x-t4"

model "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" {
    precision: 4bit
    context: 1024
}

dataset "data/train.jsonl" {
    format: chatml
}

adapter lora {
    r: 16
    alpha: 32
}

train {
    epochs: 2
    lr: 2e-4
    batch: 2
    accum: 4
    output: "runs/qwen7b"
}

# Валидация и защита от переобучения (Early Stopping)
eval "data/val.jsonl" {
    every: 50
    patience: 3
}

# Автоэкспорт после обучения в GGUF (или merged_16bit)
export {
    format: gguf
    quantization: q4_k_m
}
```

## Возможности

- **Анализ токенов до старта (`plan`)**: мгновенно показывает min/avg/max токенов и процент примеров, влезающих в `context`.
- **Защита от OOM**: статическая оценка VRAM под профили (`rtx-3060-laptop`, `1x-t4`, `2x-t4`).
- **Защита от переобучения**: блок `eval` с Early Stopping и автоматическим сохранением лучшего чекпоинта.
- **Безопасная остановка**: создайте пустой файл `STOP` в папке вывода, чтобы обучение аккуратно сохранило чекпоинт и завершилось без Ctrl+C.
- **NanGuard**: аварийный останов при NaN/Inf в loss.
- **Кроссплатформенность**: Unsloth на Linux/Kaggle, нативный Transformers/PEFT на Windows.

## Тесты

```bash
python -m unittest tests.test_qdsl -v
```
