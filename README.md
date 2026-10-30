# QDSL

Декларативный язык для дообучения LLM на Unsloth. Спецификация: SPEC.md.

## Установка

    pip install unsloth trl datasets transformers torch

## Команды (запускать из F:\DSL)

    python -m qdsl check examples\minimal_3060.qdsl     # проверка файла
    python -m qdsl plan  examples\dual_t4.qdsl          # итоговая конфигурация и батч
    python -m qdsl train examples\dual_t4.qdsl          # генерация скрипта и запуск
    python -m qdsl resume examples\dual_t4.qdsl         # продолжить с последнего чекпоинта

Сгенерированный скрипт сохраняется в папку output, рядом с результатами (generated_train.py).

## Тесты

    python -m unittest tests.test_qdsl -v

## Структура

    qdsl\
      lexer.py      # комментарии с учётом строк, номера строк
      parser.py     # дерево блоков
      schema.py     # ключи, типы, диапазоны, подсказки, проверка VRAM
      presets.py    # профили железа и таблицы VRAM
      codegen.py    # генерация train.py
      cli.py        # команды
      __main__.py   # точка входа python -m qdsl
    examples\       # готовые примеры
    tests\          # unit-тесты

## Важно

- На T4 bf16 не поддерживается. Используйте precision: 4bit (обучение в fp16).
- Проверка VRAM — оценка по документации Unsloth, на одну карту. Перед длинным запуском проверьте `plan`.
