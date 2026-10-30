"""Парсер QDSL: строит дерево блоков. Синтаксические ошибки — с номером строки."""
import re
from dataclasses import dataclass, field

from .lexer import iter_lines, QdslSyntaxError

KEY_RE = re.compile(r"([A-Za-z_]\w*)\s*:\s*(.+)")
NAME_RE = re.compile(r"[A-Za-z_]\w*")
TOKEN_RE = re.compile(r'"[^"]*"|\S+')
NUM_RE = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?")
IDENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\-]*")


@dataclass
class Value:
    val: object
    line: int


@dataclass
class Block:
    name: str
    args: list
    line: int
    entries: dict = field(default_factory=dict)
    blocks: list = field(default_factory=list)


def parse_value(text: str, line: int):
    if text.startswith('"'):
        if len(text) < 2 or not text.endswith('"') or '"' in text[1:-1]:
            raise QdslSyntaxError(line, f"некорректная строка {text}")
        return text[1:-1]
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if NUM_RE.fullmatch(text):
        return float(text)
    if IDENT_RE.fullmatch(text):
        return text
    raise QdslSyntaxError(line, f"недопустимое значение '{text}' (строки берите в кавычки)")


def parse(src: str) -> Block:
    root = Block("", [], 0)
    stack = [root]
    for n, text in iter_lines(src):
        cur = stack[-1]
        if text == "}":
            if len(stack) == 1:
                raise QdslSyntaxError(n, "лишняя '}'")
            stack.pop()
        elif text.endswith("{"):
            tokens = TOKEN_RE.findall(text[:-1])
            if not tokens or not NAME_RE.fullmatch(tokens[0]):
                raise QdslSyntaxError(n, 'блок должен начинаться с имени, например: model "..." {')
            block = Block(tokens[0], [parse_value(t, n) for t in tokens[1:]], n)
            cur.blocks.append(block)
            stack.append(block)
        else:
            m = KEY_RE.fullmatch(text)
            if not m:
                raise QdslSyntaxError(n, f"не понимаю строку: '{text}'")
            key, raw = m.group(1), m.group(2).strip()
            if key in cur.entries:
                raise QdslSyntaxError(n, f"ключ '{key}' уже задан на строке {cur.entries[key].line}")
            cur.entries[key] = Value(parse_value(raw, n), n)
    if len(stack) > 1:
        raise QdslSyntaxError(stack[-1].line, f"блок '{stack[-1].name}' не закрыт: не хватает '}}'")
    return root
