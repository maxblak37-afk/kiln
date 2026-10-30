"""Лексер QDSL: вырезает комментарии (с учётом строк в кавычках) и отдаёт непустые строки."""


class QdslSyntaxError(Exception):
    def __init__(self, line: int, msg: str):
        super().__init__(f"строка {line}: {msg}")
        self.line = line
        self.msg = msg


def strip_comment(line: str) -> str:
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif ch == "#" and not in_str:
            return line[:i]
    return line


def iter_lines(src: str):
    for n, raw in enumerate(src.splitlines(), 1):
        text = strip_comment(raw).strip()
        if text:
            yield n, text
