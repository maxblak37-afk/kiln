import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qdsl.codegen import generate  # noqa: E402
from qdsl.lexer import QdslSyntaxError, strip_comment  # noqa: E402
from qdsl.parser import parse  # noqa: E402
from qdsl.presets import HARDWARE  # noqa: E402
from qdsl.schema import validate  # noqa: E402

EX = os.path.join(ROOT, "examples")


def check(text):
    return validate(parse(text), HARDWARE)


class LexerTests(unittest.TestCase):
    def test_hash_inside_string_is_kept(self):
        self.assertEqual(strip_comment('x: "a#b"  # note'), 'x: "a#b"  ')

    def test_comment_removed(self):
        self.assertEqual(strip_comment("epochs: 3  # три эпохи"), "epochs: 3  ")


class ParserTests(unittest.TestCase):
    def test_duplicate_key_rejected(self):
        with self.assertRaises(QdslSyntaxError) as cm:
            parse("train {\n epochs: 1\n epochs: 2\n}\n")
        self.assertEqual(cm.exception.line, 3)

    def test_unclosed_block_rejected(self):
        with self.assertRaises(QdslSyntaxError):
            parse("train {\n epochs: 1\n")

    def test_extra_brace_rejected(self):
        with self.assertRaises(QdslSyntaxError):
            parse("}\n")


class SchemaTests(unittest.TestCase):
    def test_examples_are_valid(self):
        for name in ("minimal_3060.qdsl", "dual_t4.qdsl"):
            with open(os.path.join(EX, name), encoding="utf-8") as f:
                cfg, errs, _ = check(f.read())
            self.assertEqual(errs, [], name)
            self.assertIsNotNone(cfg, name)

    def test_bf16_rejected_on_t4(self):
        src = ('target: "2x-t4"\n'
               'model "Qwen/Qwen2.5-Coder-7B" {\n precision: bf16\n}\n'
               'dataset "d.jsonl" {\n}\n'
               'train {\n output: "o"\n}\n')
        _, errs, _ = check(src)
        self.assertTrue(any("bf16" in msg for _, msg in errs))

    def test_unknown_key_gets_hint(self):
        src = ('target: "rtx-3060-laptop"\n'
               'model "m" {\n}\n'
               'dataset "d.jsonl" {\n}\n'
               'train {\n epoch: 3\n output: "o"\n}\n')
        _, errs, _ = check(src)
        self.assertTrue(any("epochs" in msg for _, msg in errs))

    def test_missing_output_is_error(self):
        src = ('target: "rtx-3060-laptop"\n'
               'model "m" {\n}\n'
               'dataset "d.jsonl" {\n}\n'
               'train {\n epochs: 1\n}\n')
        cfg, errs, _ = check(src)
        self.assertIsNone(cfg)
        self.assertTrue(any("output" in msg for _, msg in errs))

    def test_lr_out_of_range(self):
        src = ('target: "rtx-3060-laptop"\n'
               'model "m" {\n}\n'
               'dataset "d.jsonl" {\n}\n'
               'train {\n lr: 5\n output: "o"\n}\n')
        _, errs, _ = check(src)
        self.assertTrue(any("lr" in msg for _, msg in errs))

    def test_vram_warning_for_7b_on_3060(self):
        src = ('target: "rtx-3060-laptop"\n'
               'model "Qwen/Qwen2.5-Coder-7B" {\n}\n'
               'dataset "d.jsonl" {\n}\n'
               'train {\n output: "o"\n}\n')
        _, _, warns = check(src)
        self.assertTrue(any("впритык" in msg or "не запустится" in msg for _, msg in warns))

    def test_effective_batch_on_t4(self):
        with open(os.path.join(EX, "dual_t4.qdsl"), encoding="utf-8") as f:
            cfg, _, _ = check(f.read())
        self.assertEqual(cfg["effective_batch"], 2 * 4 * 2)


class CodegenTests(unittest.TestCase):
    def test_generated_code_compiles(self):
        for name in ("minimal_3060.qdsl", "dual_t4.qdsl"):
            with open(os.path.join(EX, name), encoding="utf-8") as f:
                cfg, _, _ = check(f.read())
            code = generate(cfg, name)
            compile(code, name, "exec")

    def test_generated_constants(self):
        with open(os.path.join(EX, "dual_t4.qdsl"), encoding="utf-8") as f:
            cfg, _, _ = check(f.read())
        code = generate(cfg, "dual_t4.qdsl")
        self.assertIn("BF16 = False", code)
        self.assertIn("LOAD_4BIT = True", code)
        self.assertIn("BATCH = 2", code)


if __name__ == "__main__":
    unittest.main()
