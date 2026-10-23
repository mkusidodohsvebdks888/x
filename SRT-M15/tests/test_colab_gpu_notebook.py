"""Colab training notebook remains a valid, fully executable Python notebook.

These checks run in standard GitHub CI without mounting Drive or requiring GPU.
They validate syntax/invariants only; they do not prove remote GPU availability.
"""
import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "SRT_M15_110k_GPU_Colab.ipynb"


class TestColabGpuNotebook(unittest.TestCase):
    def test_notebook_json_schema_and_python_syntax(self):
        obj = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(obj["nbformat"], 4)
        self.assertGreaterEqual(len(obj["cells"]), 8)
        code_cells = [c for c in obj["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 6)
        for cell in code_cells:
            src = "".join(cell["source"])
            try:
                ast.parse(src, filename=cell.get("id", "notebook-cell"))
            except SyntaxError as e:
                self.fail(f"invalid Python code cell {cell.get('id')}: {e}")

    def test_gpu_and_checkpoint_boundaries_are_explicit(self):
        content = NOTEBOOK.read_text(encoding="utf-8")
        self.assertIn('torch.cuda.is_available()', content)
        self.assertIn('training_spec.json', content)
        self.assertIn('generate_bilingual_110k.py', content)
        self.assertIn('--sampling', content)
        self.assertIn('latest.pt', content)
        self.assertIn('best.pt', content)
        self.assertIn('history.json', content)
        self.assertIn('google.colab import drive', content)
        self.assertIn('os.replace(', content)
        self.assertNotIn('DEEPSEEK_API_KEY', content)
        self.assertNotIn('OPENAI_API_KEY', content)


if __name__ == "__main__":
    unittest.main()
