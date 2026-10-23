"""Install SRT-M15's bundled core source into the checked-out GitHub tree.

The mobile Colab notebook contains the original starter modules in an embedded
ZIP. This local bootstrap extracts *only* known safe source files, without
overwriting any updated files maintained separately in this repository.
No network access required.
"""
import base64
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "SRT_M15_Phone_Colab.ipynb"
ARCHIVE_PREFIX = "SRT_M15_Starter/"
SOURCE_FILES = {
    "srt_m15/__init__.py",
    "srt_m15/model.py",
    "srt_m15/language.py",
    "srt_m15/tasks.py",
    "srt_m15/learning.py",
    "srt_m15/exploration.py",
    "srt_m15/storage.py",
    "train.py",
    "train_lm.py",
    "generate.py",
    "demo.py",
}


def bootstrap(root=ROOT):
    root = Path(root).resolve()
    notebook_path = root / NOTEBOOK.name
    cells = json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
    source_cell = next(
        ("" .join(cell.get("source", [])) for cell in cells
         if "SOURCE_ZIP_BASE64" in "".join(cell.get("source", []))),
        None,
    )
    if source_cell is None:
        raise ValueError("Notebook 中找不到嵌入式源代码。")
    match = re.search(r'^SOURCE_ZIP_BASE64\s*=\s*["\\x27]([A-Za-z0-9+/=]+)["\\x27]',
                      source_cell, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Notebook 中未找到合法的源代码压缩包常量。")
    archive_bytes = base64.b64decode(match.group(1), validate=True)
    installed, existing = [], []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as package:
        for name in sorted(SOURCE_FILES):
            member = ARCHIVE_PREFIX + name
            if member not in package.namelist():
                raise ValueError(f"源代码压缩包缺少 {member}")
            target = root.joinpath(*Path(name).parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError("Unsafe source file path")
            if target.is_file():
                existing.append(name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # These are preapproved, exact source filenames; never extractall().
            target.write_bytes(package.read(member))
            installed.append(name)
    return installed, existing


if __name__ == "__main__":
    new_files, skipped = bootstrap()
    print("已安装缺失的底层源码:", ", ".join(new_files) or "无")
    if skipped:
        print("已保留现有版本:", ", ".join(skipped))
    print("接下来可运行: pip install -r requirements_api.txt")
