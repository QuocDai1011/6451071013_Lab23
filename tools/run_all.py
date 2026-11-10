"""Chay tat ca notebook va luu output vao chinh file. Dung: python tools/run_all.py [ten_file ...]"""
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
files = sys.argv[1:] or sorted(p.name for p in ROOT.glob("6451071013_LAB3_*.ipynb"))
for name in files:
    path = ROOT / name
    nb = nbformat.read(path, as_version=4)
    t = time.time()
    try:
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        status = "OK"
    except Exception as e:  # van luu de xem loi
        status = f"FAIL: {str(e)[-600:]}"
    nbformat.write(nb, path)
    print(f"{name}: {status} ({time.time() - t:.0f}s)", flush=True)
