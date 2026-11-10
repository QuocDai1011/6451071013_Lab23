"""In output text cua notebook de kiem tra. Dung: python tools/dump.py <file> [max_lines]"""
import sys
import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
maxl = int(sys.argv[2]) if len(sys.argv) > 2 else 30
for c in nb.cells:
    if c.cell_type != "code":
        continue
    print(f"--- [{c.metadata.get('cau')}]")
    for o in c.outputs:
        if o.output_type == "stream" and o.name == "stdout":
            lines = o.text.splitlines()
            print("\n".join(lines[:maxl] + (["..."] + lines[-5:] if len(lines) > maxl else [])))
        elif o.output_type == "error":
            print("ERROR", o.ename, o.evalue)
        elif o.output_type in ("execute_result", "display_data"):
            print("<", ",".join(o.data.keys()), ">")
