"""Tao bao cao Word tu cac notebook da chay. Dung: python tools/build_report.py"""
import base64
import io
import re
from pathlib import Path

import nbformat
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor, Cm
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
import tempfile
IMG_DIR = Path(tempfile.mkdtemp(prefix="lab3_report_"))
MSSV = "6451071013"
REPO_LINK = "https://github.com/<tài-khoản>/6451071013_Lab23"

# (file, tieu de trong bao cao, co dung chu "Cau" truoc nhan cell hay khong)
SECTIONS = [
    ("Bai1", "Bài 1 (Trang 1)", True),
    ("Bai2", "Bài 2 (Trang 5)", True),
    ("Bai3", "Bài 3 (Trang 7)", True),
    ("Bai4", "Bài 4 (Trang 9)", True),
    ("Bai5", "Bài 5 (Trang 9 – 11) – Dự đoán doanh thu game", True),
    ("Bai2_Phan2", "Bài 2 (Trang 18 - 19)", True),
    ("Bai5_Phan2", "Bài 5 (Trang 26)", False),
    ("Bai6_Phan2", "Bài 6 (Trang 26)", False),
    ("Bai5_BaiTapThem", "Bài 5 (Trang 29)", False),
]

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT = ImageFont.truetype(FONT_PATH, 15)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
img_counter = 0


# --------------------------------------------------------------------- helpers
def clean_stream(text):
    out = []
    for line in text.split("\n"):
        line = ANSI.sub("", line).rstrip("\r")
        line = line.split("\r")[-1].replace("\x08", "")
        out.append(line.rstrip())
    return out


def wrap(lines, width=118):
    res = []
    for l in lines:
        while len(l) > width:
            res.append(l[:width]); l = "  " + l[width:]
        res.append(l)
    return res


def truncate(lines, head=22, tail=14):
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) <= head + tail + 4:
        return lines
    return lines[:head] + [f"... (lược bớt {len(lines) - head - tail} dòng) ..."] + lines[-tail:]


def text_to_png(lines):
    """Ve output dang text thanh anh giong o output cua Jupyter."""
    global img_counter
    lines = wrap(truncate(lines))
    bbox = FONT.getbbox("Ag")
    lh = bbox[3] + 5
    w = max(FONT.getlength(l) for l in lines) if lines else 100
    W, H = int(max(w, 300)) + 28, lh * len(lines) + 22
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W - 1, H - 1], outline=(200, 200, 200))
    y = 11
    for l in lines:
        d.text((14, y), l, font=FONT, fill=(20, 20, 20))
        y += lh
    img_counter += 1
    p = IMG_DIR / f"out_{img_counter:03d}.png"
    im.save(p)
    return p, W


def save_b64_png(b64):
    global img_counter
    img_counter += 1
    p = IMG_DIR / f"fig_{img_counter:03d}.png"
    p.write_bytes(base64.b64decode(b64))
    with Image.open(p) as im:
        return p, im.size[0]


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def set_font(run, name="Times New Roman", size=13, bold=False, italic=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size); run.bold = bold; run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def para(doc, text="", size=13, bold=False, italic=False, align=None, color=None, space_after=6, underline=False):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        r = p.add_run(text)
        set_font(r, size=size, bold=bold, italic=italic, color=color)
        r.underline = underline
    return p


def add_code(doc, src):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    cell.width = Cm(16)
    shade(cell, "F3F3F3")
    lines = src.split("\n")
    first = True
    for line in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(line if line else " ")
        set_font(r, name="Consolas", size=8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_image(doc, path, px_width):
    width = min(Inches(6.3), Inches(px_width / 110))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    p.add_run().add_picture(str(path), width=width)


def add_outputs(doc, cell):
    text_buf = []

    def flush():
        if any(l.strip() for l in text_buf):
            p, w = text_to_png(list(text_buf))
            add_image(doc, p, w)
        text_buf.clear()

    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            if out.name == "stderr":
                continue
            text_buf.extend(clean_stream(out.text))
        elif out.output_type in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "image/png" in data:
                flush()
                p, w = save_b64_png(data["image/png"])
                add_image(doc, p, w)
            elif "text/plain" in data and not data["text/plain"].startswith("<"):
                text_buf.extend(clean_stream(data["text/plain"]))
        elif out.output_type == "error":
            text_buf.extend(clean_stream("\n".join(out.traceback)))
    flush()


def md_to_paras(doc, src):
    for line in src.split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        bullet = s.startswith(("- ", "* "))
        if bullet:
            s = s[2:]
        s = s.replace("`", "")
        p = doc.add_paragraph(style="List Bullet" if bullet else None)
        p.paragraph_format.space_after = Pt(4)
        # **dam**
        for i, part in enumerate(re.split(r"\*\*", s)):
            if part:
                set_font(p.add_run(part), size=12, bold=(i % 2 == 1))


def page_border(section):
    sectPr = section._sectPr
    pg = OxmlElement("w:pgBorders")
    pg.set(qn("w:offsetFrom"), "page")
    for side in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "thinThickSmallGap"); b.set(qn("w:sz"), "24")
        b.set(qn("w:space"), "24"); b.set(qn("w:color"), "1F3864")
        pg.append(b)
    sectPr.append(pg)


# --------------------------------------------------------------------- document
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"; st.font.size = Pt(13)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2)
page_border(sec)

# ---- Trang bia
para(doc, "PHÂN HIỆU TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI", 14, True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
para(doc, "TẠI TP. HỒ CHÍ MINH", 14, True, align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "BỘ MÔN CÔNG NGHỆ THÔNG TIN", 14, True, align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "--------", 13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=90)
para(doc, "KHAI PHÁ DỮ LIỆU", 18, True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
para(doc, "BÁO CÁO THỰC HÀNH BUỔI 3", 26, True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=110)
para(doc, f"Mai Quốc Đại - {MSSV}", 14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para(doc, "CQ.64.CNTT", 14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=70)
para(doc, "GVHD: Ths. Nguyễn Thiện Dương", 13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=60)
para(doc, "TP. Hồ Chí Minh - 2025", 13, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

# ---- Noi dung (section moi, khong vien trang, co header)
body = doc.add_section(WD_SECTION.NEW_PAGE)
body.footer.is_linked_to_previous = False
body.header.is_linked_to_previous = False
hp = body.header.paragraphs[0]
set_font(hp.add_run("KHAI PHÁ DỮ LIỆU\t\tThs. Nguyễn Thiện Dương"), size=10)
pPr = hp._p.get_or_add_pPr()
bdr = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "000000")):
    bt.set(qn(k), v)
bdr.append(bt); pPr.append(bdr)
fp = body.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if tag:
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tag); r._r.append(fc)
    else:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._r.append(it)
sp = body._sectPr.find(qn("w:pgBorders"))
if sp is not None:
    body._sectPr.remove(sp)

para(doc, "BÁO CÁO THỰC HÀNH", 16, True, align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "Link repository GitHub (public) chứa tất cả các bài:", 13, True, color=(0xC0, 0, 0), space_after=2)
para(doc, REPO_LINK, 12, italic=True, space_after=4)
para(doc, "Môi trường: Python 3.11, TensorFlow/Keras 3, scikit-learn, pandas, matplotlib, seaborn, plotly. "
          "Mỗi bài là một file mã nguồn riêng <MSSV>_LAB3_Bai<X>.ipynb; mỗi câu đều in họ tên – MSSV bằng lệnh print.",
     11, italic=True, space_after=10)

for fname, title, is_cau in SECTIONS:
    path = ROOT / f"{MSSV}_LAB3_{fname}.ipynb"
    nb = nbformat.read(path, as_version=4)
    para(doc, title, 14, True, underline=True, space_after=2)
    para(doc, f"File mã nguồn: {path.name}", 11, italic=True, color=(0x55, 0x55, 0x55), space_after=8)
    last_label = None
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            if cell.source.lstrip().startswith("# "):
                continue
            if "Nhận xét" in cell.source or not cell.source.lstrip().startswith("#"):
                md_to_paras(doc, cell.source)
            continue
        label = cell.metadata.get("cau", "")
        head = label.split(" - ")[0]
        if head != last_label:
            para(doc, f"Câu {head})" if is_cau else head, 13, True, space_after=4)
            last_label = head
        if " - " in label:
            para(doc, "– " + label.split(" - ", 1)[1], 12, italic=True, space_after=3)
        para(doc, "Mã nguồn:", 11, italic=True, color=(0x44, 0x44, 0x44), space_after=2)
        add_code(doc, cell.source)
        para(doc, "Kết quả:", 11, italic=True, color=(0x44, 0x44, 0x44), space_after=2)
        add_outputs(doc, cell)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

para(doc, "--- HẾT ---", 13, True, align=WD_ALIGN_PARAGRAPH.CENTER)
out = ROOT / f"{MSSV}_LAB3_BaoCao.docx"
doc.save(out)
print("saved", out)
