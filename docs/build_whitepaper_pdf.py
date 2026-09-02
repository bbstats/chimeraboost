"""Render docs/whitepaper.md to docs/whitepaper.pdf.

Chain: `markdown` (already a docs dependency via mkdocs) -> styled HTML ->
Microsoft Edge headless --print-to-pdf. No new dependencies; Edge ships with
Windows. Remote chart URLs are rewritten to the committed files in images/ so
the build needs no network.

Usage: python docs/build_whitepaper_pdf.py
"""

import pathlib
import subprocess
import sys
import tempfile

import markdown

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / "docs" / "whitepaper.md"
OUTPUT = REPO / "docs" / "whitepaper.pdf"
RAW_IMAGES = "https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/"

EDGE_CANDIDATES = [
    pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    pathlib.Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
@page { size: Letter; margin: 22mm 20mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.55; color: #1a1a1a; max-width: 17cm; margin: 0 auto; }
h1, h2, h3 { font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
             color: #2a2040; line-height: 1.25; }
h1 { font-size: 21pt; margin-bottom: 0.2em; }
h2 { font-size: 14pt; margin-top: 1.6em; border-bottom: 1px solid #d5cfe6;
     padding-bottom: 0.15em; }
h3 { font-size: 11.5pt; margin-top: 1.3em; }
em { color: #444; }
a { color: #4527a0; text-decoration: none; }
table { border-collapse: collapse; margin: 1em auto; font-size: 9.5pt;
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif; }
th, td { border: 1px solid #c9c3d9; padding: 4px 10px; text-align: left; }
th { background: #efecf6; }
pre { background: #f5f4f8; border: 1px solid #e0dcec; border-radius: 4px;
      padding: 9px 12px; font-size: 8.8pt; overflow-x: hidden;
      white-space: pre-wrap; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 0.92em; }
img { max-width: 100%; display: block; margin: 1em auto; }
li { margin: 0.25em 0; }
h2, h3 { page-break-after: avoid; }
table, pre, img { page-break-inside: avoid; }
"""


def main() -> int:
    edge = next((p for p in EDGE_CANDIDATES if p.exists()), None)
    if edge is None:
        print("Edge not found; cannot render the PDF.", file=sys.stderr)
        return 1

    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace(RAW_IMAGES, (REPO / "images").as_uri() + "/")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>ChimeraBoost whitepaper</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )

    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / "whitepaper.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                str(edge),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={OUTPUT}",
                page.as_uri(),
            ],
            check=True,
            timeout=120,
        )

    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
