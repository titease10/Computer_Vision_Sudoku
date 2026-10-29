"""Exporte la note de cadrage (docs/00_note_de_cadrage.md) en PDF A4 d'une page.

Markdown -> HTML (avec une mise en page sobre) -> PDF via Chrome sans interface.

Usage :
    python scripts/export_note_pdf.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/00_note_de_cadrage.md"
OUT = ROOT / "docs/note_de_cadrage.pdf"
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")

CSS = """
@page { size: A4; margin: 13mm 15mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.6pt; line-height: 1.32; color: #1a1a1a; }
h1 { font-size: 15pt; margin: 0 0 6px; padding-bottom: 4px; border-bottom: 2px solid #2b5797; color: #2b5797; }
h2 { font-size: 11pt; margin: 9px 0 3px; color: #2b5797; }
p { margin: 3px 0; }
ul, ol { margin: 2px 0; padding-left: 20px; }
li { margin: 1px 0; }
table { border-collapse: collapse; width: 100%; margin: 3px 0; }
th, td { border: 1px solid #c8c8c8; padding: 2px 6px; text-align: left; vertical-align: top; }
th { background: #eef2f8; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 2px; }
"""


def main() -> None:
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables"])
    html = OUT.with_suffix(".html")
    html.write_text(f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><style>{CSS}</style>'
                    f"</head><body>{body}</body></html>", encoding="utf-8")
    subprocess.run([str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT}", html.as_uri()], check=True, capture_output=True)
    html.unlink()
    print(f"PDF écrit : {OUT}")


if __name__ == "__main__":
    main()
