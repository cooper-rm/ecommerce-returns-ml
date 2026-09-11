"""Render the memo to PDF.

Edit `returns_ml_memo.html` in any text editor, then run:

    python memo/build_memo.py

Works offline. The only dependency is Chrome (already on this machine); no Python
packages beyond the standard library are used, so it runs even outside the conda env.
"""
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "returns_ml_memo.html"
OUTPUT = HERE / "returns_ml_memo.pdf"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def find_chrome():
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def main():
    if not SOURCE.exists():
        sys.exit(f"missing source: {SOURCE}")

    chrome = find_chrome()
    if chrome is None:
        sys.exit("No Chrome/Chromium/Edge found. Open the HTML in a browser and use Print > Save as PDF.")

    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=30000", f"--print-to-pdf={OUTPUT}", SOURCE.resolve().as_uri()],
        check=True, capture_output=True,
    )
    print(f"wrote {OUTPUT}  ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
