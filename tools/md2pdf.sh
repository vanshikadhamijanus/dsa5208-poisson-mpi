#!/bin/bash
# Markdown -> HTML (pandoc) -> PDF (headless Chrome).
# Usage: tools/md2pdf.sh input.md output.pdf [extra pandoc args...]
# Produces a clickable table of contents, page numbers, running title.
set -euo pipefail
IN=$1; OUT=$2; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
HTML="${OUT%.pdf}.html"
# pandoc >= 3.8 renamed --mathml/--highlight-style; pick whichever this pandoc understands
if pandoc --help 2>/dev/null | grep -q -- '--math-method'; then
  PANDOC_COMPAT=(--math-method=mathml --syntax-highlighting=monochrome)
else
  PANDOC_COMPAT=(--mathml --highlight-style=monochrome)
fi
pandoc "$IN" -s -o "$HTML" --toc --toc-depth=2 "${PANDOC_COMPAT[@]}" --embed-resources \
  --css "$HERE/doc.css" ${EXTRA_CSS:+--css "$EXTRA_CSS"} \
  --resource-path="$(dirname "$IN"):$(dirname "$IN")/..:$HERE/.." "$@"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
command -v google-chrome >/dev/null 2>&1 && CHROME=google-chrome
command -v chromium >/dev/null 2>&1 && CHROME=chromium
"$CHROME" --headless=new --disable-gpu --no-sandbox --run-all-compositor-stages-before-draw \
  --virtual-time-budget=10000 --no-pdf-header-footer --print-to-pdf="$OUT" "file://$(cd "$(dirname "$HTML")" && pwd)/$(basename "$HTML")" 2>/dev/null
rm -f "$HTML"
# page count: page objects minus the page-tree objects (no Python needed)
PAGES=$(( $(LC_ALL=C grep -ao '/Type /Page' "$OUT" | wc -l) - $(LC_ALL=C grep -ao '/Type /Pages' "$OUT" | wc -l) ))
echo "wrote $OUT ($PAGES pages)"
