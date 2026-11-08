#!/bin/bash
# Build report.pdf from report.md (pandoc -> HTML -> headless Chrome).
set -euo pipefail
cd "$(dirname "$0")"
EXTRA_CSS=../tools/report.css ../tools/md2pdf.sh report.md report.pdf --number-sections
