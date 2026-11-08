#!/bin/bash
# Draw the maps for one day: observed, 2D fill, 3D fill, difference.
#   scripts/local/maps.sh GRID K FILL2D.grid FILL3D.grid OUT.png
# GRID is the input grid file, K the day index (0-based) of the 2D fill, FILL2D.grid
# the output of a 2D run with --day K --out, FILL3D.grid the output of the 3D run.
set -eu
cd "$(dirname "$0")/../.."
GRID=$1; K=$2; F2=$3; F3=$4; OUT=$5
TMP=$(mktemp -d)
bin/gridtool slice "$GRID" "$K" "$TMP/o.txt"  >/dev/null
bin/gridtool slice "$F2" 0 "$TMP/f2.txt"      >/dev/null
bin/gridtool slice "$F3" "$K" "$TMP/f3.txt"   >/dev/null
bin/gridtool diff  "$F3" "$K" "$F2" 0 "$TMP/d.txt" | sed "s/^/difference: /"
# blank the observed pixels in the difference map so only the filled ones show
awk 'NR==FNR { if (FNR>1) obs[FNR]=$0; next } FNR==1 { print; next } { n=split(obs[FNR],o," "); for (i=1;i<=NF;i++) if (o[i]!="NaN") $i="NaN"; print }' "$TMP/o.txt" "$TMP/d.txt" > "$TMP/d2.txt"
read LON0 DLON LAT0 DLAT <<< "$(head -1 "$TMP/o.txt" | awk '{print $3, $5, $7, $9}')"
DATE=$(bin/gridtool info "$GRID" --days | awk -v k="$K" 'NR==3+k {print $1}')
gnuplot -e "nx=$(awk "NR==2{print NF}" "$TMP/o.txt"); ny=$(($(wc -l < "$TMP/o.txt")-1)); obs='$TMP/o.txt'; f2='$TMP/f2.txt'; f3='$TMP/f3.txt'; d32='$TMP/d2.txt'; out='$OUT'; lon0=$LON0; dlon=$DLON; lat0=$LAT0; dlat=$DLAT; date='$DATE'" analysis/maps.gp 2>&1 | grep -v 'missing or undefined' || true
rm -rf "$TMP"
echo "wrote $OUT"
