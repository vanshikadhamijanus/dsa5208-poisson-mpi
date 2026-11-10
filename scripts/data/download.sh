#!/bin/bash
# Download daily satellite sea-surface temperature for a region from NOAA CoastWatch's
# ERDDAP server (no account needed) as plain-text grids, one file per day.
#
#   scripts/data/download.sh REGION START [END] [ref]
#
#   REGION  india      box  5N..25N,  65E..95E   (Arabian Sea, Bay of Bengal, India in the middle)
#           singapore  box  8S..8N,   96E..120E  (Malacca Strait, South China Sea, Java Sea)
#           or a custom box "S,N,W,E" in degrees, e.g. "0,10,100,110"
#   START   a date YYYY-MM-DD, or a year YYYY for the whole year
#   END     last date (default: START, or 31 December of the year)
#   ref     also download the gap-free blended analysis for the same days (reference)
#
# Output: data/raw/REGION/sst_YYYY-MM-DD.asc (VIIRS, 4 km, with cloud gaps),
#         data/raw/REGION/mask.asc            (land mask of the blended product, 5 km),
#         data/raw/REGION/ref_YYYY-MM-DD.asc  (blended analysis, if "ref" was given).
# Each daily file is about 1 MB for the India box; a full year is about 400 MB and
# takes 15-20 minutes. Existing files are skipped, so the script can be re-run.
# Then build the grid file with bin/gridtool (see README).
set -u
cd "$(dirname "$0")/../.."
REGION=${1:-india}; START=${2:-2025}; END=${3:-}; REF=${4:-}
[ "$END" = ref ] && { REF=ref; END=; }
case $REGION in
  india)     S=5.0;  N=25.0; W=65.0; E=95.0 ;;
  singapore) S=-8.0; N=8.0;  W=96.0; E=120.0 ;;
  *) IFS=, read S N W E <<< "$REGION" ;;
esac
if [ ${#START} -eq 4 ]; then END=${END:-$START-12-31}; START=$START-01-01; fi
END=${END:-$START}
OUT=data/raw/$REGION; mkdir -p "$OUT"
BASE=https://coastwatch.noaa.gov/erddap/griddap
SST=noaacwN20ACSPOSSTL3GCDaily       # NOAA-20 VIIRS ACSPO L3 collated, daily, 0.0375 deg (about 4 km)
BLEND=noaacwBLENDEDsstDaily          # NOAA geo-polar blended analysis, daily, 0.05 deg, gap-free

next_day() { date -j -v+1d -f %Y-%m-%d "$1" +%Y-%m-%d 2>/dev/null || date -d "$1 + 1 day" +%Y-%m-%d; }
fetch() { # url file
  local url=$1 file=$2 try
  [ -s "$file" ] && return 0
  for try in 1 2 3; do
    if curl -g -sS -f --connect-timeout 30 --max-time 300 -o "$file.part" "$url"; then mv "$file.part" "$file"; return 0; fi
    sleep $((try * 10))
  done
  rm -f "$file.part"; echo "FAILED: $file" >&2; return 1
}

echo "region $REGION (lat $S..$N, lon $W..$E), $START .. $END -> $OUT"
fetch "$BASE/$BLEND.asc?mask[(${START}T12:00:00Z)][($S):($N)][($W):($E)]" "$OUT/mask.asc"
d=$START; n=0; fail=0
while [ "$d" \< "$END" ] || [ "$d" = "$END" ]; do
  fetch "$BASE/$SST.asc?sea_surface_temperature[(${d}T12:00:00Z)][(0.0)][($S):($N)][($W):($E)]" "$OUT/sst_$d.asc" || fail=$((fail+1))
  if [ -n "$REF" ]; then fetch "$BASE/$BLEND.asc?analysed_sst[(${d}T12:00:00Z)][($S):($N)][($W):($E)]" "$OUT/ref_$d.asc" || fail=$((fail+1)); fi
  n=$((n+1)); [ $((n % 20)) -eq 0 ] && echo "  $n days done ($d)"
  d=$(next_day "$d")
done
echo "done: $n days, $fail failures. Files in $OUT ($(du -sh "$OUT" | cut -f1))."
[ $fail -eq 0 ] || echo "re-run the same command to retry the failed days."
