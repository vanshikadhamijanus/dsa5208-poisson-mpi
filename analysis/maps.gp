# maps.gp -- draw the satellite maps: observed day, 2D fill, 3D fill, 3D minus 2D.
# Usage (see scripts/local/maps.sh):
#   gnuplot -e "obs='o.txt'; f2='f2.txt'; f3='f3.txt'; d32='d.txt'; out='maps.png'; lon0=65.00625; dlon=0.0375; lat0=5.00625; dlat=0.0375; date='2025-04-10'" analysis/maps.gp
set terminal pngcairo size 1500,1000 enhanced font "Helvetica,11"
set output out
set multiplot layout 2,2
set view map; set size ratio -1
set palette defined (0 "#313695", 0.25 "#74add1", 0.5 "#ffffbf", 0.75 "#f46d43", 1 "#a50026")
set cbrange [24:32]; set cblabel "sea-surface temperature (deg C)"
set xlabel "longitude (deg E)"; set ylabel "latitude (deg N)"
X(c) = lon0 + c*dlon
Y(r) = lat0 + r*dlat
set xrange [lon0-dlon/2:lon0+(nx-0.5)*dlon]; set yrange [lat0-dlat/2:lat0+(ny-0.5)*dlat]
set title "Observed by VIIRS on ".date." (white: clouds, land)"
plot obs matrix u (X($1)):(Y($2)):3 with image notitle
set title "Filled in 2D: this day only (Laplace, space neighbours)"
plot f2  matrix u (X($1)):(Y($2)):3 with image notitle
set title "Filled in 3D: the whole year, time as the third axis"
plot f3  matrix u (X($1)):(Y($2)):3 with image notitle
set palette defined (0 "#2166ac", 0.5 "#f7f7f7", 1 "#b2182b")
set cbrange [-2:2]; set cblabel "3D fill minus 2D fill (deg C)"
set title "Where the two fills disagree (only in the filled pixels)"
plot d32 matrix u (X($1)):(Y($2)):3 with image notitle
unset multiplot
