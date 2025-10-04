\# Schema v1 — Water Polo xG



\## Canonical fields (aliases \& derived — non-destructive)

\- x ← shooter\_x (float, m)          # lateral from goal center (+R/−L)

\- y ← shooter\_y (float, m)          # distance from goal line (toward midfield)

\- outcome ← shot\_result (str)       # goal | saved | miss | block | turnover

\- handed ← shooter\_handedness (str) # R | L | U

\- distance\_m = sqrt(x^2 + y^2)

\- angle\_deg\_signed = atan2(x, y) \* 180/π   # −90..+90



\## Optional video linkage

\- source\_video\_id (str)     # must match data/videos.csv

\- video\_timecode (float s)  # seconds within file

(Keep legacy: video\_file, video\_timestamp\_mmss)



\## Notes

\- Keep ALL original columns. Canonical fields are added alongside.

\- Angle bins use −90..−60..−30..0..30..60..90; distance bins 0..3..6..9..12..15..21.



