# Requires ffmpeg installed and on PATH
param(
  [string]$source = "data/sample_videos/match.mp4",
  [string]$out = "images/preview.gif",
  [int]$start = 0,
  [int]$duration = 6,
  [int]$width = 640
)

if (-not (Test-Path $source)) {
  Write-Error "Source video not found: $source"
  exit 1
}

$tmp = "tmp_frames_%03d.png"
ffmpeg -ss $start -t $duration -i $source -vf "fps=12,scale=$width:-1:flags=lanczos" -y $tmp
ffmpeg -f image2 -i tmp_frames_%03d.png -vf "palettegen" -y palette.png
ffmpeg -f image2 -i tmp_frames_%03d.png -i palette.png -lavfi "paletteuse" -y $out

Remove-Item tmp_frames_*.png -ErrorAction SilentlyContinue
Remove-Item palette.png -ErrorAction SilentlyContinue
Write-Host "GIF generated: $out"
