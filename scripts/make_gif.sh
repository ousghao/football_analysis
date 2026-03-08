#!/usr/bin/env bash
# Requires ffmpeg installed
SOURCE="data/sample_videos/match.mp4"
OUT="images/preview.gif"
START=0
DURATION=6
WIDTH=640

if [ ! -f "$SOURCE" ]; then
  echo "Source video not found: $SOURCE"
  exit 1
fi

ffmpeg -ss $START -t $DURATION -i "$SOURCE" -vf "fps=12,scale=$WIDTH:-1:flags=lanczos" -y tmp_frames_%03d.png
ffmpeg -f image2 -i tmp_frames_%03d.png -vf palettegen -y palette.png
ffmpeg -f image2 -i tmp_frames_%03d.png -i palette.png -lavfi "paletteuse" -y "$OUT"
rm tmp_frames_*.png palette.png
echo "GIF generated: $OUT"
