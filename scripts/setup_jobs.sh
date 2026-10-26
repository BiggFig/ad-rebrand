#!/bin/zsh
# Create one job folder per source video inside a project folder and print a probe line for each.
# Usage: setup_jobs.sh PROJECT_DIR START_NUMBER VIDEO [VIDEO ...]
# Makes PROJECT_DIR/job-NN/{assets,analysis,project,renders} with assets/source.mp4 (a copy; sources are never edited).
set -e
export PATH=/opt/homebrew/bin:$PATH
P="$1"; n="$2"; shift 2
mkdir -p "$P"
for f in "$@"; do
  J="$P/job-$(printf %02d $n)"
  mkdir -p "$J"/{assets,analysis,project,renders}
  cp "$f" "$J/assets/source.mp4"
  echo "$(basename "$J")  $(basename "$f")  $(ffprobe -v error -show_entries format=duration:stream=codec_type,width,height,r_frame_rate -of csv=p=0 "$J/assets/source.mp4" | tr '\n' ' ')"
  n=$((n+1))
done
