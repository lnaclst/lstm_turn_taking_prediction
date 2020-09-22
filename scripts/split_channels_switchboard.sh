#!/bin/sh
# script written by Elliott Gruzin

mkdir ./dialogues_mono
for i in `cat swb_complete.txt`; do
	echo "VAR: $i"
	sox swb1/$i.sph -b 16 -c 1 dialogues_mono/$i.g.wav remix 1
	sox swb1/$i.sph -b 16 -c 1 dialogues_mono/$i.f.wav remix 2

done