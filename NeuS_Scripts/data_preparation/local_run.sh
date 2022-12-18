#!/bin/bash
rm -r images
rm images.tgz
mkdir -p images
ffmpeg -i ${1} images/temp_frame_%05d.png
python3 removeColor.py
python3 renameColor.py
tar cvzf images.tgz images 
