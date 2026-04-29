"""Prepare dataset layout under `data/brain_tumor_dataset/`.

This script copies image files found under `data/` into a canonical layout:
    data/brain_tumor_dataset/<class_name>/*.jpg

It infers class name from the immediate parent folder of each image.
"""
import os
from pathlib import Path
import shutil


SRC = Path('data')
DEST = Path('data') / 'brain_tumor_dataset'


def prepare():
    if not SRC.exists():
        print('Source data directory not found:', SRC)
        return
    DEST.mkdir(parents=True, exist_ok=True)
    images = list(SRC.rglob('*.jpg')) + list(SRC.rglob('*.png')) + list(SRC.rglob('*.jpeg'))
    print(f'Found {len(images)} image files under {SRC}')
    moved = 0
    for img in images:
        # skip files already in DEST
        if DEST in img.parents:
            continue
        parent = img.parent.name or 'unknown'
        out_dir = DEST / parent
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_file = out_dir / img.name
        try:
            shutil.copy2(img, dest_file)
            moved += 1
        except Exception as e:
            print('Failed to copy', img, e)
    print(f'Copied {moved} files into {DEST}')


if __name__ == '__main__':
    prepare()
