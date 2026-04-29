"""Split prepared dataset into train/val/test folders preserving class subfolders.

Usage:
    python scripts/split_dataset.py --src data/brain_tumor_dataset --dest data/splits --train 0.8 --val 0.1 --test 0.1
"""
import argparse
from pathlib import Path
import random
import shutil


def split_folder(src: Path, dest: Path, train=0.8, val=0.1, test=0.1, seed=42):
    assert abs(train + val + test - 1.0) < 1e-6
    random.seed(seed)
    dest.mkdir(parents=True, exist_ok=True)
    for cls_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(train * n)
        n_val = int(val * n)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            out_dir = dest / split_name / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in split_imgs:
                shutil.copy2(p, out_dir / p.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, default=Path('archive (1)') / 'Training')
    parser.add_argument('--dest', type=Path, default=Path('data') / 'splits')
    parser.add_argument('--train', type=float, default=0.8)
    parser.add_argument('--val', type=float, default=0.1)
    parser.add_argument('--test', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not args.src.exists():
        print('Source dataset not found at', args.src)
        return
    split_folder(args.src, args.dest, args.train, args.val, args.test, args.seed)
    print('Split complete. Files placed under', args.dest)


if __name__ == '__main__':
    main()
