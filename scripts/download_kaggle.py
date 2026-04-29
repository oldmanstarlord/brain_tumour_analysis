"""Download the Kaggle Brain Tumor MRI dataset to `data/` using the kaggle CLI.

Usage:
    python scripts/download_kaggle.py

Ensure you have the kaggle CLI configured (place `kaggle.json` in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY).
"""
import os
import subprocess
import sys

DATASET_SLUG = 'masoudnickparvar/brain-tumor-mri-dataset'
DEST = 'data'


def main():
    cmd = ['kaggle', 'datasets', 'download', '-d', DATASET_SLUG, '-p', DEST, '--unzip']
    print('Running:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
        print('Download complete. Files are in', os.path.abspath(DEST))
    except subprocess.CalledProcessError as e:
        print('kaggle CLI failed:', e)
        print('Make sure kaggle is installed and configured: https://github.com/Kaggle/kaggle-api')
        sys.exit(1)


if __name__ == '__main__':
    main()
