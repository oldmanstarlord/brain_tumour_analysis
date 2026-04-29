# Brain Tumor Analyzer

Lightweight two-stage pipeline: EfficientNet-B0 classifier (4 classes) + U-Net segmentation (binary tumor mask) with Grad-CAM explainability.

Phase 0: Environment & repo setup — included files and basic instructions.

Phase 1 note

- The downloaded dataset is expected under `archive (1)/Training` and `archive (1)/Testing` in this workspace.
- Use `Training` for EDA and train/validation setup, and `Testing` for held-out evaluation.

Quick start

- Create and activate a virtual environment:

  PowerShell (Windows):

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

  macOS / Linux:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- Kaggle dataset (set up `kaggle` CLI first):

  ```bash
  kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/ --unzip
  ```

- Check GPU availability:

  ```bash
  python scripts/check_gpu.py
  ```

- Run the notebook:

  ```bash
  jupyter notebook tumor_analysis.ipynb
  ```

- Run the Streamlit demo (after training or placing model weights in `models/`):

  ```bash
  streamlit run app.py
  ```

Files added in Phase 0

- `requirements.txt` — Python dependencies
- `README.md` — this quick start
- `.gitignore` — common ignores
- `scripts/check_gpu.py` — small GPU check helper
- `.github/workflows/ci.yml` — minimal CI smoke workflow

Notes

- The notebook `tumor_analysis.ipynb` and `app.py` are scaffolds — later phases will fill training and evaluation code and add tests/CI.
- Place pretrained model weights in `models/` as `efficientnet_classifier.pth` and `unet_segmentation.pth` for the Streamlit app to load.
