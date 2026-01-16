# 72hr Shckthon — Delivery Scheduling & Optimization 🧩

**Overview:**
This repository contains tools and experiments for a Kaggle-style delivery scheduling and optimization task (farm deliveries → STPs). It includes data loaders, precomputation utilities, optimization strategies, scoring, and verification tools to create, evaluate, and export submission files.

---

## 📁 Project Structure

```
model/
├── main.py           # Primary entry point, orchestrates pipeline
├── main_v*.py        # Variant experiment entry points (v2..v12)
├── config.py         # Loads constants from data/config.json
├── data_loader.py    # Data loading & validation helpers
├── precompute.py     # Precompute distances / travel times / factors
├── optimizer.py      # Core optimization / scheduling algorithms
├── scoring.py        # Scoring & validation logic
├── export.py         # Export/formatting utilities for submissions
├── verify_solution.py# Additional solution checks
├── data/             # Input CSV datasets (sample + training)
└── output/           # Generated outputs (e.g., solution.csv)
```

Key data in `data/`:
- `config.json`, `farm_locations.csv`, `stp_registry.csv`, `daily_n_demand.csv`, `daily_weather_2025.csv`, `planting_schedule_2025.csv`, `sample_submission.csv`.

---

## 🚀 Quick Start

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Run precomputation (optional):
```bash
python precompute.py
```

3. Run the main pipeline (choose an experiment file):
```bash
python main.py        # or try: python main_v9.py
```

4. Validate or score a solution:
```bash
python verify_solution.py
python scoring.py     # provides utilities used programmatically
```

Outputs will be written to `output/` (e.g., `output/solution.csv`).

---

## ⚙️ Usage Options & Examples

Examples of how to run several variants (see CLI flags in each `main_*.py`):

```bash
# Default run
python main.py

# Run an experiment variant
python main_v9.py

# Validate an existing output
python verify_solution.py --solution output/solution.csv
```

---

## 🔧 Dependencies

Minimum:
- Python 3.8+
- pandas
- numpy

Install all required packages with `pip install -r requirements.txt`.

Optional but useful for modeling/experiments: `scikit-learn`, `xgboost`, `lightgbm`.

---

## 📝 How to push this repository to GitHub (add everything)

From the repository root, run:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/CHRIS5992/72hrshckthon.git
# Push — you may be prompted for username/password or a PAT
git push -u origin main
```

If your account uses 2FA, create a GitHub Personal Access Token and use it in place of your password.

If your dataset is large, consider using Git LFS: https://git-lfs.github.com/

---

## ✅ Contributing

- Open an issue to discuss ideas or bugs
- Send a PR with tests and a clear description

---

## 📄 License

MIT License — see `LICENSE` (or add one) if you want to formalize licensing.

---

If you'd like, I can initialize a local Git repository and make an initial commit here, and then attempt to push to `https://github.com/CHRIS5992/72hrshckthon.git` (push may require you to authenticate). Let me know if you'd like me to proceed.

