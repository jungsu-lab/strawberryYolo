# Publish to GitHub

Use the clean package directory, not the larger local preprocessing workspace.

```bash
cd strawberry_yolo_kgcv_github
```

## 1. Inspect the Package

```bash
find . -maxdepth 3 -type f | sort
```

Confirm that raw data, processed data, model weights, runs, virtual environments, and caches are not present.

```bash
test ! -e raw
test ! -e processed
test ! -e runs
test ! -e .venv
test ! -e .pytest_cache
```

## 2. Run Tests

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider \
  tests/test_convert_aihub_strawberry_to_yolo.py \
  tests/test_yolo_local_scripts.py \
  tests/test_yolo_growth_pipeline_tables.py \
  tests/test_build_yolo_github_package.py \
  tests/test_build_yolo_ablation_plan.py \
  -q
```

Expected:

```text
13 passed
```

## 3. Initialize Git

```bash
git init
git add .
git commit -m "feat: add strawberry YOLO KGCV pipeline"
```

## 4. Create a GitHub Repository

Create an empty GitHub repository, then connect it:

```bash
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

## 5. After Push

Check GitHub Actions:

- `YOLO KGCV Pipeline Tests` should run.
- The test job should pass before sharing the repository.

## 6. Do Not Commit

Do not commit:

- raw image files
- raw JSON labels
- converted YOLO datasets
- `.pt`, `.pth`, `.onnx`, or other model weights
- `runs/`
- `.venv/`
- `.pytest_cache/`
- `__pycache__/`
