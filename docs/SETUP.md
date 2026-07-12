# Setup

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 2. Data

Download and place the raw files per instructions in:
- `data/raw/onet/SOURCE.md`
- `data/raw/oews/SOURCE.md`

## 3. Run tests

```bash
pytest
```

## 4. Run the pipeline (once Phase 1 lands)

```bash
python -m src.data.onet_loader   # sanity-check O*NET loading
python -m src.data.oews_loader   # sanity-check OEWS loading
```

## 5. API / dashboard

Documented once Phases 4-5 land.
