from pathlib import Path

# ────────────── 프로젝트 기본 경로 ──────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # src 디렉토리
MODEL_DIR = BASE_DIR / 'saved_models'
DATA_DIR = BASE_DIR / 'data'  # src/data 디렉토리

# ────────────── 데이터 파일 경로 ──────────────
DRAVNIEKS_CSV = DATA_DIR / 'dravnieks_whisky_dataset_final.csv'
TRIAL_CSV = DATA_DIR / 'mixture_trials_log.csv'
TRIAL_LONG_CSV = DATA_DIR / 'mixture_trials_long.csv'
LEARN_JSONL = DATA_DIR / 'mixture_trials_learn.jsonl'

# ────────────── 모델 파일 경로 ──────────────
ODOR_FINETUNE_PTH = MODEL_DIR / 'odor_finetune.pth'
TASTE_FINETUNE_PTH = MODEL_DIR / 'taste_finetune.pth'
MODEL_META_ODOR = MODEL_DIR / 'model_meta_odor.json'
MODEL_META_TASTE = MODEL_DIR / 'model_meta_taste.json'

# ────────────── 특성 추출 관련 상수 ──────────────
FP_DIM = 2048  # Morgan fingerprint dimension

# ────────────── 설명자 목록 ──────────────
ODOR_DESCRIPTORS = [
    'Fragrant', 'Woody', 'Fruity', 'Citrus', 'Sweet',
    'Floral', 'Spicy', 'Minty', 'Green', 'Earthy',
    'Vanilla', 'Almond'
]

TASTE_DESCRIPTORS = [
    'Taste_Sweet', 'Taste_Bitter', 'Taste_Fruity',
    'Taste_Floral', 'Taste_Sour', 'Taste_OffFlavor',
    'Taste_Nutty'
]
