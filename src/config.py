from pathlib import Path

# ────────────── 프로젝트 기본 경로 ──────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent  # 루트 디렉토리로 변경
DATA_DIR = BASE_DIR / 'data'

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

# ────────────── Descriptor 리스트 ──────────────
ODOR_DESCRIPTORS = [
    'Fragrant','Woody','Fruity','Citrus','Sweet','Floral',
    'Spicy','Minty','Green','Earthy','Vanilla','Almond'
]
TASTE_DESCRIPTORS = [
    'Taste_Bitter','Taste_Floral','Taste_Fruity','Taste_OffFlavor',
    'Taste_Nutty','Taste_Sour','Taste_Sweet'
]

# ────────────── 하이브리드 예측 기본값 ──────────────
HYBRID_DEFAULTS = {
    'method': 'weighted',
    'weight_finetune': 0.5,
    'max_molecules': 32
}

# ────────────── RDKit / Fingerprint 설정 ──────────────
FP_DIM = 2048

# ────────────── Ontology 룰 인덱스 ──────────────
FG_NAMES = ['Ester','Alcohol','Ketone','Aldehyde','CarboxylicAcid','Ether']
VOLATILITY_IDX = -2
TANIMOTO_IDX = -1

# ────────────── 파인튜닝 학습 설정 ──────────────
TF_BATCH_SIZE = 16
TF_EPOCHS = 200
TF_LR = 1e-3

# ────────────── Streamlit / OpenAI 설정 ──────────────
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # 환경변수에서 로드 권장
# RADAR_CHART 기본 설정
RADAR_MAX_INTENSITY = 10
RADAR_TICKS = [0, 2.5, 5, 7.5, RADAR_MAX_INTENSITY]
