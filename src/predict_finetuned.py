import torch
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from data_processor import build_input_vector
from config import (
    ODOR_FINETUNE_PTH,
    TASTE_FINETUNE_PTH,
    MODEL_META_ODOR,
    MODEL_META_TASTE
)


class FinetuneMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(mode: str):
    """
    mode: 'odor' or 'taste'
    Returns: (model, descriptor_list)
    """
    model_path = Path(ODOR_FINETUNE_PTH if mode == 'odor' else TASTE_FINETUNE_PTH)
    meta_path = Path(MODEL_META_ODOR if mode == 'odor' else MODEL_META_TASTE)

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model or meta file not found for mode '{mode}'")

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    input_dim = meta['input_dim']
    output_dim = meta['output_dim']
    desc_list = meta.get('y_cols', [])

    model = FinetuneMLP(input_dim, output_dim)
    try:
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load {mode} model: {str(e)}")
    return model, desc_list


def predict_finetuned(input_molecules: list, mode: str = 'odor') -> dict:
    """
    Predict descriptor intensities using fine-tuned MLP models.
    Applies dynamic MinMax scaling to [0,10] range and clamps.

    input_molecules: list of dicts with 'SMILES' and 'peak_area'.
    mode: 'odor' or 'taste'

    Returns: {'predicted': {descriptor: value, ...}}
    """
    assert mode in ['odor', 'taste'], "mode must be 'odor' or 'taste'"

    # Prepare feature vector
    smiles_list = [m['SMILES'] for m in input_molecules]
    peak_areas = np.array([m['peak_area'] for m in input_molecules], dtype=np.float32)
    feat_vec = build_input_vector(smiles_list, peak_areas, mode=mode)
    
    # Load model and check dimensions
    model, desc_list = load_model(mode)
    
    # 차원 불일치 해결 - 저장된 모델 차원에 맞춤
    expected_input_dim = model.net[0].in_features
    current_dim = feat_vec.shape[0]
    
    if current_dim != expected_input_dim:
        if current_dim < expected_input_dim:
            # 부족한 차원은 0으로 패딩
            padding = np.zeros(expected_input_dim - current_dim)
            feat_vec = np.concatenate([feat_vec, padding])
        else:
            # 초과 차원은 잘라내기
            feat_vec = feat_vec[:expected_input_dim]

    # Model inference
    x = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        raw_pred = model(x).squeeze(0).cpu().numpy()

    # Dynamic MinMax scaling to [0,10]
    scaler = MinMaxScaler(feature_range=(0, 10))
    norm_pred = scaler.fit_transform(raw_pred.reshape(-1, 1)).flatten()

    # Clamp to [0,10]
    final_pred = np.clip(norm_pred, 0.0, 10.0)

    return {
        'predicted': {
            desc: float(score)
            for desc, score in zip(desc_list, final_pred)
        }
    }


if __name__ == '__main__':
    # 예시 실행
    input_mixture = [
        {"SMILES": "CCCCCCCC(=O)OCC(C)C", "peak_area": 803959.33},
        {"SMILES": "CN(C)C(=O)OC1=CC(C)=C(C)C=C1", "peak_area": 14293392.4}
    ]
    result = predict_finetuned(input_mixture, mode='odor')
    print(json.dumps(result['predicted'], indent=2))