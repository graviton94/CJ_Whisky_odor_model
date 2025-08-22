import numpy as np
import pandas as pd

from data_processor import prepare_mixture_data, get_note_weight
from ontology_manager import OntologyManager
from config import DRAVNIEKS_CSV, ODOR_DESCRIPTORS, TASTE_DESCRIPTORS


def predict_mixture(input_molecules, mode='odor'):
    """
    Predicts descriptor intensities for a mixture of molecules.

    Args:
        input_molecules: List of dicts with 'SMILES' and 'peak_area'.
        mode: 'odor' or 'taste'.
    Returns:
        Dict with keys 'raw', 'corrected', and 'rule_log'.
    """
    # Descriptor 선택
    descriptors = ODOR_DESCRIPTORS if mode == 'odor' else TASTE_DESCRIPTORS

    # 입력 데이터 준비
    smiles_list = [m['SMILES'] for m in input_molecules]
    peak_areas = np.array([m['peak_area'] for m in input_molecules], dtype=np.float32)
    weights = np.array([get_note_weight(smi) for smi in smiles_list], dtype=np.float32)
    weighted_areas = peak_areas * weights
    peak_ratios = weighted_areas / (np.sum(weighted_areas) + 1e-8)

    # 혼합물 intensity matrix 및 보정 특성 불러오기
    intensity_matrix, descs, extra_feats, extra_names, detectabilities = prepare_mixture_data(
        smiles_list, peak_areas, mode=mode, csv_path=str(DRAVNIEKS_CSV)
    )

    # Raw 예측
    pred_scores = np.sum(intensity_matrix * peak_ratios[:, None], axis=0)
    raw_dict = {d: float(s) for d, s in zip(descs, pred_scores)}

    # 최신 OntologyManager 기반 보정
    ontology_manager = OntologyManager()
    concentrations = [m['peak_area'] for m in input_molecules]
    
    rule_result = ontology_manager.apply_rules(
        result_dict=raw_dict.copy(),
        smiles_list=smiles_list,
        descriptor_type=mode,
        extra_features=extra_feats,
        detectability=detectabilities,
        concentrations=concentrations
    )
    
    corrected_scores = rule_result['corrected_scores']
    # corrected_scores가 리스트 형태일 수 있어 처리
    if isinstance(corrected_scores, dict):
        corrected_dict = {d: float(corrected_scores[d]) for d in descs}
    else:
        corrected_dict = {d: float(v) for d, v in zip(descs, corrected_scores)}
    rule_log = rule_result.get('rule_log', [])

    return {
        'raw': raw_dict,
        'corrected': corrected_dict,
        'rule_log': rule_log,
    }


if __name__ == "__main__":
    # 예시 실행
    input_molecules = [
        {"SMILES": "CCCCCCCC(=O)OCC(C)C", "peak_area": 803959.33},
        {"SMILES": "CN(C)C(=O)OC1=CC(C)=C(C)C=C1", "peak_area": 14293392.4}
    ]
    result = predict_mixture(input_molecules, mode='odor')
    print(result)
