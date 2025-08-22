import numpy as np
import pandas as pd
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

from config import (
    DRAVNIEKS_CSV,
    HYBRID_DEFAULTS,
    ODOR_DESCRIPTORS,
    TASTE_DESCRIPTORS,
    MODEL_DIR
)
from predict import predict_mixture
from predict_finetuned import predict_finetuned
from data_processor import prepare_mixture_data
from features.odor_specific_features import OdorSpecificFeatureExtractor

# 1. CSV 기반 분자정보 DB 초기화
try:
    df_base = pd.read_csv(str(DRAVNIEKS_CSV), encoding='utf-8-sig')
except UnicodeDecodeError:
    df_base = pd.read_csv(str(DRAVNIEKS_CSV), encoding='cp949')
df_base = df_base.drop_duplicates('SMILES').set_index('SMILES')

# 2. Load hybrid parameters (optimized by Optuna)
param_odor_path = Path(MODEL_DIR) / 'hybrid_best_odor.json'
param_taste_path = Path(MODEL_DIR) / 'hybrid_best_taste.json'
if param_odor_path.exists() and param_taste_path.exists():
    with open(param_odor_path, 'r', encoding='utf-8') as f:
        params_odor = json.load(f)
    with open(param_taste_path, 'r', encoding='utf-8') as f:
        params_taste = json.load(f)
else:
    params_odor = {}
    params_taste = {}
hybrid_params = {'odor': params_odor, 'taste': params_taste}

# 3. RDKit 분자 특성 계산
def calc_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"MolWt": None, "LogP": None, "TPSA": None}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol)
    }

# 4. 기여도 계산
def calc_contributions(input_mixture, intensity_matrix, descs):
    peak_areas = np.array([m['peak_area'] for m in input_mixture], dtype=np.float32)
    # 1) 로그 변환
    log_areas = np.log1p(peak_areas)
    # 2) 정규화
    ratios = log_areas / (log_areas.sum() + 1e-8)

    contribs = {desc: [] for desc in descs}
    for j, desc in enumerate(descs):
        for i, smi in enumerate([m['SMILES'] for m in input_mixture]):
            contribs[desc].append({
                "molecule": smi,
                "contribution": float(intensity_matrix[i, j] * ratios[i])
            })
        contribs[desc] = sorted(contribs[desc], key=lambda x: -x['contribution'])[:3]
    return contribs

# 5. description_support 생성
def build_description_support(contribs, mode="odor"):
    primary_key = 'Primary_Odor' if mode == 'odor' else 'Primary_Taste'
    support = {}
    for desc, mols in contribs.items():
        mol_info_list = []
        for m in mols:
            smiles = m['molecule']
            name = df_base.loc[smiles, 'Molecule_Name'] if smiles in df_base.index else smiles
            primary = df_base.loc[smiles, primary_key] if smiles in df_base.index else ""
            props = calc_props(smiles)
            if any(v is None for v in props.values()):
                props = {"MolWt": 180.0, "LogP": 2.0, "TPSA": 50.0}
            mol_info_list.append({
                "name": name,
                "smiles": smiles,
                "primary": primary,
                "contribution": m['contribution'],
                **props
            })
        support[desc] = mol_info_list
    return support

# 6. 향미별 특화 예측 통합
def enhance_with_odor_specific_features(hybrid_result, input_mixture):
    """향미별 특화 특성으로 예측 결과 개선"""
    try:
        # 분자 데이터 준비
        molecules_data = [
            {'smiles': m['SMILES'], 'peak_area': m['peak_area']} 
            for m in input_mixture
        ]
        
        # 향미별 특화 점수 계산
        enhanced_scores = {}
        
        for mol_data in molecules_data:
            odor_scores = OdorSpecificFeatureExtractor.calculate_odor_potency(
                mol_data['smiles'], mol_data['peak_area']
            )
            
            for odor_type, score in odor_scores.items():
                if odor_type not in enhanced_scores:
                    enhanced_scores[odor_type] = 0.0
                enhanced_scores[odor_type] += score
        
        # 기존 예측과 융합 (가중평균)
        odor_result = hybrid_result['odor']['corrected']
        
        # 특화 특성 적용 (기존 예측의 80% + 특화 예측의 20%)
        fusion_weight = 0.2
        
        for desc in ['Fruity', 'Woody', 'Sweet', 'Floral']:
            if desc in odor_result and desc in enhanced_scores:
                original_score = odor_result[desc]
                enhanced_score = min(10.0, enhanced_scores[desc])  # 0-10 범위로 정규화
                
                # 융합된 점수 계산
                fused_score = (1 - fusion_weight) * original_score + fusion_weight * enhanced_score
                odor_result[desc] = max(0.0, min(10.0, fused_score))
        
        # 분자별 기여도 정보 추가
        hybrid_result['enhanced_molecular_insights'] = {
            'individual_odor_scores': enhanced_scores,
            'fusion_applied': True,
            'fusion_weight': fusion_weight
        }
        
    except Exception as e:
        print(f"[WARN] 향미별 특화 예측 실패: {e}")
        hybrid_result['enhanced_molecular_insights'] = {
            'fusion_applied': False,
            'error': str(e)
        }
    
    return hybrid_result

# 7. 하이브리드 예측 (향미별 특화 예측 통합)
def predict_hybrid(
    input_mixture,
    method: str = HYBRID_DEFAULTS['method'],
    weight_finetune: float = HYBRID_DEFAULTS['weight_finetune'],
    verbose: bool = True
):
    hybrid_result = {}
    smiles_list = [m['SMILES'] for m in input_mixture]
    peak_areas = np.array([m['peak_area'] for m in input_mixture], dtype=np.float32)

    for mode in ['odor', 'taste']:
        # 규칙 기반 예측
        sci = predict_mixture(input_mixture, mode=mode)
        desc_list = ODOR_DESCRIPTORS if mode == 'odor' else TASTE_DESCRIPTORS
        rule_log = sci.get('rule_log', [])

        # intensity matrix 및 contributions
        intensity_matrix, descs, *_ = prepare_mixture_data(smiles_list, peak_areas, mode=mode)
        contributions = calc_contributions(input_mixture, intensity_matrix, descs)

        # 파인튜닝 예측
        try:
            fin = predict_finetuned(input_mixture, mode=mode)
            finetune_used = True
        except Exception as e:
            if verbose:
                print(f"[WARN] finetuned {mode} 예측 불가: {e}")
            fin = {'predicted': {k: 0.0 for k in desc_list}}
            finetune_used = False

        # weight_finetune override from hybrid params
        wt = hybrid_params.get(mode, {}).get('weight_finetune', weight_finetune)

        # 결과 병합
        merged = {}
        for k in desc_list:
            s = sci['corrected'].get(k, 0.0)
            f = fin['predicted'].get(k, 0.0)
            # 병합 전 값 계산
            if finetune_used and k in fin['predicted']:
                if method == 'average':
                    v = (s + f) / 2
                elif method == 'weighted':
                    v = (1 - wt) * s + wt * f
                else:
                    v = s
            else:
                v = s

            # 디버그 출력
            print(f"[DEBUG {mode}] desc={k}, rule={s:.3f}, finetune={f:.3f}, merged_preclamp={v:.3f}")

            # clamp
            merged[k] = float(max(0.0, min(10.0, v)))

        hybrid_result[mode] = {
            'corrected': merged,
            'rule_log': rule_log,
            'finetune_used': finetune_used,
            'contributions': contributions
        }

    # description_support 생성
    support_odor = build_description_support(hybrid_result['odor']['contributions'], mode='odor')
    support_taste = build_description_support(hybrid_result['taste']['contributions'], mode='taste')
    hybrid_result['description_support'] = {'odor': support_odor, 'taste': support_taste}

    # 🚀 향미별 특화 예측 통합
    hybrid_result = enhance_with_odor_specific_features(hybrid_result, input_mixture)

    return hybrid_result

# 7. CLI debug entry
if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='Debug predict_hybrid')
    parser.add_argument('--input', type=str, help='Path to JSON file with mixture data')
    args = parser.parse_args()

    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load input JSON: {e}")
            sys.exit(1)
    else:
        data = [
            {"SMILES": "CCCCCCCC(=O)OCC(C)C", "peak_area": 803959.33},
            {"SMILES": "CN(C)C(=O)OC1=CC(C)=C(C)C=C1", "peak_area": 14293392.4}
        ]
    print("=== Debug Input ===")
    import json as _j; print(_j.dumps(data, indent=2, ensure_ascii=False))
    result = predict_hybrid(data)
    print("=== Debug Output ===")
    print(_j.dumps(result, indent=2, ensure_ascii=False))
