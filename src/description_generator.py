import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from config import DRAVNIEKS_CSV

# 캐시 및 맵 초기화
_df_cache = None
_primary_odor_map = {}
_primary_taste_map = {}

# 1회만 로딩
def get_primary_maps():
    global _df_cache, _primary_odor_map, _primary_taste_map
    if _df_cache is None:
        try:
            _df_cache = pd.read_csv(str(DRAVNIEKS_CSV), encoding='utf-8-sig')
        except Exception:
            _df_cache = pd.read_csv(str(DRAVNIEKS_CSV), encoding='cp949')
        _df_cache.columns = _df_cache.columns.str.strip()
        _primary_odor_map = dict(zip(_df_cache['SMILES'], _df_cache['Primary_Odor'].fillna('')))
        _primary_taste_map = dict(zip(_df_cache['SMILES'], _df_cache['Primary_Taste'].fillna('')))
    return _primary_odor_map, _primary_taste_map

# 강도 구문 변환 (개선된 버전)
def get_intensity_phrase(normalized_score, use_korean=False):
    """점수 기반 강도 표현 (0-10 스케일)"""
    if normalized_score >= 8.0:
        return '매우 강한' if use_korean else 'very strong'
    elif normalized_score >= 6.0:
        return '강한' if use_korean else 'strong'
    elif normalized_score >= 4.0:
        return '적당한' if use_korean else 'moderate'
    elif normalized_score >= 2.0:
        return '약한' if use_korean else 'weak'
    else:
        return '매우 약한' if use_korean else 'very weak'

# 개선된 노트 위치 추정 (농도와 분자 특성 모두 고려)
def estimate_note_position_enhanced(molwt, logp, tpsa, concentration=1.0, contribution=1.0):
    """
    분자의 물리화학적 성질과 농도, 기여도를 종합적으로 고려한 노트 위치 추정
    """
    try:
        # 기본 분자 특성 점수
        volatility_score = 0
        if molwt < 100:
            volatility_score += 3
        elif molwt < 150:
            volatility_score += 2
        elif molwt < 200:
            volatility_score += 1
        
        if logp < 1.0:
            volatility_score += 2
        elif logp < 3.0:
            volatility_score += 1
        
        if tpsa < 30:
            volatility_score += 2
        elif tpsa < 60:
            volatility_score += 1
        
        # 농도와 기여도에 따른 가중치
        impact_factor = min(concentration * contribution, 3.0)  # 최대 3배까지
        weighted_score = volatility_score * (0.7 + 0.3 * impact_factor)
        
        # 최종 분류
        if weighted_score >= 6:
            return 'top note'
        elif weighted_score >= 3:
            return 'mid note'
        else:
            return 'base note'
            
    except Exception:
        return 'mid note'

# 시너지/마스킹 효과를 고려한 기여도 조정
def adjust_contribution_for_interactions(contribution, smiles, all_molecules, interaction_data=None):
    """
    분자 간 상호작용을 고려한 기여도 조정
    """
    if not interaction_data:
        return contribution
    
    adjusted = contribution
    
    # 시너지 효과
    for synergy in interaction_data.get('synergy', []):
        if smiles in [synergy.get('molecule1'), synergy.get('molecule2')]:
            synergy_factor = synergy.get('synergy_score', 1.0)
            adjusted *= (1.0 + synergy_factor * 0.3)  # 최대 30% 증가
    
    # 마스킹 효과
    for masking in interaction_data.get('masking', []):
        if smiles == masking.get('masked'):
            masking_ratio = masking.get('ratio', 1.0)
            adjusted *= max(0.3, 1.0 - (masking_ratio - 1.0) * 0.1)  # 최대 70% 감소
    
    return adjusted

# 개선된 기여도 스코어 계산
def calculate_enhanced_rank_scores(corrected_dict, support_dict, interaction_data=None, input_molecules=None):
    """
    분자 특성, 농도, 상호작용을 모두 고려한 점수 계산
    """
    scored = []
    
    # 입력 분자 정보를 SMILES로 매핑
    molecule_map = {}
    if input_molecules:
        for mol in input_molecules:
            if 'SMILES' in mol:
                molecule_map[mol['SMILES']] = mol.get('peak_area', 1.0)
    
    for desc, intensity in corrected_dict.items():
        descriptor_molecules = support_dict.get(desc, [])
        
        for mol in descriptor_molecules:
            smiles = mol.get('smiles', '')
            base_contribution = mol.get('contribution', 0)
            concentration = molecule_map.get(smiles, 1.0)
            
            # 상호작용을 고려한 기여도 조정
            adjusted_contribution = adjust_contribution_for_interactions(
                base_contribution, smiles, input_molecules, interaction_data
            )
            
            # 농도 가중치 적용 (로그 스케일)
            concentration_weight = min(np.log10(concentration + 1), 2.0)
            
            # 최종 점수 계산
            final_score = intensity * adjusted_contribution * (1.0 + concentration_weight * 0.2)
            
            scored.append({
                'descriptor': desc,
                'smiles': smiles,
                'score': final_score,
                'base_contribution': base_contribution,
                'adjusted_contribution': adjusted_contribution,
                'concentration': concentration,
                'intensity': intensity,
                'MolWt': mol.get('MolWt', 150),
                'LogP': mol.get('LogP', 2.0),
                'TPSA': mol.get('TPSA', 50)
            })
    
    return sorted(scored, key=lambda x: -x['score'])

# 개선된 문장 생성
def generate_enhanced_sensory_description(predict_result: dict, input_molecules=None):
    """
    향상된 감각 설명 생성 - 상호작용과 농도를 고려
    """
    odor = predict_result.get('odor', {})
    taste = predict_result.get('taste', {})
    support = predict_result.get('description_support', {})
    interaction_data = predict_result.get('interactions', {})

    # 개선된 점수 계산
    odor_scores = calculate_enhanced_rank_scores(
        odor.get('corrected', {}), 
        support.get('odor', {}),
        interaction_data.get('odor', {}),
        input_molecules
    )
    taste_scores = calculate_enhanced_rank_scores(
        taste.get('corrected', {}), 
        support.get('taste', {}),
        interaction_data.get('taste', {}),
        input_molecules
    )

    def summarize_enhanced(scores, mode):
        """개선된 요약 생성"""
        note_dict = {'top note': [], 'mid note': [], 'base note': []}
        primary_map = get_primary_maps()[0] if mode == 'odor' else get_primary_maps()[1]
        
        # 점수 정규화 (0-10 스케일)
        max_score = max([s['score'] for s in scores[:15]], default=1)
        
        for s in scores[:15]:
            normalized_score = (s['score'] / max_score) * 10
            
            strength = get_intensity_phrase(normalized_score)
            
            # 개선된 노트 위치 추정
            note = estimate_note_position_enhanced(
                s['MolWt'], s['LogP'], s['TPSA'],
                s.get('concentration', 1.0),
                s.get('adjusted_contribution', 1.0)
            )
            
            primary = primary_map.get(s['smiles'], '')
            
            # 상호작용 정보 추가
            interaction_note = ""
            if s.get('adjusted_contribution', 0) > s.get('base_contribution', 0) * 1.2:
                interaction_note = " (synergistic)"
            elif s.get('adjusted_contribution', 0) < s.get('base_contribution', 0) * 0.8:
                interaction_note = " (masked)"
            
            for p in [p.strip() for p in primary.split(',') if p.strip()]:
                descriptor_with_intensity = f"{strength} {p}{interaction_note}"
                note_dict[note].append(descriptor_with_intensity)
        
        return note_dict

    def assemble_enhanced_note(note_dict, label):
        """개선된 노트 조합 - 구조화된 요약 형식으로 변경"""
        lines = []
        
        # 강도 순서 정의 (높은 순서부터)
        intensity_order = ['very strong', 'strong', 'moderate', 'weak', 'very weak']
        
        # Nose 또는 Taste 섹션 시작
        if label == 'Aroma':
            lines.append("Nose")
        else:
            lines.append("Taste")
        
        for i, note in enumerate(['top note', 'mid note', 'base note'], 1):
            # descriptor별로 가장 강한 강도만 유지
            descriptor_intensity_map = {}
            
            for desc_with_intensity in note_dict[note]:
                # 강도와 descriptor 분리
                parts = desc_with_intensity.split(' ', 2)  # 최대 3개로 분리
                if len(parts) >= 2:
                    if len(parts) == 2:  # "strong Almond"
                        intensity, descriptor = parts[0], parts[1]
                        suffix = ""
                    else:  # "very strong Almond" 또는 "strong Almond (synergistic)"
                        if parts[0] == 'very':
                            intensity = f"{parts[0]} {parts[1]}"
                            rest = parts[2]
                        else:
                            intensity = parts[0]
                            rest = ' '.join(parts[1:])
                        
                        # (synergistic) 또는 (masked) 처리
                        if '(' in rest:
                            descriptor = rest.split('(')[0].strip()
                            suffix = f" ({rest.split('(')[1]}"
                        else:
                            descriptor = rest.strip()
                            suffix = ""
                    
                    # 현재 descriptor에 대해 더 강한 강도가 있는지 확인
                    if descriptor not in descriptor_intensity_map:
                        descriptor_intensity_map[descriptor] = (intensity, suffix)
                    else:
                        current_intensity = descriptor_intensity_map[descriptor][0]
                        # 더 강한 강도로 업데이트
                        if intensity_order.index(intensity) < intensity_order.index(current_intensity):
                            descriptor_intensity_map[descriptor] = (intensity, suffix)
            
            # 최종 descriptor 리스트 생성
            final_descriptors = []
            for descriptor, (intensity, suffix) in descriptor_intensity_map.items():
                final_descriptors.append(f"{intensity} {descriptor}{suffix}")
            
            # 구조화된 형식으로 출력
            if label == 'Aroma':
                note_names = ['Top notes', 'Mid notes', 'Base notes']
            else:
                note_names = ['Top palates', 'Mid palates', 'Base palates']
            
            if final_descriptors:
                lines.append(f"{i}. {note_names[i-1]} : {', '.join(final_descriptors[:6])}")
            else:
                lines.append(f"{i}. {note_names[i-1]} : (no significant notes)")
        
        return '\n'.join(lines)

    # 최종 결과 생성
    odor_summary = summarize_enhanced(odor_scores, 'odor')
    taste_summary = summarize_enhanced(taste_scores, 'taste')
    
    en_summary = (
        assemble_enhanced_note(odor_summary, 'Aroma') + 
        '\n\n' + 
        assemble_enhanced_note(taste_summary, 'Palate')
    )
    
    # 추가 메타데이터
    metadata = {
        'total_molecules_analyzed': len(input_molecules) if input_molecules else 0,
        'top_aroma_contributors': len(odor_scores[:5]),
        'top_taste_contributors': len(taste_scores[:5]),
        'interaction_effects': {
            'synergy_count': len(interaction_data.get('odor', {}).get('synergy', [])),
            'masking_count': len(interaction_data.get('odor', {}).get('masking', []))
        }
    }
    
    return {
        'en': en_summary,
        'metadata': metadata,
        'detailed_scores': {
            'odor': odor_scores[:10],
            'taste': taste_scores[:10]
        }
    }

# 호환성을 위한 기존 함수 유지
def generate_sensory_description(predict_result: dict):
    """기존 호환성을 위한 래퍼 함수"""
    return generate_enhanced_sensory_description(predict_result)
