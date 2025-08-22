"""
Phase 3: Concentration Adaptivity Enhancement Module
===================================================

This module implements advanced concentration-based weighting systems
that adapt rule strength based on molecular concentration patterns,
sensory thresholds, and mixture complexity factors.

Features:
- S-curve concentration response modeling
- Sensory threshold integration  
- Mixture complexity analysis
- Adaptive weighting algorithms

Author: CJ Whisky Odor Model Team
Version: 3.0 - Phase 3 Implementation
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

@dataclass
class SensoryThresholds:
    """Sensory threshold data for functional groups"""
    detection_threshold: float  # 인지 가능 최소 농도
    recognition_threshold: float  # 식별 가능 농도
    saturation_point: float  # 감각 포화 농도

class ConcentrationAdaptivityEngine:
    """농도 적응성 엔진 - Phase 3 핵심 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 감각 임계값 데이터베이스
        self.sensory_thresholds = self._load_sensory_thresholds()
        
        # 농도 반응 곡선 파라미터 (Phase 3 수정: 더 강한 억제)
        self.s_curve_params = {
            'steepness': 8.0,       # S-curve 기울기 (4.0 → 8.0: 더 빠른 포화)
            'midpoint': 0.25,       # 중점 농도 (0.3 → 0.25: 더 낮은 포화점)
            'min_response': 0.1,    # 최소 반응
            'max_response': 0.9     # 최대 반응 (1.0 → 0.9: 억제 강화)
        }
        
        # 복잡도 분석 파라미터 (Phase 3 수정: 강화된 패널티)
        self.complexity_params = {
            'entropy_weight': 0.7,     # 엔트로피 가중치 (0.5 → 0.7)
            'diversity_weight': 0.3,   # 다양성 가중치  
            'concentration_weight': 0.4, # 농도 분산 가중치
            'interaction_weight': 0.3,   # 상호작용 가중치
            'max_penalty': 0.6         # 최대 패널티 (0.3 → 0.6: 더 강한 억제)
        }
        
        self.logger.info("농도 적응성 엔진 초기화 완료")
    
    def _load_sensory_thresholds(self) -> Dict[str, SensoryThresholds]:
        """감각 임계값 데이터베이스 로드"""
        # 문헌 기반 실제 임계값 데이터
        thresholds = {
            'alcohol': SensoryThresholds(0.05, 0.15, 0.8),
            'ester': SensoryThresholds(0.02, 0.08, 0.6),
            'terpene': SensoryThresholds(0.01, 0.05, 0.4),
            'aldehyde': SensoryThresholds(0.03, 0.10, 0.7),
            'phenol': SensoryThresholds(0.08, 0.25, 0.9),
            'furan': SensoryThresholds(0.04, 0.12, 0.5),
            'sulfur': SensoryThresholds(0.001, 0.005, 0.02),  # 매우 낮은 임계값
            'lactone': SensoryThresholds(0.02, 0.08, 0.4),
            'pyrazine': SensoryThresholds(0.01, 0.06, 0.3),
            'fatty_chain': SensoryThresholds(0.10, 0.30, 1.0),
            'amine': SensoryThresholds(0.005, 0.02, 0.1)
        }
        
        self.logger.info(f"감각 임계값 데이터 로드 완료: {len(thresholds)}개 작용기")
        return thresholds
    
    def calculate_concentration_factor(self, concentration: float, 
                                     functional_group: str,
                                     descriptor_type: str = 'odor') -> float:
        """
        농도별 적응 계수 계산 (S-curve 모델)
        
        Args:
            concentration: 분자 농도 (0-1)
            functional_group: 작용기 타입
            descriptor_type: 'odor' 또는 'taste'
            
        Returns:
            적응 계수 (0.1-1.0)
        """
        # 작용기별 임계값 가져오기
        thresholds = self.sensory_thresholds.get(functional_group, 
                                                 SensoryThresholds(0.05, 0.15, 0.8))
        
        # S-curve 함수: f(c) = min + (max-min) / (1 + e^(-k*(c-c0)))
        steepness = self.s_curve_params['steepness']
        midpoint = thresholds.recognition_threshold  # 작용기별 인식 임계값 사용
        min_resp = self.s_curve_params['min_response']
        max_resp = self.s_curve_params['max_response']
        
        # S-curve 계산
        exp_term = math.exp(-steepness * (concentration - midpoint))
        s_factor = min_resp + (max_resp - min_resp) / (1 + exp_term)
        
        # 포화 효과 적용
        if concentration > thresholds.saturation_point:
            saturation_factor = 1 - (concentration - thresholds.saturation_point) * 0.3
            s_factor *= max(0.7, saturation_factor)
        
        # Taste vs Odor 민감도 차이
        if descriptor_type == 'taste':
            s_factor *= 0.85  # Taste는 일반적으로 덜 민감
        
        return np.clip(s_factor, 0.1, 1.0)
    
    def analyze_mixture_complexity(self, molecules: List[str], 
                                  concentrations: List[float],
                                  functional_groups: List[str]) -> float:
        """
        혼합물 복잡도 분석
        
        Args:
            molecules: SMILES 리스트
            concentrations: 농도 리스트
            functional_groups: 작용기 리스트
            
        Returns:
            복잡도 점수 (0-1)
        """
        if not molecules or not concentrations:
            return 0.0
        
        concentrations = np.array(concentrations)
        
        # 1. 분자 다양성 (Shannon 엔트로피)
        normalized_conc = concentrations / np.sum(concentrations)
        entropy = -np.sum(normalized_conc * np.log2(normalized_conc + 1e-10))
        max_entropy = math.log2(len(molecules))
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        # 2. 농도 분산 (균등 분포 vs 지배적 분포)
        variance_score = np.var(normalized_conc) * 4  # 0-1 정규화
        variance_score = np.clip(variance_score, 0, 1)
        
        # 3. 작용기 상호작용 잠재성
        unique_groups = set(functional_groups)
        interaction_score = min(1.0, len(unique_groups) / 6)  # 6개 이상이면 포화
        
        # 가중 평균으로 최종 복잡도 계산
        complexity = (
            diversity_score * self.complexity_params['diversity_weight'] +
            variance_score * self.complexity_params['concentration_weight'] +
            interaction_score * self.complexity_params['interaction_weight']
        )
        
        self.logger.debug(f"복잡도 분석: 다양성={diversity_score:.3f}, "
                         f"분산={variance_score:.3f}, 상호작용={interaction_score:.3f}, "
                         f"최종={complexity:.3f}")
        
        return complexity
    
    def adaptive_rule_strength(self, base_strength: float,
                              concentration: float,
                              functional_group: str,
                              complexity_score: float,
                              descriptor_type: str = 'odor') -> float:
        """
        적응적 규칙 강도 계산
        
        Args:
            base_strength: 기본 규칙 강도
            concentration: 농도
            functional_group: 작용기
            complexity_score: 복잡도 점수
            descriptor_type: 서술자 타입
            
        Returns:
            조정된 규칙 강도
        """
        # 농도 기반 적응 계수
        conc_factor = self.calculate_concentration_factor(
            concentration, functional_group, descriptor_type
        )
        
        # 복잡도 기반 조정 (Phase 3 수정: 더 강한 억제)
        complexity_penalty = complexity_score * self.complexity_params['max_penalty']  # 0-0.6 패널티
        complexity_adjustment = 1.0 - complexity_penalty  # 복잡할수록 강도 감소
        complexity_adjustment = np.clip(complexity_adjustment, 0.4, 1.0)  # 최대 60% 감소
        
        # 최종 강도 계산 (Phase 3 수정: 더 보수적)
        strength_delta = (base_strength - 1.0) * conc_factor * complexity_adjustment
        adjusted_strength = 1.0 + strength_delta * 0.7  # 추가로 30% 감쇠
        
        # 안전 범위 보장 (Phase 3 수정: 더 엄격한 제한)
        adjusted_strength = np.clip(adjusted_strength, 0.95, 1.2)  # 1.3 → 1.2
        
        self.logger.debug(f"적응적 강도: {base_strength:.3f} → {adjusted_strength:.3f} "
                         f"(농도계수={conc_factor:.3f}, 복잡도={complexity_adjustment:.3f})")
        
        return adjusted_strength
    
    def concentration_based_threshold(self, concentrations: List[float],
                                    functional_groups: List[str],
                                    base_threshold: float = 0.15) -> float:
        """
        농도 기반 동적 임계값 계산
        
        Args:
            concentrations: 농도 리스트
            functional_groups: 작용기 리스트
            base_threshold: 기본 임계값
            
        Returns:
            동적 임계값
        """
        if not concentrations or not functional_groups:
            return base_threshold
        
        concentrations = np.array(concentrations)
        
        # 작용기별 민감도 고려
        sensitivity_weights = []
        for group in functional_groups:
            threshold_data = self.sensory_thresholds.get(group, 
                                                       SensoryThresholds(0.05, 0.15, 0.8))
            # 민감도 = 1 / 인식 임계값 (낮을수록 민감)
            sensitivity = 1.0 / threshold_data.recognition_threshold
            sensitivity_weights.append(sensitivity)
        
        # 가중 평균 민감도
        if sensitivity_weights:
            avg_sensitivity = np.mean(sensitivity_weights)
            # 민감도에 따른 임계값 조정 (민감할수록 낮은 임계값)
            adjusted_threshold = base_threshold / (1 + (avg_sensitivity - 5) * 0.1)
            adjusted_threshold = np.clip(adjusted_threshold, 0.05, 0.3)
        else:
            adjusted_threshold = base_threshold
        
        self.logger.debug(f"동적 임계값: {base_threshold:.3f} → {adjusted_threshold:.3f}")
        return adjusted_threshold
    
    def get_enhancement_profile(self, molecules_data: List[Dict]) -> Dict[str, Any]:
        """
        혼합물에 대한 전체 농도 적응성 프로파일 생성
        
        Args:
            molecules_data: [{'SMILES': str, 'peak_area': float, 'functional_groups': List[str]}]
            
        Returns:
            농도 적응성 프로파일
        """
        if not molecules_data:
            return {}
        
        # 농도 정규화
        total_area = sum(mol.get('peak_area', 1.0) for mol in molecules_data)
        concentrations = [mol.get('peak_area', 1.0) / total_area for mol in molecules_data]
        
        # 작용기 수집
        all_functional_groups = []
        for mol in molecules_data:
            groups = mol.get('functional_groups', [])
            all_functional_groups.extend(groups)
        
        # 복잡도 분석
        smiles_list = [mol.get('SMILES', '') for mol in molecules_data]
        complexity_score = self.analyze_mixture_complexity(
            smiles_list, concentrations, all_functional_groups
        )
        
        # 각 분자별 적응 계수 계산
        molecule_profiles = []
        for i, mol_data in enumerate(molecules_data):
            mol_groups = mol_data.get('functional_groups', [])
            conc = concentrations[i]
            
            group_factors = {}
            for group in mol_groups:
                odor_factor = self.calculate_concentration_factor(conc, group, 'odor')
                taste_factor = self.calculate_concentration_factor(conc, group, 'taste')
                group_factors[group] = {'odor': odor_factor, 'taste': taste_factor}
            
            molecule_profiles.append({
                'smiles': mol_data.get('SMILES', ''),
                'concentration': conc,
                'functional_groups': mol_groups,
                'adaptation_factors': group_factors
            })
        
        return {
            'complexity_score': complexity_score,
            'total_molecules': len(molecules_data),
            'concentration_variance': np.var(concentrations),
            'dominant_concentration': max(concentrations),
            'molecule_profiles': molecule_profiles,
            'adaptive_thresholds': {
                'odor': self.concentration_based_threshold(concentrations, all_functional_groups, 0.15),
                'taste': self.concentration_based_threshold(concentrations, all_functional_groups, 0.18)
            }
        }


# 전역 인스턴스
concentration_engine = ConcentrationAdaptivityEngine()
