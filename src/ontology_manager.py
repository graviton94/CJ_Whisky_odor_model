"""
Enhanced Ontology Manager for Dynamic Rule Learning

Phase 1: Dynamic Ontology Rules Implementation
- Synergy Rules: Descriptor enhancement based on molecular combinations
- Masking Rules: Descriptor attenuation from interference
- Functional Group Rules: Chemical group-based adjustments
- Derived Feature Rules: Complex feature interactions

Author: CJ Whisky Odor Model Team
Version: 1.0 - Phase 1 Implementation
"""

import json
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Phase 3: 농도 적응성 엔진 import
try:
    from concentration_adaptivity import concentration_engine
    CONCENTRATION_ADAPTIVITY_AVAILABLE = True
except ImportError:
    CONCENTRATION_ADAPTIVITY_AVAILABLE = False
    logging.warning("농도 적응성 엔진을 찾을 수 없습니다. 기본 농도 처리를 사용합니다.")

class OntologyManager:
    """Enhanced Ontology Manager with dynamic rule learning capabilities"""
    
    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialize the enhanced ontology manager
        
        Args:
            rules_file: Path to JSON rules file (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Rule storage
        self.synergy_rules = {}
        self.masking_rules = {}
        self.functional_group_rules = {}
        self.derived_feature_rules = {}
        
        # Parameters for rule application (TUNED: 더 엄격한 조건)
        self.synergy_strength = 1.0
        self.masking_strength = 1.0
        self.concentration_threshold = 0.15  # 0.1 → 0.15 (50% 상향)
        
        # Phase 3: 농도 적응성 설정
        self.use_concentration_adaptivity = CONCENTRATION_ADAPTIVITY_AVAILABLE
        self.adaptive_threshold_enabled = True
        
        # Load rules if file provided
        if rules_file and Path(rules_file).exists():
            self.load_rules(rules_file)
        else:
            self._create_default_rules()
            
        if self.use_concentration_adaptivity:
            self.logger.info("농도 적응성 엔진 활성화됨")
        else:
            self.logger.warning("농도 적응성 엔진 비활성화 - 기본 모드로 동작")
    
    def _create_default_rules(self):
        """Create default rule sets for initialization - TUNED VERSION"""
        # Synergy rules: combinations that enhance descriptors (TUNED: 축소된 가중치 + 강화된 조건)
        self.synergy_rules = {
            "Fruity": {
                "triggers": [["ester", "alcohol"], ["ester", "aldehyde"], ["ester", "furan"]],
                "strength": 1.08,  # 1.15 → 1.08 (추가 축소 - Phase 3 수정)
                "conditions": {"min_concentration": 0.20}  # 0.15 → 0.20 (강화)
            },
            "Sweet": {
                "triggers": [["ester", "alcohol"], ["aldehyde", "ester"], ["alcohol", "aldehyde"], ["furan", "ester"]],
                "strength": 1.08,  # 1.12 → 1.08 (추가 축소 - Phase 3 수정)
                "conditions": {"min_concentration": 0.20}  # 0.15 → 0.20 (강화)
            },
            "Floral": {
                "triggers": [["ester", "alcohol"], ["alcohol", "terpene"], ["ester", "terpene"]],
                "strength": 1.06,  # 1.10 → 1.06 (추가 축소 - Phase 3 수정)
                "conditions": {"min_concentration": 0.18}  # 0.15 → 0.18 (강화)
            },
            "Woody": {
                "triggers": [["phenol", "alcohol"], ["furan", "phenol"], ["phenol", "aldehyde"], ["lactone", "phenol"]],
                "strength": 1.12,  # 1.4 → 1.12 (70% 축소)
                "conditions": {"min_concentration": 0.25}  # 0.2 → 0.25 (강화)
            },
            "Vanilla": {
                "triggers": [["aldehyde", "phenol"], ["furan", "alcohol"], ["lactone", "aldehyde"], ["phenol", "ester"]],
                "strength": 1.18,  # 1.6 → 1.18 (74% 축소)
                "conditions": {"min_concentration": 0.20}  # 0.15 → 0.20 (강화)
            },
            "Almond": {
                "triggers": [["aldehyde", "ester"], ["furan", "aldehyde"], ["lactone", "furan"], ["pyrazine", "aldehyde"]],
                "strength": 1.15,  # 1.5 → 1.15 (67% 축소)
                "conditions": {"min_concentration": 0.22}  # 0.2 → 0.22 (강화)
            },
            "Citrus": {
                "triggers": [["terpene", "ester"], ["terpene", "aldehyde"], ["ester", "alcohol"]],
                "strength": 1.08,  # 1.3 → 1.08 (69% 축소)
                "conditions": {"min_concentration": 0.15}  # 0.1 → 0.15 (강화)
            },
            "Earthy": {
                "triggers": [["phenol", "furan"], ["lactone", "phenol"], ["sulfur", "phenol"]],
                "strength": 1.08,  # 1.3 → 1.08 (69% 축소)
                "conditions": {"min_concentration": 0.25}  # 0.2 → 0.25 (강화)
            },
            "Spicy": {
                "triggers": [["phenol", "aldehyde"], ["furan", "phenol"], ["sulfur", "phenol"]],
                "strength": 1.12,  # 1.4 → 1.12 (70% 축소)
                "conditions": {"min_concentration": 0.20}  # 0.15 → 0.20 (강화)
            },
            "Green": {
                "triggers": [["aldehyde", "alcohol"], ["terpene", "aldehyde"], ["alcohol", "fatty_chain"]],
                "strength": 1.06,  # 1.2 → 1.06 (65% 축소)
                "conditions": {"min_concentration": 0.15}  # 0.1 → 0.15 (강화)
            },
            "Fragrant": {
                "triggers": [["ester", "terpene"], ["alcohol", "ester"], ["terpene", "alcohol"]],
                "strength": 1.08,  # 1.3 → 1.08 (69% 축소)
                "conditions": {"min_concentration": 0.15}  # 0.1 → 0.15 (강화)
            },
            "Minty": {
                "triggers": [["terpene", "alcohol"], ["ester", "terpene"], ["aldehyde", "terpene"]],
                "strength": 1.06,  # 1.2 → 1.06 (65% 축소)
                "conditions": {"min_concentration": 0.20}  # 0.15 → 0.20 (강화)
            }
        }
        
        # Masking rules: combinations that suppress descriptors (확장)
        self.masking_rules = {
            "Sweet": {
                "triggers": [["sulfur", "any"], ["phenol", "high_concentration"], ["fatty_chain", "high_concentration"]],
                "strength": 0.7,
                "conditions": {"min_concentration": 0.1}
            },
            "Floral": {
                "triggers": [["sulfur", "any"], ["fatty_chain", "high_concentration"], ["pyrazine", "high_concentration"]],
                "strength": 0.6,
                "conditions": {"min_concentration": 0.2}
            },
            "Citrus": {
                "triggers": [["sulfur", "any"], ["phenol", "high_concentration"], ["lactone", "high_concentration"]],
                "strength": 0.8,
                "conditions": {"min_concentration": 0.15}
            },
            "Fragrant": {
                "triggers": [["sulfur", "any"], ["fatty_chain", "phenol"]],
                "strength": 0.75,
                "conditions": {"min_concentration": 0.15}
            },
            "Vanilla": {
                "triggers": [["sulfur", "high_concentration"], ["fatty_chain", "high_concentration"]],
                "strength": 0.8,
                "conditions": {"min_concentration": 0.2}
            },
            "Almond": {
                "triggers": [["sulfur", "phenol"], ["fatty_chain", "high_concentration"]],
                "strength": 0.75,
                "conditions": {"min_concentration": 0.2}
            }
        }
        
        # Functional group rules (PHASE 3 수정: 추가 축소된 가중치)
        self.functional_group_rules = {
            "ester": {"Fruity": 1.06, "Sweet": 1.04, "Floral": 1.03, "Citrus": 1.03, "Fragrant": 1.04},  # 1.12→1.06
            "alcohol": {"Woody": 1.03, "Earthy": 1.04, "Sweet": 1.02, "Green": 1.03, "Fragrant": 1.03},   # 1.06→1.03
            "phenol": {"Woody": 1.08, "Earthy": 1.06, "Spicy": 1.04, "Vanilla": 1.03},                    # 1.15→1.08
            "aldehyde": {"Green": 1.08, "Almond": 1.09, "Vanilla": 1.06, "Sweet": 1.03, "Fragrant": 1.04}, # 1.15→1.08
            "furan": {"Almond": 1.10, "Vanilla": 1.08, "Woody": 1.04, "Earthy": 1.03, "Sweet": 1.03},     # 1.20→1.10
            "sulfur": {"Earthy": 1.06, "Spicy": 1.04},                                                      # 1.12→1.06
            "terpene": {"Citrus": 1.09, "Minty": 1.08, "Floral": 1.04, "Fragrant": 1.06, "Green": 1.03},  # 1.18→1.09
            "lactone": {"Vanilla": 1.09, "Almond": 1.06, "Sweet": 1.04, "Woody": 1.03},                   # 1.18→1.09
            "pyrazine": {"Earthy": 1.08, "Spicy": 1.06, "Almond": 1.04},                                   # 1.15→1.08
            "fatty_chain": {"Green": 1.04, "Earthy": 1.03},                                                # 1.08→1.04
            "amine": {"Earthy": 1.04, "Spicy": 1.03}                                                       # 1.08→1.04
        }
        
        # Taste specific functional group rules (TUNED: 50-70% 축소된 가중치)
        self.taste_functional_group_rules = {
            "ester": {"Taste_Fruity": 1.15, "Taste_Sweet": 1.12, "Taste_Floral": 1.08},
            "alcohol": {"Taste_Sweet": 1.08, "Taste_Bitter": 0.96, "Taste_Sour": 1.02},
            "phenol": {"Taste_Bitter": 1.18, "Taste_Sour": 1.12, "Taste_OffFlavor": 1.05},
            "aldehyde": {"Taste_Nutty": 1.15, "Taste_Sweet": 1.08, "Taste_Floral": 1.05},
            "furan": {"Taste_Nutty": 1.18, "Taste_Sweet": 1.12, "Taste_Bitter": 1.05},
            "sulfur": {"Taste_OffFlavor": 1.20, "Taste_Bitter": 1.12, "Taste_Sour": 1.08},
            "terpene": {"Taste_Floral": 1.12, "Taste_Sweet": 1.05, "Taste_Sour": 1.02},
            "lactone": {"Taste_Sweet": 1.15, "Taste_Nutty": 1.08, "Taste_Floral": 1.05},
            "pyrazine": {"Taste_Bitter": 1.12, "Taste_OffFlavor": 1.08, "Taste_Nutty": 1.05},
            "fatty_chain": {"Taste_OffFlavor": 1.08, "Taste_Bitter": 1.05},
            "amine": {"Taste_OffFlavor": 1.12, "Taste_Bitter": 1.08}
        }
        
        # Derived feature rules (complex interactions) - 확장
        self.derived_feature_rules = {
            "high_concentration_boost": {
                "condition": "total_concentration > 0.7",
                "effects": {"intensity_multiplier": 1.3}
            },
            "medium_concentration_boost": {
                "condition": "total_concentration > 0.4",
                "effects": {"intensity_multiplier": 1.15}
            },
            "diversity_complexity": {
                "condition": "num_unique_groups > 4",
                "effects": {"complexity_boost": 1.2}
            },
            "ester_alcohol_synergy": {
                "condition": "ester_alcohol_combination",
                "effects": {"Fruity": 1.4, "Sweet": 1.3, "Taste_Fruity": 1.3, "Taste_Sweet": 1.2}
            },
            "furan_aldehyde_synergy": {
                "condition": "furan_aldehyde_combination",
                "effects": {"Almond": 1.5, "Vanilla": 1.4, "Taste_Nutty": 1.3}
            },
            "phenol_complexity": {
                "condition": "phenol_present",
                "effects": {"Woody": 1.3, "Earthy": 1.2, "Spicy": 1.2, "Taste_Bitter": 1.2}
            },
            "terpene_freshness": {
                "condition": "terpene_present",
                "effects": {"Citrus": 1.4, "Minty": 1.3, "Fragrant": 1.2, "Taste_Floral": 1.2}
            },
            "sulfur_interference": {
                "condition": "sulfur_present",
                "effects": {"masking_factor": 0.8}
            }
        }
    
    def _safe_score_clamp(self, score: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
        """
        TUNED: 안전한 점수 범위 보호
        
        Args:
            score: 조정할 점수
            min_val: 최소값 (기본: 0.0)
            max_val: 최대값 (기본: 10.0)
            
        Returns:
            범위 내로 클램핑된 점수
        """
        return np.clip(score, min_val, max_val)
    
    def _apply_logarithmic_scaling(self, old_score: float, multiplier: float, max_score: float = 10.0) -> float:
        """
        PHASE 2: Weber-Fechner 법칙 기반 로그 스케일 적용
        
        Args:
            old_score: 원본 점수
            multiplier: 가중치
            max_score: 최대 점수 (기본: 10.0)
            
        Returns:
            로그 스케일 적용된 점수
        """
        if old_score <= 0:
            return old_score
        
        # Weber-Fechner law: 감각 강도는 자극의 로그에 비례
        log_old = np.log(old_score + 1)
        log_max = np.log(max_score + 1)
        
        # 높은 점수일수록 증폭 효과 감소 (0.3 ~ 1.0 범위)
        sensitivity_factor = 1 - (log_old / log_max) * 0.7
        enhancement = (multiplier - 1.0) * sensitivity_factor
        
        new_score = old_score * (1.0 + enhancement)
        return self._safe_score_clamp(new_score, 0.0, max_score)
    
    def _calculate_adaptive_strength(self, base_strength: float, 
                                   concentration: float,
                                   functional_groups: List[str],
                                   mixture_analysis: Dict[str, Any]) -> float:
        """
        PHASE 3: 농도 적응적 강도 계산
        
        Args:
            base_strength: 기본 강도
            concentration: 농도
            functional_groups: 작용기 리스트
            mixture_analysis: 혼합물 분석 결과
            
        Returns:
            적응적 강도
        """
        if not self.use_concentration_adaptivity:
            return base_strength * self.synergy_strength
        
        try:
            # 주요 작용기 추출 (첫 번째 작용기를 대표로 사용)
            primary_group = functional_groups[0] if functional_groups else 'unknown'
            
            # 복잡도 점수 계산
            complexity_score = mixture_analysis.get('weighted_strength', 0.5)
            
            # 농도 엔진을 통한 적응적 강도 계산
            adaptive_strength = concentration_engine.adaptive_rule_strength(
                base_strength, concentration, primary_group, complexity_score
            )
            
            # Synergy strength 적용
            final_strength = adaptive_strength * self.synergy_strength
            
            self.logger.debug(f"적응적 강도 계산: {base_strength:.3f} → {final_strength:.3f} "
                             f"(농도={concentration:.3f}, 그룹={primary_group})")
            
            return final_strength
            
        except Exception as e:
            self.logger.warning(f"적응적 강도 계산 실패: {e}, 기본 강도 사용")
            return base_strength * self.synergy_strength
    
    def _calculate_group_concentration(self, target_group: str, 
                                     all_groups: List[str], 
                                     concentrations: np.ndarray) -> float:
        """
        특정 작용기의 농도 계산
        
        Args:
            target_group: 대상 작용기
            all_groups: 전체 작용기 리스트
            concentrations: 농도 배열
            
        Returns:
            해당 작용기의 총 농도
        """
        group_concentration = 0.0
        
        # 해당 작용기를 포함하는 분자들의 농도 합산
        for i, group in enumerate(all_groups):
            if group == target_group and i < len(concentrations):
                group_concentration += concentrations[i]
        
        return group_concentration
    
    def _calculate_adaptive_multiplier(self, base_multiplier: float,
                                     concentration: float,
                                     functional_group: str,
                                     mixture_analysis: Dict[str, Any],
                                     descriptor_type: str) -> float:
        """
        적응적 승수 계산 (기능군 규칙용)
        
        Args:
            base_multiplier: 기본 승수
            concentration: 농도
            functional_group: 작용기
            mixture_analysis: 혼합물 분석 결과
            descriptor_type: 서술자 타입
            
        Returns:
            적응적 승수
        """
        if not self.use_concentration_adaptivity:
            return base_multiplier
        
        try:
            complexity_score = mixture_analysis.get('weighted_strength', 0.5)
            
            # 농도 엔진을 통한 적응적 승수 계산
            adaptive_multiplier = concentration_engine.adaptive_rule_strength(
                base_multiplier, concentration, functional_group, complexity_score, descriptor_type
            )
            
            self.logger.debug(f"기능군 적응적 승수: {base_multiplier:.3f} → {adaptive_multiplier:.3f} "
                             f"({functional_group}, 농도={concentration:.3f})")
            
            return adaptive_multiplier
            
        except Exception as e:
            self.logger.warning(f"적응적 승수 계산 실패: {e}, 기본 승수 사용")
            return base_multiplier
    
    def _calculate_cumulative_change(self, original_scores: Dict[str, float], 
                                   final_scores: Dict[str, float]) -> float:
        """
        TUNED: 누적 변화량 계산
        
        Args:
            original_scores: 원본 점수
            final_scores: 최종 점수
            
        Returns:
            평균 변화율
        """
        total_change = 0.0
        count = 0
        
        for descriptor in original_scores:
            if descriptor in final_scores:
                original = original_scores[descriptor]
                final = final_scores[descriptor]
                
                if original > 0:
                    change_rate = abs((final - original) / original)
                    total_change += change_rate
                    count += 1
        
        return total_change / count if count > 0 else 0.0
    
    def apply_rules(self, result_dict: Dict[str, float], smiles_list: List[str], 
                   descriptor_type: str = "odor", detectability: Optional[Dict] = None,
                   extra_features: Optional[Dict] = None, 
                   concentrations: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Apply all ontology rules to prediction results
        
        Args:
            result_dict: Prediction scores {descriptor: score}
            smiles_list: List of SMILES strings
            descriptor_type: 'odor' or 'taste'
            detectability: Detection thresholds (optional)
            extra_features: Additional molecular features (optional)
            concentrations: Molecular concentrations (optional)
            
        Returns:
            Dictionary with corrected scores and application log
        """
        try:
            # Initialize
            corrected_scores = result_dict.copy()
            rule_log = []
            
            # Normalize concentrations
            if concentrations is None:
                concentrations = [1.0] * len(smiles_list)
            
            concentrations = np.array(concentrations)
            if len(concentrations) > 0:
                concentrations = concentrations / np.sum(concentrations)
            
            # Analyze mixture interactions
            mixture_analysis = self._analyze_mixture_interactions(
                smiles_list, concentrations, descriptor_type
            )
            
            # PHASE 3: 농도 적응성 프로파일 생성
            if self.use_concentration_adaptivity and self.adaptive_threshold_enabled:
                molecules_data = []
                for i, smiles in enumerate(smiles_list):
                    mol_conc = concentrations[i] if i < len(concentrations) else 1.0
                    mol_groups = self._identify_functional_groups(smiles)
                    molecules_data.append({
                        'SMILES': smiles,
                        'peak_area': mol_conc,
                        'functional_groups': mol_groups
                    })
                
                concentration_profile = concentration_engine.get_enhancement_profile(molecules_data)
                mixture_analysis['concentration_profile'] = concentration_profile
                
                # 동적 임계값 업데이트
                dynamic_threshold = concentration_profile.get('adaptive_thresholds', {}).get(descriptor_type, 0.15)
                self.concentration_threshold = dynamic_threshold
                
                self.logger.debug(f"동적 임계값 적용: {dynamic_threshold:.3f} ({descriptor_type})")
            
            # Apply synergy rules
            corrected_scores, synergy_log = self._apply_synergy_rules(
                corrected_scores, mixture_analysis, concentrations
            )
            rule_log.extend(synergy_log)
            
            # Apply masking rules
            corrected_scores, masking_log = self._apply_masking_rules(
                corrected_scores, mixture_analysis, concentrations
            )
            rule_log.extend(masking_log)
            
            # Apply functional group rules
            corrected_scores, fg_log = self._apply_functional_group_rules(
                corrected_scores, mixture_analysis, concentrations, descriptor_type
            )
            rule_log.extend(fg_log)
            
            # Apply derived feature rules
            corrected_scores, derived_log = self._apply_derived_feature_rules(
                corrected_scores, mixture_analysis, extra_features
            )
            rule_log.extend(derived_log)
            
            # PHASE 3 수정안 3: 증폭 제한 메커니즘 적용
            limited_scores, limit_log = self._apply_amplification_limits(
                corrected_scores, result_dict
            )
            rule_log.extend(limit_log)
            
            # Clamp scores to valid range (0-10)
            clamped_scores, clamp_log = self._clamp_scores(limited_scores)
            rule_log.extend(clamp_log)
            
            return {
                'corrected_scores': clamped_scores,
                'rule_log': rule_log,
                'mixture_interactions': mixture_analysis,
                'parameters': {
                    'synergy_strength': self.synergy_strength,
                    'masking_strength': self.masking_strength,
                    'total_molecules': len(smiles_list),
                    'total_concentration': np.sum(concentrations)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Rule application failed: {e}")
            return {
                'corrected_scores': result_dict,
                'rule_log': [{'error': str(e)}],
                'mixture_interactions': {},
                'parameters': {}
            }
    
    def _analyze_mixture_interactions(self, smiles_list: List[str], 
                                    concentrations: np.ndarray,
                                    descriptor_type: str) -> Dict[str, Any]:
        """Analyze molecular interactions in the mixture"""
        try:
            # Basic functional group analysis (simplified)
            functional_groups = []
            for smiles in smiles_list:
                groups = self._identify_functional_groups(smiles)
                functional_groups.extend(groups)
            
            # Calculate interaction strength
            interaction_strength = len(set(functional_groups)) / max(1, len(functional_groups))
            
            # Concentration analysis
            concentration_variance = np.var(concentrations) if len(concentrations) > 1 else 0
            dominant_molecule = np.argmax(concentrations) if len(concentrations) > 0 else 0
            
            return {
                'functional_groups': functional_groups,
                'unique_groups': list(set(functional_groups)),
                'interaction_strength': interaction_strength,
                'concentration_variance': float(concentration_variance),
                'dominant_molecule_idx': int(dominant_molecule),
                'num_molecules': len(smiles_list),
                'weighted_strength': interaction_strength * (1 + concentration_variance)
            }
            
        except Exception as e:
            self.logger.warning(f"Mixture analysis failed: {e}")
            return {
                'functional_groups': [],
                'unique_groups': [],
                'interaction_strength': 0.0,
                'concentration_variance': 0.0,
                'dominant_molecule_idx': 0,
                'num_molecules': len(smiles_list),
                'weighted_strength': 0.0
            }
    
    def _identify_functional_groups(self, smiles: str) -> List[str]:
        """Identify functional groups in SMILES (enhanced for whisky compounds)"""
        groups = []
        
        # Enhanced pattern matching for whisky-relevant functional groups
        smiles_upper = smiles.upper()
        
        # Aldehydes (C=O with H)
        if 'C=O' in smiles and ('CC=O' in smiles or 'O=CC' in smiles or 'CHO' in smiles):
            groups.append('aldehyde')
        
        # Esters (COC or C(=O)O)
        if 'COC' in smiles or 'C(=O)O' in smiles or '(=O)O' in smiles:
            groups.append('ester')
        
        # Alcohols (OH not in phenol)
        if 'OH' in smiles or 'CO' in smiles:
            if not ('c' in smiles and 'OH' in smiles):  # not phenolic
                groups.append('alcohol')
        
        # Phenols (aromatic OH)
        if ('c' in smiles and 'O' in smiles) or 'C1=CC=CC=C1' in smiles:
            groups.append('phenol')
        
        # Furans (5-membered ring with oxygen)
        if 'C1=COC=C1' in smiles or 'C1OC=CC=1' in smiles or 'CCCO1' in smiles:
            groups.append('furan')
        
        # Pyrazines (6-membered ring with two nitrogens)
        if 'N=C' in smiles and 'C=N' in smiles:
            groups.append('pyrazine')
        
        # Lactones (cyclic esters)
        if 'C1OC' in smiles or 'CO1' in smiles:
            groups.append('lactone')
        
        # Sulfur compounds
        if 'S' in smiles:
            groups.append('sulfur')
        
        # Nitrogen compounds (amines, etc.)
        if 'N' in smiles and 'pyrazine' not in groups:
            groups.append('amine')
        
        # Terpenes (multiple methyl branches)
        if smiles.count('C(C)') >= 2 or smiles.count('CC') >= 3:
            groups.append('terpene')
        
        # Fatty acids/esters (long carbon chains)
        if 'CCCCCC' in smiles:
            groups.append('fatty_chain')
        
        return groups if groups else ['unknown']
    
    def _apply_synergy_rules(self, scores: Dict[str, float], 
                           mixture_analysis: Dict[str, Any],
                           concentrations: np.ndarray) -> Tuple[Dict[str, float], List[Dict]]:
        """Apply synergy enhancement rules"""
        corrected_scores = scores.copy()
        log_entries = []
        
        functional_groups = mixture_analysis.get('unique_groups', [])
        print(f"[DEBUG SYNERGY] Functional groups: {functional_groups}")
        print(f"[DEBUG SYNERGY] Checking scores: {list(scores.keys())}")
        
        for descriptor, rule in self.synergy_rules.items():
            print(f"[DEBUG SYNERGY] Checking descriptor: {descriptor}")
            if descriptor in corrected_scores:
                triggers = rule.get('triggers', [])
                strength = rule.get('strength', 1.0)
                conditions = rule.get('conditions', {})
                
                print(f"[DEBUG SYNERGY] {descriptor} triggers: {triggers}")
                print(f"[DEBUG SYNERGY] {descriptor} in corrected_scores: ✓ (score: {corrected_scores[descriptor]:.3f})")
                
                # Check if triggers are present
                trigger_found = False
                for i, trigger_set in enumerate(triggers):
                    trigger_met = all(group in functional_groups for group in trigger_set)
                    print(f"[DEBUG SYNERGY] {descriptor} trigger[{i}] {trigger_set}: {'✓' if trigger_met else '✗'}")
                    
                    if trigger_met:
                        trigger_found = True
                        # Check conditions
                        min_conc = conditions.get('min_concentration', 0.1)
                        max_conc = np.max(concentrations) if len(concentrations) > 0 else 0
                        conc_met = max_conc >= min_conc
                        print(f"[DEBUG SYNERGY] {descriptor} concentration check: {max_conc:.3f} >= {min_conc} = {'✓' if conc_met else '✗'}")
                        
                        if conc_met:
                            # PHASE 3: 농도 적응성 적용
                            if self.use_concentration_adaptivity:
                                # 적응적 강도 계산
                                adaptive_strength = self._calculate_adaptive_strength(
                                    strength, max_conc, trigger_set, mixture_analysis
                                )
                            else:
                                # 기존 방식
                                adaptive_strength = strength * self.synergy_strength
                            
                            # PHASE 2: 로그 스케일 적용
                            old_score = corrected_scores[descriptor]
                            corrected_scores[descriptor] = self._apply_logarithmic_scaling(
                                old_score, adaptive_strength
                            )
                            
                            log_entries.append({
                                'type': 'synergy',
                                'descriptor': descriptor,
                                'trigger': trigger_set,
                                'strength': strength,
                                'old_score': old_score,
                                'new_score': corrected_scores[descriptor]
                            })
                            print(f"[DEBUG SYNERGY] ✓ Applied {descriptor}: {old_score:.3f} → {corrected_scores[descriptor]:.3f}")
                            break
                        else:
                            print(f"[DEBUG SYNERGY] {descriptor} concentration too low, skipping")
                
                if not trigger_found:
                    print(f"[DEBUG SYNERGY] {descriptor} no triggers matched from functional groups: {functional_groups}")
            else:
                print(f"[DEBUG SYNERGY] {descriptor} not in corrected_scores: {list(corrected_scores.keys())}")
        
        print(f"[DEBUG SYNERGY] Total synergy rules applied: {len(log_entries)}")
        return corrected_scores, log_entries
    
    def _apply_masking_rules(self, scores: Dict[str, float], 
                           mixture_analysis: Dict[str, Any],
                           concentrations: np.ndarray) -> Tuple[Dict[str, float], List[Dict]]:
        """Apply masking suppression rules"""
        corrected_scores = scores.copy()
        log_entries = []
        
        functional_groups = mixture_analysis.get('unique_groups', [])
        
        for descriptor, rule in self.masking_rules.items():
            if descriptor in corrected_scores:
                triggers = rule.get('triggers', [])
                strength = rule.get('strength', 1.0)
                conditions = rule.get('conditions', {})
                
                # Check if triggers are present
                for trigger_set in triggers:
                    triggered = False
                    if 'any' in trigger_set:
                        triggered = len(functional_groups) > 0
                    else:
                        triggered = all(group in functional_groups for group in trigger_set)
                    
                    if triggered:
                        # Check conditions
                        min_conc = conditions.get('min_concentration', 0.1)
                        if np.max(concentrations) >= min_conc:
                            # Apply suppression
                            old_score = corrected_scores[descriptor]
                            corrected_scores[descriptor] *= (strength * self.masking_strength)
                            
                            log_entries.append({
                                'type': 'masking',
                                'descriptor': descriptor,
                                'trigger': trigger_set,
                                'strength': strength,
                                'old_score': old_score,
                                'new_score': corrected_scores[descriptor]
                            })
                            break
        
        return corrected_scores, log_entries
    
    def _apply_functional_group_rules(self, scores: Dict[str, float], 
                                    mixture_analysis: Dict[str, Any],
                                    concentrations: np.ndarray,
                                    descriptor_type: str = "odor") -> Tuple[Dict[str, float], List[Dict]]:
        """Apply functional group specific rules"""
        corrected_scores = scores.copy()
        log_entries = []
        
        functional_groups = mixture_analysis.get('unique_groups', [])
        
        # Choose appropriate rule set based on descriptor type
        if descriptor_type == "taste":
            rule_set = getattr(self, 'taste_functional_group_rules', {})
        else:
            rule_set = self.functional_group_rules
        
        print(f"[DEBUG FG] Descriptor type: {descriptor_type}")
        print(f"[DEBUG FG] Functional groups: {functional_groups}")
        print(f"[DEBUG FG] Rule set keys: {list(rule_set.keys())}")
        print(f"[DEBUG FG] Corrected scores keys: {list(corrected_scores.keys())}")
        
        for group in functional_groups:
            if group in rule_set:
                group_effects = rule_set[group]
                print(f"[DEBUG FG] Group '{group}' effects: {group_effects}")
                
                for descriptor, multiplier in group_effects.items():
                    if descriptor in corrected_scores:
                        old_score = corrected_scores[descriptor]
                        
                        # PHASE 3: 농도 적응성 적용
                        if self.use_concentration_adaptivity:
                            # 해당 그룹의 농도 계산
                            group_concentration = self._calculate_group_concentration(group, functional_groups, concentrations)
                            
                            # 적응적 승수 계산
                            adaptive_multiplier = self._calculate_adaptive_multiplier(
                                multiplier, group_concentration, group, mixture_analysis, descriptor_type
                            )
                        else:
                            adaptive_multiplier = multiplier
                        
                        # PHASE 2: 로그 스케일 적용
                        corrected_scores[descriptor] = self._apply_logarithmic_scaling(old_score, adaptive_multiplier)
                        
                        log_entries.append({
                            'type': 'functional_group',
                            'descriptor': descriptor,
                            'group': group,
                            'multiplier': adaptive_multiplier,
                            'original_multiplier': multiplier,
                            'old_score': old_score,
                            'new_score': corrected_scores[descriptor],
                            'descriptor_type': descriptor_type
                        })
                        print(f"[DEBUG FG] ✓ Applied {group} → {descriptor}: {old_score:.3f} × {adaptive_multiplier:.3f} = {corrected_scores[descriptor]:.3f}")
                        if self.use_concentration_adaptivity and abs(adaptive_multiplier - multiplier) > 0.01:
                            print(f"[DEBUG FG]   농도 적응: {multiplier:.3f} → {adaptive_multiplier:.3f}")
                    else:
                        print(f"[DEBUG FG] ✗ Descriptor '{descriptor}' not in corrected_scores")
            else:
                print(f"[DEBUG FG] Group '{group}' not in rule_set")
        
        print(f"[DEBUG FG] Total functional group rules applied: {len(log_entries)}")
        return corrected_scores, log_entries
    
    def _apply_derived_feature_rules(self, scores: Dict[str, float], 
                                   mixture_analysis: Dict[str, Any],
                                   extra_features: Optional[Dict]) -> Tuple[Dict[str, float], List[Dict]]:
        """Apply derived feature rules (complex interactions)"""
        corrected_scores = scores.copy()
        log_entries = []
        
        num_molecules = mixture_analysis.get('num_molecules', 0)
        functional_groups = mixture_analysis.get('unique_groups', [])
        weighted_strength = mixture_analysis.get('weighted_strength', 0)
        
        for rule_name, rule in self.derived_feature_rules.items():
            condition = rule.get('condition', '')
            effects = rule.get('effects', {})
            
            # Enhanced condition evaluation
            condition_met = False
            
            # Concentration-based conditions
            if 'total_concentration > 0.7' in condition and weighted_strength > 0.7:
                condition_met = True
            elif 'total_concentration > 0.4' in condition and weighted_strength > 0.4:
                condition_met = True
            elif 'total_concentration > 0.5' in condition and weighted_strength > 0.5:
                condition_met = True
            
            # Molecule count conditions
            elif 'num_molecules > 10' in condition and num_molecules > 10:
                condition_met = True
            elif 'num_unique_groups > 4' in condition and len(functional_groups) > 4:
                condition_met = True
            
            # Functional group presence conditions
            elif 'ester_alcohol_combination' in condition:
                condition_met = 'ester' in functional_groups and 'alcohol' in functional_groups
            elif 'furan_aldehyde_combination' in condition:
                condition_met = 'furan' in functional_groups and 'aldehyde' in functional_groups
            elif 'phenol_present' in condition:
                condition_met = 'phenol' in functional_groups
            elif 'terpene_present' in condition:
                condition_met = 'terpene' in functional_groups
            elif 'sulfur_present' in condition:
                condition_met = 'sulfur' in functional_groups
            elif 'furan_present' in condition:
                condition_met = 'furan' in functional_groups
            
            if condition_met:
                for effect_name, effect_value in effects.items():
                    if effect_name == 'intensity_multiplier':
                        # Apply to all scores
                        for descriptor in corrected_scores:
                            old_score = corrected_scores[descriptor]
                            corrected_scores[descriptor] *= effect_value
                            
                            log_entries.append({
                                'type': 'derived_feature',
                                'rule': rule_name,
                                'descriptor': descriptor,
                                'effect': effect_name,
                                'value': effect_value,
                                'old_score': old_score,
                                'new_score': corrected_scores[descriptor]
                            })
                    
                    elif effect_name == 'complexity_boost':
                        # Apply to complex descriptors
                        complex_descriptors = ['Woody', 'Earthy', 'Spicy', 'Almond', 'Vanilla']
                        for descriptor in complex_descriptors:
                            if descriptor in corrected_scores:
                                old_score = corrected_scores[descriptor]
                                corrected_scores[descriptor] *= effect_value
                                
                                log_entries.append({
                                    'type': 'derived_feature',
                                    'rule': rule_name,
                                    'descriptor': descriptor,
                                    'effect': effect_name,
                                    'value': effect_value,
                                    'old_score': old_score,
                                    'new_score': corrected_scores[descriptor]
                                })
                    
                    elif effect_name == 'masking_factor':
                        # Apply masking to sensitive descriptors
                        sensitive_descriptors = ['Sweet', 'Floral', 'Fragrant', 'Citrus']
                        for descriptor in sensitive_descriptors:
                            if descriptor in corrected_scores:
                                old_score = corrected_scores[descriptor]
                                corrected_scores[descriptor] *= effect_value
                                
                                log_entries.append({
                                    'type': 'derived_feature',
                                    'rule': rule_name,
                                    'descriptor': descriptor,
                                    'effect': effect_name,
                                    'value': effect_value,
                                    'old_score': old_score,
                                    'new_score': corrected_scores[descriptor]
                                })
                    
                    elif effect_name in corrected_scores:
                        # Direct descriptor enhancement
                        old_score = corrected_scores[effect_name]
                        corrected_scores[effect_name] *= effect_value
                        
                        log_entries.append({
                            'type': 'derived_feature',
                            'rule': rule_name,
                            'descriptor': effect_name,
                            'effect': 'direct_enhancement',
                            'value': effect_value,
                            'old_score': old_score,
                            'new_score': corrected_scores[effect_name]
                        })
        
        return corrected_scores, log_entries
    
    def explain_rules(self, rule_log: List[Dict]) -> str:
        """Generate human-readable explanation of applied rules"""
        if not rule_log:
            return "No ontology rules were applied."
        
        explanations = []
        
        for entry in rule_log:
            rule_type = entry.get('type', 'unknown')
            descriptor = entry.get('descriptor', 'unknown')
            
            if rule_type == 'synergy':
                trigger = entry.get('trigger', [])
                explanations.append(f"Enhanced '{descriptor}' due to synergy between {', '.join(trigger)}")
            
            elif rule_type == 'masking':
                trigger = entry.get('trigger', [])
                explanations.append(f"Reduced '{descriptor}' due to masking from {', '.join(trigger)}")
            
            elif rule_type == 'functional_group':
                group = entry.get('group', 'unknown')
                explanations.append(f"Adjusted '{descriptor}' based on {group} functional group")
            
            elif rule_type == 'derived_feature':
                rule_name = entry.get('rule', 'unknown')
                explanations.append(f"Applied '{rule_name}' rule to '{descriptor}'")
            
            elif rule_type == 'score_clamp':
                clamp_type = entry.get('clamp_type', 'unknown')
                original = entry.get('original_score', 0)
                clamped = entry.get('clamped_score', 0)
                if clamp_type == 'max_clamp':
                    explanations.append(f"Clamped '{descriptor}' from {original:.2f} to max 10.0")
                elif clamp_type == 'min_clamp':
                    explanations.append(f"Clamped '{descriptor}' from {original:.2f} to min 0.0")
        
        return "; ".join(explanations)
    
    def load_rules(self, rules_file: str):
        """Load rules from JSON file"""
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            self.synergy_rules = rules_data.get('synergy_rules', {})
            self.masking_rules = rules_data.get('masking_rules', {})
            self.functional_group_rules = rules_data.get('functional_group_rules', {})
            self.derived_feature_rules = rules_data.get('derived_feature_rules', {})
            
            # Load parameters
            params = rules_data.get('parameters', {})
            self.synergy_strength = params.get('synergy_strength', 1.0)
            self.masking_strength = params.get('masking_strength', 1.0)
            self.concentration_threshold = params.get('concentration_threshold', 0.1)
            
            self.logger.info(f"Rules loaded from {rules_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load rules from {rules_file}: {e}")
            self._create_default_rules()
    
    def save_rules(self, rules_file: str):
        """Save current rules to JSON file"""
        try:
            rules_data = {
                'synergy_rules': self.synergy_rules,
                'masking_rules': self.masking_rules,
                'functional_group_rules': self.functional_group_rules,
                'derived_feature_rules': self.derived_feature_rules,
                'parameters': {
                    'synergy_strength': self.synergy_strength,
                    'masking_strength': self.masking_strength,
                    'concentration_threshold': self.concentration_threshold
                }
            }
            
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Rules saved to {rules_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save rules to {rules_file}: {e}")
    
    def update_rule_parameters(self, **kwargs):
        """Update rule application parameters"""
        if 'synergy_strength' in kwargs:
            self.synergy_strength = kwargs['synergy_strength']
        if 'masking_strength' in kwargs:
            self.masking_strength = kwargs['masking_strength']
        if 'concentration_threshold' in kwargs:
            self.concentration_threshold = kwargs['concentration_threshold']
        
        self.logger.info(f"Updated rule parameters: {kwargs}")
    
    def _clamp_scores(self, scores: Dict[str, float], min_score: float = 0.0, max_score: float = 10.0) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Clamp scores to valid range (0-10)
        
        Args:
            scores: Dictionary of descriptor scores
            min_score: Minimum allowed score (default: 0.0)
            max_score: Maximum allowed score (default: 10.0)
            
        Returns:
            Tuple of (clamped_scores, log_entries)
        """
        clamped_scores = {}
        log_entries = []
        
        for descriptor, score in scores.items():
            original_score = score
            clamped_score = max(min_score, min(max_score, score))
            
            clamped_scores[descriptor] = clamped_score
            
            # Log only if clamping occurred
            if abs(original_score - clamped_score) > 1e-6:
                clamp_type = "max_clamp" if original_score > max_score else "min_clamp"
                log_entries.append({
                    'type': 'score_clamp',
                    'descriptor': descriptor,
                    'clamp_type': clamp_type,
                    'original_score': original_score,
                    'clamped_score': clamped_score,
                    'min_allowed': min_score,
                    'max_allowed': max_score
                })
        
        if log_entries:
            self.logger.info(f"Clamped {len(log_entries)} scores to valid range [{min_score}-{max_score}]")
        
        return clamped_scores, log_entries
    
    def _apply_amplification_limits(self, corrected_scores: Dict[str, float], 
                                   original_scores: Dict[str, float]) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Phase 3 수정안 3: 증폭 제한 메커니즘
        
        Args:
            corrected_scores: 온톨로지 규칙 적용 후 점수
            original_scores: 원본 점수
            
        Returns:
            제한된 점수와 로그
        """
        limited_scores = {}
        log_entries = []
        
        # 최대 증폭률 설정 (±50%)
        max_amplification = 1.5  # 50% 증가
        min_amplification = 0.5  # 50% 감소
        
        for descriptor, corrected_score in corrected_scores.items():
            original_score = original_scores.get(descriptor, corrected_score)
            
            # 원본 점수가 0에 가까우면 제한 없이 적용
            if original_score < 0.1:
                limited_scores[descriptor] = corrected_score
                continue
            
            # 증폭률 계산
            amplification_ratio = corrected_score / original_score
            
            # 제한 적용
            if amplification_ratio > max_amplification:
                limited_score = original_score * max_amplification
                log_entries.append({
                    'type': 'amplification_limit',
                    'descriptor': descriptor,
                    'limit_type': 'max_amplification',
                    'original_score': original_score,
                    'corrected_score': corrected_score,
                    'limited_score': limited_score,
                    'amplification_ratio': amplification_ratio,
                    'max_allowed_ratio': max_amplification
                })
                limited_scores[descriptor] = limited_score
                
            elif amplification_ratio < min_amplification:
                limited_score = original_score * min_amplification
                log_entries.append({
                    'type': 'amplification_limit',
                    'descriptor': descriptor,
                    'limit_type': 'min_amplification',
                    'original_score': original_score,
                    'corrected_score': corrected_score,
                    'limited_score': limited_score,
                    'amplification_ratio': amplification_ratio,
                    'min_allowed_ratio': min_amplification
                })
                limited_scores[descriptor] = limited_score
                
            else:
                limited_scores[descriptor] = corrected_score
        
        return limited_scores, log_entries
