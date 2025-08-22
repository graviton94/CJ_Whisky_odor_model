from typing import List, Tuple, Dict, Optional
from itertools import combinations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from functools import lru_cache
import logging

# Configure RDKit logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class MixtureInteractionModel:
    def __init__(self):
        """Initialize the mixture interaction model"""
        self.interaction_cache = {}
        self.ontology = {
            "odor": {
                "synergy": {
                    ("Fruity", "Sweet"): 0.449,
                    ("Fruity", "Floral"): 0.304,
                    ("Vanilla", "Almond"): 0.157,
                    ("Citrus", "Fruity"): 0.064,
                    ("Woody", "Earthy"): 0.026
                },
                "masking": {
                    ("Green", "Sweet"): -0.197,
                    ("Green", "Floral"): -0.301,
                    ("Earthy", "Sweet"): -0.100,
                    ("Earthy", "Fruity"): -0.232,
                    ("Spicy", "Vanilla"): -0.260
                }
            },
            "taste": {
                "synergy": {
                    ("Taste_Fruity", "Taste_Sweet"): 0.187,
                    ("Taste_Sour", "Taste_Fruity"): 0.213,
                    ("Taste_Nutty", "Taste_OffFlavor"): 0.169
                },
                "masking": {
                    ("Taste_Bitter", "Taste_Sweet"): -0.019,
                    ("Taste_Bitter", "Taste_Fruity"): -0.021,
                    ("Taste_OffFlavor", "Taste_Fruity"): -0.124
                }
            }
        }
        
    def calculate_interactions(self, molecules: List[Tuple[str, float]], mode: str = "odor") -> Dict:
        """분자간 상호작용 분석
        Args:
            molecules: List of (SMILES, peak_area) tuples
            mode: "odor" or "taste"
        Returns:
            Dictionary containing interaction data and scores
        """
        # 단일 분자 처리
        if len(molecules) == 1:
            smiles, conc = molecules[0]
            normalized_conc = self._normalize_concentration(conc)
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 단일 분자의 경우 단순한 점수만 반환
            base_score = {
                'pair_scores': {},
                'total_score': normalized_conc,
                'ontology_rules': self.ontology[mode]
            }
            
            return {
                'interactions': [],
                'scores': base_score,
                'mode': mode
            }
            
        # 여러 분자 처리
        normalized_molecules = [(smiles, self._normalize_concentration(conc)) 
                              for smiles, conc in molecules]
        
        # 리스트를 문자열로 변환하여 캐시 키 생성
        cache_key = f"{mode}|" + "|".join(f"{smiles}:{conc}" 
            for smiles, conc in sorted(normalized_molecules))
        if cache_key in self.interaction_cache:
            return self.interaction_cache[cache_key]
            
        interactions = []
        for (mol1_smiles, conc1), (mol2_smiles, conc2) in combinations(molecules, 2):
            # 분자 준비
            mol1 = Chem.AddHs(Chem.MolFromSmiles(mol1_smiles))
            mol2 = Chem.AddHs(Chem.MolFromSmiles(mol2_smiles))
            
            # 3D 구조 생성
            AllChem.EmbedMolecule(mol1, randomSeed=42)
            AllChem.EmbedMolecule(mol2, randomSeed=42)
            
            # 구조 최적화
            AllChem.MMFFOptimizeMolecule(mol1)
            AllChem.MMFFOptimizeMolecule(mol2)
            
            # 상호작용 분석
            interaction_data = {
                'pair': (mol1_smiles, mol2_smiles),
                'concentrations': (conc1, conc2),
                'distance': self._calculate_molecular_distance(mol1, mol2),
                'electronic': self._calculate_electronic_interactions(mol1, mol2),
                'hbonds': self._analyze_hbond_network(mol1, mol2),
                'steric': self._analyze_steric_effects(mol1, mol2)
            }
            
            interactions.append(interaction_data)
        
            # Calculate total interaction scores with ontology rules
        interaction_scores = self._calculate_total_interaction_score(interactions, mode)
            
        # 결과 캐싱
        cache_result = {
            'interactions': interactions,
            'scores': {k: v for k, v in interaction_scores.items() if k != 'mode'},
            'mode': mode
        }
        self.interaction_cache[cache_key] = cache_result
        return cache_result
    
    def _calculate_molecular_distance(self, mol1, mol2) -> float:
        """분자간 최소 거리 계산"""
        conf1 = mol1.GetConformer()
        conf2 = mol2.GetConformer()
        
        min_dist = float('inf')
        for atom1 in mol1.GetAtoms():
            pos1 = conf1.GetAtomPosition(atom1.GetIdx())
            for atom2 in mol2.GetAtoms():
                pos2 = conf2.GetAtomPosition(atom2.GetIdx())
                dist = np.sqrt(
                    (pos1.x - pos2.x)**2 + 
                    (pos1.y - pos2.y)**2 + 
                    (pos1.z - pos2.z)**2
                )
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _calculate_electronic_interactions(self, mol1, mol2) -> Dict:
        """전자적 상호작용 계산"""
        # 분자의 전자적 특성 계산
        dipole = self._calculate_dipole_interaction(mol1, mol2)
        pi_stack = self._detect_pi_stacking(mol1, mol2)
        charge = self._estimate_charge_transfer(mol1, mol2)
        
        return {
            'dipole_interaction': dipole,
            'pi_stacking': pi_stack,
            'charge_transfer': charge
        }

    def _calculate_dipole_interaction(self, mol1, mol2) -> float:
        """쌍극자 상호작용 강도 추정"""
        # RDKit의 MMFF94 전하를 사용한 간단한 추정
        charge_sum1 = sum(atom.GetDoubleProp('_MMFF94Charge') 
                         if atom.HasProp('_MMFF94Charge') else 0 
                         for atom in mol1.GetAtoms())
        charge_sum2 = sum(atom.GetDoubleProp('_MMFF94Charge')
                         if atom.HasProp('_MMFF94Charge') else 0
                         for atom in mol2.GetAtoms())
        return abs(charge_sum1 * charge_sum2) / 100  # Normalized to [0,1]

    def _detect_pi_stacking(self, mol1, mol2) -> float:
        """π-π 스태킹 가능성 검출"""
        # 방향족 고리 수 계산 (NumAtomRings 사용)
        rings1 = sum(1 for ring in mol1.GetRingInfo().AtomRings() 
                    if all(mol1.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        rings2 = sum(1 for ring in mol2.GetRingInfo().AtomRings()
                    if all(mol2.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        
        if rings1 == 0 or rings2 == 0:
            return 0.0
            
        # 거리에 기반한 스태킹 가능성
        dist = self._calculate_molecular_distance(mol1, mol2)
        if dist > 5.0:  # 5 Å 이상이면 스태킹 불가능
            return 0.0
            
        # 방향족 고리 수와 거리에 기반한 점수
        return min(rings1, rings2) * (1 - dist/5.0)

    def _estimate_charge_transfer(self, mol1, mol2) -> float:
        """전하 이동 가능성 추정"""
        # HOMO-LUMO gap 추정을 위한 단순화된 계산
        # 전자 공여체/수용체 그룹 수 계산
        donors1 = sum(1 for atom in mol1.GetAtoms() 
                     if atom.GetSymbol() in ['N', 'O', 'S'])
        acceptors1 = sum(1 for atom in mol1.GetAtoms()
                        if atom.GetFormalCharge() < 0)
        
        donors2 = sum(1 for atom in mol2.GetAtoms()
                     if atom.GetSymbol() in ['N', 'O', 'S'])
        acceptors2 = sum(1 for atom in mol2.GetAtoms()
                        if atom.GetFormalCharge() < 0)
        
        # 전하 이동 가능성 점수
        transfer_potential = (donors1 * acceptors2 + donors2 * acceptors1) / \
                           (mol1.GetNumAtoms() * mol2.GetNumAtoms())
        return min(transfer_potential, 1.0)  # Normalized to [0,1]
    
    def _analyze_hbond_network(self, mol1, mol2) -> Dict:
        """수소 결합 네트워크 분석"""
        # 수소 결합 공여체/수용체 식별
        hbond_donors1 = self._count_hbond_donors(mol1)
        hbond_acceptors1 = self._count_hbond_acceptors(mol1)
        hbond_donors2 = self._count_hbond_donors(mol2)
        hbond_acceptors2 = self._count_hbond_acceptors(mol2)
        
        # 거리에 따른 수소 결합 강도 추정
        dist = self._calculate_molecular_distance(mol1, mol2)
        dist_factor = max(0, 1 - dist/3.5)  # 3.5 Å 이상이면 0
        
        # 수소 결합 점수 계산
        hbond_score = dist_factor * min(
            hbond_donors1 * hbond_acceptors2,
            hbond_donors2 * hbond_acceptors1
        )
        
        return {
            'donors1': hbond_donors1,
            'acceptors1': hbond_acceptors1,
            'donors2': hbond_donors2,
            'acceptors2': hbond_acceptors2,
            'strength': min(hbond_score, 1.0)  # Normalized to [0,1]
        }
        
    def _count_hbond_donors(self, mol) -> int:
        """수소 결합 공여체 수 계산"""
        return sum(1 for atom in mol.GetAtoms()
                  if atom.GetSymbol() in ['N', 'O'] and
                  any(n.GetSymbol() == 'H' for n in atom.GetNeighbors()))
        
    def _count_hbond_acceptors(self, mol) -> int:
        """수소 결합 수용체 수 계산"""
        return sum(1 for atom in mol.GetAtoms()
                  if atom.GetSymbol() in ['N', 'O', 'F'])
        """수소결합 네트워크 분석"""
        
    def _normalize_concentration(self, peak_area: float) -> float:
        """Peak area를 0-1 사이의 농도값으로 정규화
        Args:
            peak_area: GC-MS peak area value
        Returns:
            Normalized concentration value between 0 and 1
        """
        # From mixture_trials_learn.jsonl analysis
        # Using log scaling due to wide range of peak areas
        log_area = np.log10(peak_area + 1)  # +1 to handle zero values
        # Observed range in data: ~2-9 in log scale
        normalized = (log_area - 2) / (9 - 2)
        return np.clip(normalized, 0, 1)

    
    def _analyze_steric_effects(self, mol1, mol2) -> Dict:
        """입체 효과 분석"""
        # 분자 부피 추정
        vol1 = self._estimate_molecular_volume(mol1)
        vol2 = self._estimate_molecular_volume(mol2)
        
        # 거리 기반 겹침 정도
        dist = self._calculate_molecular_distance(mol1, mol2)
        overlap = max(0, 1 - dist/(vol1 + vol2))
        
        # 회전 가능한 결합 수
        rotatable1 = AllChem.CalcNumRotatableBonds(mol1)
        rotatable2 = AllChem.CalcNumRotatableBonds(mol2)
        
        return {
            'volume1': vol1,
            'volume2': vol2,
            'overlap': overlap,
            'rotatable_bonds1': rotatable1,
            'rotatable_bonds2': rotatable2,
            'flexibility': (rotatable1 + rotatable2) / \
                         (mol1.GetNumBonds() + mol2.GetNumBonds())
        }
        
    def _estimate_molecular_volume(self, mol) -> float:
        """분자 부피 추정 (간단한 원자 부피 합)"""
        # 원자별 반데르발스 반지름 (Å)
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52,
            'F': 1.47, 'P': 1.8, 'S': 1.8, 'Cl': 1.75,
            'Br': 1.85, 'I': 1.98
        }
        
        volume = 0.0
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            radius = vdw_radii.get(symbol, 1.5)  # Default radius if unknown
            # 구 부피 공식: V = (4/3)πr³
            volume += (4/3) * np.pi * radius**3
            
        return volume

    def _calculate_total_interaction_score(self, interactions: List[Dict], mode: str = "odor") -> Dict:
        """상호작용 점수 계산 (온톨로지 규칙 적용)
        Args:
            interactions: List of interaction data dictionaries
            mode: "odor" or "taste", defaults to "odor"
        Returns:
            Dictionary containing pair scores, total score, ontology rules, and mode
        """
        pair_scores = {}
        ontology = self.ontology[mode]
        
        if not interactions:  # 빈 상호작용 리스트 (단일 분자)
            return {
                'pair_scores': {},
                'total_score': 0.0,
                'ontology_rules': ontology,
                'mode': mode
            }
        
        # 여러 분자 간 상호작용 계산
        for interaction in interactions:
            mol1, mol2 = interaction['pair']
            conc1, conc2 = interaction['concentrations']
            
            # 기본 상호작용 점수 계산
            base_effect = (
                interaction['electronic']['dipole_interaction'] * 0.4 +
                interaction['electronic']['pi_stacking'] * 0.3 +
                interaction['hbonds'].get('strength', 0) * 0.2 +
                interaction['steric'].get('overlap', 0) * 0.1
            )
            
            # 온톨로지 규칙 적용
            rule_effect = sum(weight for _, weight in ontology["synergy"].items())
            rule_effect += sum(weight for _, weight in ontology["masking"].items())
            
            # 최종 점수 계산
            total_effect = base_effect * (1 + rule_effect)
            weight = np.sqrt(conc1 * conc2)  # geometric mean of concentrations
            
            # 튜플 키를 문자열로 변환하여 딕셔너리 비교 문제 해결
            pair_key = f"{mol1}_{mol2}"
            pair_scores[pair_key] = float(total_effect * weight)
        
        return {
            'pair_scores': pair_scores,
            'total_score': float(sum(pair_scores.values())),
            'ontology_rules': ontology,
            'mode': mode
        }
