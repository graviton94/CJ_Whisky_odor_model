"""
향미별 특화 분자 특성 계산기
논문 "Odor prediction of whiskies based on their molecular composition" 기반
"""

from typing import List, Dict, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

class OdorSpecificFeatureExtractor:
    """향미 예측을 위한 특화 분자 특성 추출기"""
    
    # 향미별 중요 분자 특성 가중치 (논문 기반)
    ODOR_FEATURE_WEIGHTS = {
        'Fruity': {
            'ester_count': 1.5,
            'acetate_presence': 1.3,
            'chain_length': 0.8,
            'branching': 0.6
        },
        'Woody': {
            'phenol_presence': 1.4,
            'aromatic_rings': 1.2,
            'terpene_structure': 1.1,
            'molecular_weight': 0.7
        },
        'Sweet': {
            'furan_presence': 1.3,
            'lactone_presence': 1.2,
            'alcohol_groups': 0.9,
            'polarity': 0.8
        },
        'Floral': {
            'indole_presence': 1.4,
            'benzene_rings': 1.1,
            'ester_variety': 1.0,
            'volatility': 0.9
        }
    }
    
    @staticmethod
    def calculate_odor_potency(smiles: str, peak_area: float) -> Dict[str, float]:
        """분자의 향미별 기여도 계산
        
        Args:
            smiles: SMILES 문자열
            peak_area: 피크 면적 (농도)
            
        Returns:
            odor_scores: 향미별 점수 딕셔너리
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {odor: 0.0 for odor in OdorSpecificFeatureExtractor.ODOR_FEATURE_WEIGHTS.keys()}
        
        # 기본 분자 특성
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        
        # 휘발성 계산 (향미 강도에 중요)
        volatility = np.exp(-mw/200) * (logp + 1)
        
        # 농도 보정 (peak_area 활용)
        concentration_factor = np.log10(peak_area + 1) / 8.0  # 정규화
        
        odor_scores = {}
        
        # Fruity 점수 계산
        ester_pattern = Chem.MolFromSmarts('C(=O)OC')
        acetate_pattern = Chem.MolFromSmarts('CC(=O)O')
        ester_count = len(mol.GetSubstructMatches(ester_pattern))
        acetate_presence = mol.HasSubstructMatch(acetate_pattern)
        
        fruity_score = (
            ester_count * 1.5 + 
            acetate_presence * 1.3 + 
            volatility * 0.8
        ) * concentration_factor
        odor_scores['Fruity'] = fruity_score
        
        # Woody 점수 계산
        phenol_pattern = Chem.MolFromSmarts('c1ccc(cc1)O')
        phenol_presence = mol.HasSubstructMatch(phenol_pattern)
        
        woody_score = (
            phenol_presence * 1.4 + 
            aromatic_rings * 1.2 + 
            (mw > 150) * 0.7
        ) * concentration_factor
        odor_scores['Woody'] = woody_score
        
        # Sweet 점수 계산
        furan_pattern = Chem.MolFromSmarts('c1ccoc1')
        lactone_pattern = Chem.MolFromSmarts('[O;R1][C;R1](=O)[C;R1]')
        alcohol_pattern = Chem.MolFromSmarts('[OX2H]')
        
        furan_presence = mol.HasSubstructMatch(furan_pattern)
        lactone_presence = mol.HasSubstructMatch(lactone_pattern)
        alcohol_count = len(mol.GetSubstructMatches(alcohol_pattern))
        
        sweet_score = (
            furan_presence * 1.3 + 
            lactone_presence * 1.2 + 
            alcohol_count * 0.9 + 
            (tpsa / mw) * 0.8
        ) * concentration_factor
        odor_scores['Sweet'] = sweet_score
        
        # Floral 점수 계산
        indole_pattern = Chem.MolFromSmarts('c1ccc2[nH]ccc2c1')
        benzene_pattern = Chem.MolFromSmarts('c1ccccc1')
        
        indole_presence = mol.HasSubstructMatch(indole_pattern)
        benzene_rings = len(mol.GetSubstructMatches(benzene_pattern))
        
        floral_score = (
            indole_presence * 1.4 + 
            benzene_rings * 1.1 + 
            ester_count * 1.0 + 
            volatility * 0.9
        ) * concentration_factor
        odor_scores['Floral'] = floral_score
        
        return odor_scores
    
    @staticmethod
    def calculate_synergy_potential(smiles1: str, smiles2: str) -> float:
        """두 분자간 시너지 효과 잠재력 계산
        
        Args:
            smiles1, smiles2: SMILES 문자열들
            
        Returns:
            synergy_score: 시너지 잠재력 점수
        """
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # 분자 유사성 계산
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        similarity = Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
        
        # 상호보완적 특성 계산
        mw1, mw2 = Descriptors.MolWt(mol1), Descriptors.MolWt(mol2)
        logp1, logp2 = Descriptors.MolLogP(mol1), Descriptors.MolLogP(mol2)
        
        # 크기 상호보완성 (한쪽은 크고 한쪽은 작을 때 시너지)
        size_complement = abs(mw1 - mw2) / (mw1 + mw2)
        
        # 극성 상호보완성
        polarity_complement = abs(logp1 - logp2) / (abs(logp1) + abs(logp2) + 1)
        
        # 최종 시너지 점수 (유사하지만 상호보완적일 때 높음)
        synergy_score = similarity * (size_complement + polarity_complement) / 2
        
        return synergy_score
    
    @staticmethod
    def extract_all_odor_features(molecules_data: List[Dict]) -> np.ndarray:
        """모든 분자의 향미별 특성 추출
        
        Args:
            molecules_data: [{'smiles': str, 'peak_area': float}, ...]
            
        Returns:
            feature_matrix: 향미별 특성 행렬
        """
        all_features = []
        
        for mol_data in molecules_data:
            smiles = mol_data['smiles']
            peak_area = mol_data.get('peak_area', 1.0)
            
            odor_scores = OdorSpecificFeatureExtractor.calculate_odor_potency(smiles, peak_area)
            feature_vector = list(odor_scores.values())
            all_features.append(feature_vector)
        
        return np.array(all_features, dtype=np.float32)


if __name__ == "__main__":
    # 테스트 데이터
    test_molecules = [
        {'smiles': 'CCCCCCCC(=O)OCC', 'peak_area': 1000000},  # Ethyl octanoate (fruity)
        {'smiles': 'c1ccc(cc1)O', 'peak_area': 500000},       # Phenol (woody)
        {'smiles': 'c1ccoc1', 'peak_area': 750000},           # Furan (sweet)
        {'smiles': 'CC(=O)OCC1=CC=CC=C1', 'peak_area': 300000}  # Benzyl acetate (floral)
    ]
    
    print("향미별 특화 분자 특성 테스트\n")
    
    extractor = OdorSpecificFeatureExtractor()
    
    for i, mol_data in enumerate(test_molecules):
        print(f"분자 {i+1}: {mol_data['smiles']}")
        odor_scores = extractor.calculate_odor_potency(
            mol_data['smiles'], 
            mol_data['peak_area']
        )
        
        for odor, score in odor_scores.items():
            print(f"  {odor}: {score:.3f}")
        print()
    
    # 시너지 테스트
    print("시너지 효과 테스트:")
    synergy = extractor.calculate_synergy_potential(
        test_molecules[0]['smiles'],  # Ethyl octanoate
        test_molecules[3]['smiles']   # Benzyl acetate
    )
    print(f"Ethyl octanoate + Benzyl acetate 시너지: {synergy:.3f}")
