from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem.Crippen import MolLogP
from features.config import FP_DIM

class PhysicalPropertyCalculator:
    """물리적 특성 계산을 위한 클래스"""
    
    @staticmethod
    def get_note_weight(smiles: str) -> float:
        """분자의 Note weight 계산
        
        Args:
            smiles: SMILES 문자열
            
        Returns:
            weight: 계산된 weight 값
        """
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol)
        logp = MolLogP(mol)
        vol_index = logp / (mw + 1e-8)
        
        if mw <= 160 and vol_index >= 0.015:
            return 1.3
        elif mw <= 260 and vol_index >= 0.008:
            return 1.0
        else:
            return 0.7
    
    @staticmethod
    def compute_volatility(smiles_list: List[str]) -> np.ndarray:
        """휘발성 지수 계산
        
        Args:
            smiles_list: SMILES 문자열 리스트
            
        Returns:
            vol_array: 휘발성 지수 배열
        """
        vol_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
                
            logp = MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            vol_list.append([logp/(mw+1e-8)])
            
        return np.array(vol_list, dtype=np.float32)
    
    @staticmethod
    def compute_avg_tanimoto(smiles_list: List[str]) -> np.ndarray:
        """평균 Tanimoto 유사도 계산
        
        Args:
            smiles_list: SMILES 문자열 리스트
            
        Returns:
            sim_array: 평균 유사도 배열
        """
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smi), 2, FP_DIM
            )
            for smi in smiles_list
        ]
        
        n = len(fps)
        sim = np.zeros((n,), dtype=np.float32)
        
        for i in range(n):
            if n == 1:
                sim[i] = 0.0
            else:
                sims = [
                    DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    for j in range(n) if j != i
                ]
                sim[i] = np.mean(sims)
                
        return sim.reshape(-1, 1)
    
    @staticmethod
    def compute_detectabilities(smiles_list: List[str]) -> np.ndarray:
        """검출 가능성 점수 계산
        
        Args:
            smiles_list: SMILES 문자열 리스트
            
        Returns:
            det_array: 검출 가능성 점수 배열
        """
        vol = PhysicalPropertyCalculator.compute_volatility(smiles_list)
        vmin, vmax = vol.min(), vol.max()
        
        if vmax - vmin < 1e-6:
            det = np.ones_like(vol)
        else:
            det = (vol - vmin)/(vmax - vmin)
            
        return np.clip(det, 0.05, 0.95).astype(np.float32)


if __name__ == "__main__":
    # 다양한 구조의 테스트 분자들
    test_molecules = [
        "CC(C)(C)CC(C)(C)CC(C)(C)C",     # 알킬 사슬 (branched alkane)
        "CCC1=C(C(=O)C=CO1)O",           # 퓨란 고리 + 히드록시 + 케톤
        "CCCCCCCC(=O)OCC(C)C",           # 긴 사슬 에스테르
        "O=CC1=CC=CO1",                  # 퓨르푸랄 (furanic aldehyde)
        "CC(=O)OCC1=CC=CC=C1",           # 페닐 아세테이트
        "CC1=CC=C(C=C1)O"                # 크레졸 (메틸페놀)
    ]
    
    print("물리적 특성 계산 테스트\n")
    
    # 계산기 초기화
    calculator = PhysicalPropertyCalculator()
    
    # 1. 휘발성 계산
    print("1. 휘발성 지수:")
    volatility = calculator.compute_volatility(test_molecules)
    for i, (smiles, vol) in enumerate(zip(test_molecules, volatility)):
        print(f"분자 {i+1} ({smiles}): {vol:.4f}")
    
    # 2. Tanimoto 유사도
    print("\n2. 평균 Tanimoto 유사도:")
    similarity = calculator.compute_avg_tanimoto(test_molecules)
    for i, (smiles, sim) in enumerate(zip(test_molecules, similarity)):
        print(f"분자 {i+1} ({smiles}): {sim:.4f}")
    
    # 3. 검출 가능성 점수
    print("\n3. 검출 가능성 점수:")
    detect = calculator.compute_detectabilities(test_molecules)
    for i, (smiles, det) in enumerate(zip(test_molecules, detect)):
        print(f"분자 {i+1} ({smiles}): {det:.4f}")
    
    # 4. Note weight
    print("\n4. Note Weight:")
    for i, smiles in enumerate(test_molecules):
        weight = calculator.get_note_weight(smiles)
        print(f"분자 {i+1} ({smiles}): {weight:.2f}")
