from typing import List, Tuple
import numpy as np
from rdkit import Chem

class FunctionalGroupAnalyzer:
    """화학 작용기 분석을 위한 클래스"""
    
    SMARTS_FUNCTIONAL_GROUPS = {
        # 기존 기본 작용기
        'Ester': 'C(=O)OC',
        'Alcohol': '[OX2H]',
        'Ketone': 'C(=O)[!O]',
        'Aldehyde': '[CX3H1](=O)[#6]',
        'CarboxylicAcid': 'C(=O)[O-]',
        'Ether': 'C-O-C',
        
        # 위스키 향미 관련 특화 작용기 (논문 기반)
        'Phenol': 'c1ccc(cc1)O',  # Woody, Smoky 노트
        'Furan': 'c1ccoc1',  # Sweet, Caramel 노트
        'Lactone': '[O;R1][C;R1](=O)[C;R1]',  # Coconut, Vanilla 노트
        'Terpene': 'CC(C)=CCC',  # Herbal, Pine 노트
        'Pyrazine': 'c1cnccn1',  # Nutty, Roasted 노트
        'Thiophene': 'c1ccsc1',  # Sulfur, Meaty 노트
        'Indole': 'c1ccc2[nH]ccc2c1',  # Floral 노트
        'Benzofuran': 'c1ccc2occc2c1',  # Almond 노트
        
        # 지방족 특성
        'LongChain': 'CCCCCCCC',  # Fatty, Waxy 노트
        'Branched': 'CC(C)C',  # 분지형 구조
        'Aromatic_Ring': 'c1ccccc1',  # 방향족 특성
        'Acetate': 'CC(=O)O',  # Fruity 노트
    }
    
    def __init__(self):
        """작용기 SMARTS 패턴 초기화"""
        self._patterns = {
            name: Chem.MolFromSmarts(smarts)
            for name, smarts in self.SMARTS_FUNCTIONAL_GROUPS.items()
        }
    
    def compute_functional_groups(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        """분자의 작용기 존재 여부 분석
        
        Args:
            smiles_list: SMILES 문자열 리스트
            
        Returns:
            group_vectors: 작용기 존재 여부 배열
            group_names: 작용기 이름 리스트
        """
        group_names = list(self._patterns.keys())
        group_vectors = []
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
                
            vec = [
                int(bool(mol.HasSubstructMatch(pattern)))
                for pattern in self._patterns.values()
            ]
            group_vectors.append(vec)
            
        return np.array(group_vectors, dtype=np.float32), group_names


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
    
    print("작용기 분석 테스트\n")
    
    # 작용기 분석기 초기화
    analyzer = FunctionalGroupAnalyzer()
    
    # 작용기 분석 실행
    group_vectors, group_names = analyzer.compute_functional_groups(test_molecules)
    
    print("작용기 분석 결과:")
    for i, smiles in enumerate(test_molecules):
        print(f"\n분자 {i+1} ({smiles}):")
        for j, name in enumerate(group_names):
            if group_vectors[i][j]:
                print(f"  - {name} 작용기 존재")
                
    print(f"\n총 분석된 작용기 종류: {len(group_names)}")
    print(f"분자별 작용기 분포 행렬 형태: {group_vectors.shape}")
