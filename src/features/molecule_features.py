from typing import List, Tuple, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import DataStructs
from features.config import FP_DIM

class MoleculeFeatureExtractor:
    """분자 특성 추출을 위한 클래스"""
    
    @staticmethod
    def compute_rdkit_features(smiles_list: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
        """RDKit 기반 분자 특성 계산
        
        Args:
            smiles_list: SMILES 문자열 리스트
            
        Returns:
            features: 계산된 특성 배열
            fmap: 특성 이름과 인덱스 매핑
        """
        descriptor_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            
                # Calculate descriptors
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            hdon = Descriptors.NumHDonors(mol)
            hacc = Descriptors.NumHAcceptors(mol)
            
            # 논문 기반 확장 디스크립터 (호환성 고려)
            tpsa = Descriptors.TPSA(mol)  # Topological Polar Surface Area
            rotbonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            heteroatoms = Descriptors.NumHeteroatoms(mol)
            
            # SP3 탄소 비율 (버전 호환성 처리)
            try:
                sp3_carbon = Descriptors.FractionCsp3(mol)
            except AttributeError:
                # 구버전 RDKit에서는 수동 계산
                try:
                    sp3_carbon = rdMolDescriptors.CalcFractionCsp3(mol)
                except:
                    sp3_carbon = 0.5  # 기본값
            
            # 향미 예측에 중요한 구조적 특성
            try:
                saturation = Descriptors.BertzCT(mol)  # 구조 복잡도
            except AttributeError:
                saturation = mw / 10.0  # 대체 계산
                
            flexibility = rotbonds / (mw + 1e-8)  # 유연성 지수
            polarity_ratio = tpsa / (mw + 1e-8)  # 극성 비율
            
            descriptor_list.append([
                logp, mw, hdon, hacc, tpsa, rotbonds, 
                aromatic_rings, heteroatoms, sp3_carbon, 
                saturation, flexibility, polarity_ratio
            ])
            
        features = np.array(descriptor_list, dtype=np.float32)
        fmap = {
            'LogP': 0, 'MolWt': 1, 'NumHDonors': 2, 'NumHAcceptors': 3,
            'TPSA': 4, 'NumRotatableBonds': 5, 'NumAromaticRings': 6,
            'NumHeteroatoms': 7, 'FractionCsp3': 8, 'BertzCT': 9,
            'FlexibilityIndex': 10, 'PolarityRatio': 11
        }
        return features, fmap
    
    @staticmethod
    def compute_fingerprint(mol: Chem.Mol, radius: int = 2, nBits: int = FP_DIM) -> np.ndarray:
        """Morgan fingerprint 계산
        
        Args:
            mol: RDKit 분자 객체
            radius: Morgan fingerprint radius
            nBits: fingerprint 비트 수
            
        Returns:
            arr: fingerprint 배열
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


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
    
    print("분자 특성 추출 테스트\n")
    
    # 특성 추출기 초기화
    extractor = MoleculeFeatureExtractor()
    
    # RDKit 특성 테스트
    print("1. RDKit 분자 특성:")
    features, feature_map = extractor.compute_rdkit_features(test_molecules)
    print(f"특성 행렬 형태: {features.shape}")
    print(f"특성 맵핑: {feature_map}\n")
    print("각 분자의 주요 특성:")
    for i, smiles in enumerate(test_molecules):
        print(f"\n분자 {i+1} ({smiles}):")
        print(f"  LogP: {features[i][feature_map['LogP']]:.2f}")
        print(f"  분자량: {features[i][feature_map['MolWt']]:.2f}")
        print(f"  수소결합 공여체 수: {features[i][feature_map['NumHDonors']]}")
        print(f"  수소결합 수용체 수: {features[i][feature_map['NumHAcceptors']]}")
        print(f"  회전 가능한 결합 수: {features[i][feature_map['NumRotatableBonds']]}")
    
    # Fingerprint 테스트
    print("\n2. Morgan Fingerprint:")
    mol = Chem.MolFromSmiles(test_molecules[0])
    fp = extractor.compute_fingerprint(mol)
    print(f"Fingerprint 길이: {len(fp)}")
    print(f"비트 분포 (1의 개수): {np.sum(fp)}")
