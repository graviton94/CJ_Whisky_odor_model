from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from features.mixture_preparator import MixtureDataPreparator
from features.feature_vector_builder import FeatureVectorBuilder

class DataProcessor:
    """분자 특성 처리 및 데이터 준비를 위한 메인 클래스"""
    
    def __init__(self, max_molecules: int = 32):
        """처리기 초기화
        
        Args:
            max_molecules: 최대 분자 수
        """
        self.preparator = MixtureDataPreparator()
        self.vector_builder = FeatureVectorBuilder(max_molecules)
    
    def prepare_mixture_data(
        self,
        smiles_list: List[str],
        peak_areas: List[float],
        mode: str = 'odor',
        csv_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
        """혼합물 예측을 위한 데이터 준비
        
        Args:
            smiles_list: SMILES 문자열 리스트
            peak_areas: peak area 리스트
            mode: 'odor' 또는 'taste'
            csv_path: 선택적 CSV 파일 경로
            
        Returns:
            intensity_matrix: 강도 행렬
            descs: 설명자 리스트
            extra_feats: 추가 특성
            extra_names: 추가 특성 이름
            detect: 검출 가능성 점수
        """
        return self.preparator.prepare_data(smiles_list, peak_areas, mode, csv_path)
    
    def build_input_vector(
        self,
        smiles_list: List[str],
        peak_areas: List[float],
        mode: str = 'odor',
        csv_path: Optional[Path] = None
    ) -> np.ndarray:
        """학습 또는 예측을 위한 입력 벡터 생성
        
        Args:
            smiles_list: SMILES 문자열 리스트
            peak_areas: peak area 리스트
            mode: 'odor' 또는 'taste'
            csv_path: 선택적 CSV 파일 경로
            
        Returns:
            feature_vector: 생성된 특성 벡터
        """
        # Get mixture data
        intensity_matrix, descs, extra_feats, _, _ = self.prepare_mixture_data(
            smiles_list, peak_areas, mode, csv_path
        )
        
        # Build feature vector
        return self.vector_builder.build_vector(
            intensity_matrix,
            extra_feats,
            np.array(peak_areas),
            descs
        )
    
# 전역 인스턴스 생성 (호환성을 위해)
_global_processor = DataProcessor()

# 호환성을 위한 함수 래퍼들
def prepare_mixture_data(smiles_list, peak_areas, mode='odor', csv_path=None):
    """혼합물 데이터 준비 (호환성 함수)"""
    return _global_processor.prepare_mixture_data(smiles_list, peak_areas, mode, csv_path)

def get_note_weight(smiles_string):
    """노트 가중치 계산 (호환성 함수)"""
    # 단일 SMILES에 대해 단일 가중치 값 반환
    if isinstance(smiles_string, str):
        # 분자량이나 복잡성 등을 고려한 가중치 계산 가능
        # 현재는 기본 가중치 1.0 반환
        return 1.0
    else:
        return 1.0

def build_input_vector(smiles_list, peak_areas, mode='odor', csv_path=None):
    """입력 벡터 생성 (호환성 함수)"""
    return _global_processor.build_input_vector(smiles_list, peak_areas, mode, csv_path)

if __name__=="__main__":
    # 테스트 코드
    processor = DataProcessor()
    
    # 실제 데이터셋의 분자들
    test_molecules = [
        ("CC(C)(C)CC(C)(C)CC(C)(C)C", 123123.0),  # 첫 번째 테스트 분자
        ("CCC1=C(C(=O)C=CO1)O", 123123.0),        # 두 번째 테스트 분자
        ("CCCCCCCC(=O)OCC(C)C", 123123123.0),     # 세 번째 테스트 분자
        ("CCCCCCCC(=O)OCC(C)C", 123123123123.0)   # 네 번째 테스트 분자
    ]
    
    smiles_list = [mol[0] for mol in test_molecules]
    peak_areas = [mol[1] for mol in test_molecules]
    
    print("\n1. 혼합물 데이터 준비 테스트:")
    matrix, descs, feats, names, detect = processor.prepare_mixture_data(smiles_list, peak_areas)
    print("Intensity matrix shape:", matrix.shape)
    print("Descriptors count:", len(descs))
    print("Extra features shape:", feats.shape)
    print("Feature names count:", len(names))
    print("Detectability shape:", detect.shape)
    
    print("\n2. 전체 입력 벡터 생성 테스트:")
    vec = processor.build_input_vector(smiles_list, peak_areas)
    print("Input vector shape:", vec.shape)
