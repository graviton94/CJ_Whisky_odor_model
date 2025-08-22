from typing import List, Tuple
import numpy as np
from features.statistical_features import StatisticalFeatureCalculator

class FeatureVectorBuilder:
    """특성 벡터 생성을 담당하는 클래스"""
    
    def __init__(self, max_molecules: int = 32):
        """
        Args:
            max_molecules: 최대 분자 수
        """
        self.MAX_MOL = max_molecules
        self.stat_calculator = StatisticalFeatureCalculator()
    
    def build_vector(
        self,
        intensity_matrix: np.ndarray,
        extra_features: np.ndarray,
        peak_areas: np.ndarray,
        descriptors: List[str]
    ) -> np.ndarray:
        """특성 벡터 생성
        
        Args:
            intensity_matrix: 강도 행렬
            extra_features: 추가 특성
            peak_areas: peak area 배열
            descriptors: 설명자 리스트
            
        Returns:
            특성 벡터
        """
        feat_vec = []
        
        # 1. 평균 강도 특성
        intensity_means = np.mean(intensity_matrix, axis=0)
        feat_vec.extend(intensity_means)
        
        # 2. 통계적 특성
        stat_feats, _ = self.stat_calculator.compute_statistical_features(
            intensity_matrix, descriptors
        )
        feat_vec.extend(stat_feats)
        
        # 3. 추가 특성
        if extra_features is not None:
            extra_means = np.mean(extra_features, axis=0)
            feat_vec.extend(extra_means)
        
        # 4. 검출 가능성 패딩
        n_mol = len(peak_areas)
        if n_mol < self.MAX_MOL:
            padding = np.zeros(self.MAX_MOL - n_mol)
            feat_vec.extend(padding)
        
        # 5. Peak ratio 패딩
        if n_mol < self.MAX_MOL:
            padding = np.zeros(self.MAX_MOL - n_mol)
            feat_vec.extend(padding)
            
        return np.array(feat_vec, dtype=np.float32)
