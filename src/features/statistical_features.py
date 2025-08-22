from typing import List, Tuple
import numpy as np

class StatisticalFeatureCalculator:
    """통계적 특성 계산을 위한 클래스"""
    
    @staticmethod
    def compute_statistical_features(
        intensity_matrix: np.ndarray,
        descs: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """강도 행렬로부터 통계적 특성 계산
        
        Args:
            intensity_matrix: (n_mol, n_desc) 형태의 강도 행렬
            descs: 설명자 이름 리스트
            
        Returns:
            features: 계산된 통계적 특성
            feature_names: 특성 이름 리스트
        """
        means = np.mean(intensity_matrix, axis=0)
        variances = np.var(intensity_matrix, axis=0)
        maxima = np.max(intensity_matrix, axis=0)
        minima = np.min(intensity_matrix, axis=0)
        stds = np.std(intensity_matrix, axis=0)
        skew = ((intensity_matrix - means)**3).mean(axis=0)/(stds**3 + 1e-8)
        
        feats = np.concatenate([variances, maxima, minima, skew])
        names = (
            [f"var_{d}" for d in descs] +
            [f"max_{d}" for d in descs] +
            [f"min_{d}" for d in descs] +
            [f"skew_{d}" for d in descs]
        )
        
        return feats, names
    
    @staticmethod
    def compute_derived_descriptors(mol_props: dict) -> Tuple[np.ndarray, List[str]]:
        """분자 특성으로부터 유도 설명자 계산
        
        Args:
            mol_props: 분자 특성 딕셔너리
                'mw': 분자량 배열
                'logp': LogP 배열
                'hdon': 수소결합 공여체 수 배열
                'hacc': 수소결합 수용체 수 배열
                
        Returns:
            derived: 유도 설명자 배열 (N x 2)
            names: 설명자 이름 리스트
        """
        mw = mol_props['mw']
        logp = mol_props['logp']
        hdon = mol_props['hdon']
        hacc = mol_props['hacc']
        
        derived = np.column_stack([
            mw * logp,              # MW*LogP
            hdon / (hacc + 1e-8)    # HBD/HBA ratio
        ])
        
        names = ['MWxLogP', 'HBD_HBA_ratio']
        return derived, names


if __name__ == "__main__":
    # 다양한 구조의 테스트 분자들에 대한 가상의 강도 데이터
    test_intensity_matrix = np.array([
        [0.8, 0.2, 0.5, 0.3, 0.9],  # 첫 번째 분자의 강도
        [0.4, 0.7, 0.3, 0.8, 0.2],  # 두 번째 분자의 강도
        [0.6, 0.3, 0.9, 0.4, 0.5],  # 세 번째 분자의 강도
        [0.2, 0.8, 0.4, 0.6, 0.7],  # 네 번째 분자의 강도
        [0.7, 0.5, 0.6, 0.2, 0.4],  # 다섯 번째 분자의 강도
        [0.3, 0.6, 0.2, 0.7, 0.8]   # 여섯 번째 분자의 강도
    ])
    
    test_descriptors = ['Sweet', 'Fruity', 'Woody', 'Floral', 'Spicy']
    
    print("통계적 특성 계산 테스트\n")
    
    # 통계 계산기 초기화
    calculator = StatisticalFeatureCalculator()
    
    # 1. 통계적 특성 계산
    print("1. 통계적 특성:")
    features, feature_names = calculator.compute_statistical_features(
        test_intensity_matrix, test_descriptors
    )
    
    print(f"\n특성 벡터 크기: {features.shape}")
    print(f"특성 이름 개수: {len(feature_names)}")
    
    # 각 설명자별 통계값 출력
    for i, desc in enumerate(test_descriptors):
        print(f"\n{desc} 설명자의 통계값:")
        print(f"  분산: {features[i]:.4f}")
        print(f"  최대값: {features[i + len(test_descriptors)]:.4f}")
        print(f"  최소값: {features[i + 2*len(test_descriptors)]:.4f}")
        print(f"  왜도: {features[i + 3*len(test_descriptors)]:.4f}")
    
    # 2. 유도 설명자 계산
    print("\n2. 유도 설명자:")
    test_mol_props = {
        'mw': 150.0,
        'logp': 2.5,
        'hdon': 2,
        'hacc': 3
    }
    
    derived, derived_names = calculator.compute_derived_descriptors(test_mol_props)
    
    for name, value in zip(derived_names, derived):
        print(f"{name}: {value:.4f}")
