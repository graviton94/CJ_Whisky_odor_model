from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from features.molecule_features import MoleculeFeatureExtractor
from features.functional_groups import FunctionalGroupAnalyzer
from features.physical_properties import PhysicalPropertyCalculator
from features.intensity_matrix import IntensityMatrixLoader
from features.data_loader import DataLoader
from features.statistical_features import StatisticalFeatureCalculator

class MixtureDataPreparator:
    """혼합물 데이터 준비를 담당하는 클래스"""
    
    def __init__(self):
        """특성 추출기 초기화"""
        self.feature_extractor = MoleculeFeatureExtractor()
        self.fg_analyzer = FunctionalGroupAnalyzer()
        self.phys_calculator = PhysicalPropertyCalculator()
        self.intensity_loader = IntensityMatrixLoader()
        self.data_loader = DataLoader()
        self.stat_calculator = StatisticalFeatureCalculator()
    
    def prepare_data(
        self,
        smiles_list: List[str],
        peak_areas: List[float],
        mode: str = 'odor',
        csv_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
        """혼합물 데이터 준비
        
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
        # Load intensity matrix
        if csv_path:
            self.intensity_loader = IntensityMatrixLoader(csv_path)
        intensity_matrix, descs = self.intensity_loader.get_intensity_matrix(smiles_list, mode)
        
        # Compute molecular features
        extra_feats, extra_names = self._compute_extra_features(smiles_list, peak_areas)
        
        # Compute detectability
        detect = self.phys_calculator.compute_detectabilities(smiles_list)
        
        return intensity_matrix, descs, extra_feats, extra_names, detect
    
    def _compute_extra_features(
        self,
        smiles_list: List[str],
        peak_areas: List[float]
    ) -> Tuple[np.ndarray, List[str]]:
        """추가 특성 계산
        
        Args:
            smiles_list: SMILES 문자열 리스트
            peak_areas: peak area 리스트
            
        Returns:
            extra_feats: 추가 특성 행렬
            extra_names: 특성 이름 리스트
        """
        # Compute functional groups
        fg_array, fg_names = self.fg_analyzer.compute_functional_groups(smiles_list)
        
        # Compute physical properties
        vol = self.phys_calculator.compute_volatility(smiles_list)
        sim = self.phys_calculator.compute_avg_tanimoto(smiles_list)
        
        # Compute derived features
        rdkit_feats, _ = self.feature_extractor.compute_rdkit_features(smiles_list)
        mol_props = {
            'mw': rdkit_feats[:, 1],    # MolWt index
            'logp': rdkit_feats[:, 0],  # LogP index
            'hdon': rdkit_feats[:, 2],  # NumHDonors index
            'hacc': rdkit_feats[:, 3]   # NumHAcceptors index
        }
        derived, derived_names = self.stat_calculator.compute_derived_descriptors(mol_props)
        
        # Combine extra features
        extra_feats = np.concatenate([fg_array, vol, sim, derived], axis=1)
        extra_names = fg_names + ['VolatilityIndex', 'AvgTanimoto'] + derived_names
        
        return extra_feats, extra_names
