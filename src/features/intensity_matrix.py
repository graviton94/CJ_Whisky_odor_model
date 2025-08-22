from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from features.config import DRAVNIEKS_CSV, ODOR_DESCRIPTORS, TASTE_DESCRIPTORS

class IntensityMatrixLoader:
    """강도 행렬 로딩 및 처리를 위한 클래스"""
    
    def __init__(self, csv_path: Optional[Path] = None):
        """
        Args:
            csv_path: 데이터 CSV 파일 경로 (없으면 기본값 사용)
        """
        self.csv_path = csv_path if csv_path else DRAVNIEKS_CSV
        self._df = None
        self._idx_map = None
    
    def _load_data(self):
        """CSV 파일 로딩"""
        if self._df is None:
            try:
                self._df = pd.read_csv(str(self.csv_path), encoding='utf-8-sig')
            except UnicodeDecodeError:
                self._df = pd.read_csv(str(self.csv_path), encoding='cp949')
                
            self._df.columns = self._df.columns.str.strip()
            self._idx_map = {row['SMILES']:i for i, row in self._df.iterrows()}
    
    def get_intensity_matrix(
        self, smiles_list: List[str], mode: str = 'odor'
    ) -> Tuple[np.ndarray, List[str]]:
        """강도 행렬 및 설명자 리스트 반환
        
        Args:
            smiles_list: SMILES 문자열 리스트
            mode: 'odor' 또는 'taste'
            
        Returns:
            intensity_matrix: 강도 행렬
            descriptors: 설명자 리스트
        """
        self._load_data()
        
        descs = ODOR_DESCRIPTORS if mode == 'odor' else TASTE_DESCRIPTORS
        Y = []
        
        for smi in smiles_list:
            i = self._idx_map.get(smi)
            if i is None:
                raise ValueError(f"SMILES {smi} not in CSV")
            Y.append(self._df.iloc[i][descs].values)
            
        return np.array(Y, dtype=np.float32), descs
