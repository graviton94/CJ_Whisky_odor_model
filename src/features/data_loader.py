from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

class DataLoader:
    """데이터 로딩을 담당하는 클래스"""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Args:
            base_path: 기본 데이터 디렉토리 경로
        """
        self.base_path = base_path or Path(__file__).parent / 'data'
    
    def load_csv_data(self, filepath: Path) -> pd.DataFrame:
        """CSV 파일 로딩
        
        Args:
            filepath: CSV 파일 경로
            
        Returns:
            데이터프레임
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='cp949')
            
        df.columns = df.columns.str.strip()
        return df
    
    def get_smiles_mapping(self, df: pd.DataFrame) -> dict:
        """SMILES 매핑 생성
        
        Args:
            df: 데이터프레임
            
        Returns:
            SMILES와 인덱스 매핑 딕셔너리
        """
        return {row['SMILES']:i for i, row in df.iterrows()}
