from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from typing import Dict, Tuple, List
import numpy as np
from functools import lru_cache

class MoleculeAnalyzer:
    def __init__(self):
        self.descriptors_cache = {}
        self.fingerprints_cache = {}
        
    def analyze_molecule(self, smiles: str) -> Tuple[Dict, Dict]:
        """RDKit 기반 고급 분자 특성 분석"""
        if smiles in self.descriptors_cache:
            return self.descriptors_cache[smiles], self.fingerprints_cache[smiles]
            
        try:
            # 1. 3D 구조 생성 및 최적화
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 2. 고급 기술자 계산
            descriptors = {
                'MolLogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'Rings': Descriptors.RingCount(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'MW': Descriptors.ExactMolWt(mol),
                'LabuteASA': Descriptors.LabuteASA(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'Ipc': Descriptors.Ipc(mol),
                'HallKierAlpha': Descriptors.HallKierAlpha(mol),
                'Density': Descriptors.Density(mol) if hasattr(Descriptors, 'Density') else None,
            }
            
            # 3. 다중 분자 지문 생성 (RDKit 버전 호환성 처리)
            try:
                fingerprints = {
                    'MACCS': MACCSkeys.GenMACCSKeys(mol),
                    'Morgan': AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024),
                    'AtomPairs': Pairs.GetAtomPairFingerprintAsBitVect(mol, nBits=1024)
                }
            except TypeError:
                # 구버전 RDKit에서는 nBits 파라미터 미지원
                fingerprints = {
                    'MACCS': MACCSkeys.GenMACCSKeys(mol),
                    'Morgan': AllChem.GetMorganFingerprintAsBitVect(mol, 2),
                    'AtomPairs': Pairs.GetAtomPairFingerprintAsBitVect(mol)
                }
            
            # 캐시 저장
            self.descriptors_cache[smiles] = descriptors
            self.fingerprints_cache[smiles] = fingerprints
            
            return descriptors, fingerprints
            
        except Exception as e:
            print(f"Error analyzing molecule {smiles}: {str(e)}")
            return {}, {}
    
    def analyze_mixture(self, smiles_list: List[str], concentrations: List[float] = None) -> Dict:
        """혼합물 분석"""
        if concentrations is None:
            concentrations = [1.0] * len(smiles_list)
            
        mixture_properties = {
            'total_descriptors': {},
            'weighted_fingerprints': {},
            'interaction_scores': {}
        }
        
        # 각 분자의 특성 분석
        all_descriptors = []
        all_fingerprints = []
        for smiles in smiles_list:
            desc, fp = self.analyze_molecule(smiles)
            all_descriptors.append(desc)
            all_fingerprints.append(fp)
        
        # 가중 평균 계산
        for key in all_descriptors[0].keys():
            values = [d[key] for d in all_descriptors if d.get(key) is not None]
            if values:
                mixture_properties['total_descriptors'][key] = np.average(values, weights=concentrations[:len(values)])
        
        # 지문 가중 평균
        for fp_type in ['MACCS', 'Morgan', 'AtomPairs']:
            fp_arrays = [fp[fp_type] for fp in all_fingerprints if fp_type in fp]
            if fp_arrays:
                mixture_properties['weighted_fingerprints'][fp_type] = np.average(fp_arrays, weights=concentrations[:len(fp_arrays)], axis=0)
        
        return mixture_properties
