"""
혼합물 유사도 분석 모듈 (Mixture Similarity Analysis)

새로운 혼합물과 기존 학습된 혼합물들 간의 유사도를 계산하고,
가장 유사한 사례를 찾아 예측에 활용하는 기능을 제공합니다.

Author: CJ Whisky Odor Model Team
Version: 1.0 - Initial Implementation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

from config import LEARN_JSONL
from features.feature_vector_builder import FeatureVectorBuilder

class MixtureSimilarityAnalyzer:
    """혼합물 유사도 분석기"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        self.feature_builder = FeatureVectorBuilder()
        self.scaler = StandardScaler()
        
        # 기존 혼합물 데이터 로드
        self.historical_mixtures = []
        self.historical_features = None
        self.historical_outcomes = {'odor': [], 'taste': []}
        
        self._load_historical_data()
        self.logger.info("혼합물 유사도 분석기 초기화 완료")
    
    def _load_historical_data(self):
        """기존 학습된 혼합물 데이터 로드"""
        try:
            learn_file = Path(LEARN_JSONL)
            if not learn_file.exists():
                self.logger.warning("학습 데이터 파일이 없습니다. 유사도 분석을 건너뜁니다.")
                return
            
            mixtures = []
            features_list = []
            odor_outcomes = []
            taste_outcomes = []
            
            with open(learn_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'input_molecules' in data and 'expert_scores' in data:
                            # 혼합물 정보 저장
                            mixture_info = {
                                'trial_id': data.get('trial_id', 'unknown'),
                                'molecules': data['input_molecules'],
                                'expert_scores': data['expert_scores'],
                                'prediction_scores': data.get('prediction_scores', {}),
                                'mode': data.get('mode', 'unknown')
                            }
                            mixtures.append(mixture_info)
                            
                            # 특성 벡터 생성
                            feature_vector = self._extract_mixture_features(data['input_molecules'])
                            features_list.append(feature_vector)
                            
                            # 결과 저장 (모드별)
                            mode = data.get('mode', 'odor')
                            if mode == 'odor':
                                odor_outcomes.append(data['expert_scores'])
                            elif mode == 'taste':
                                taste_outcomes.append(data['expert_scores'])
                                
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON 파싱 오류: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"데이터 처리 오류: {e}")
                        continue
            
            if features_list:
                self.historical_mixtures = mixtures
                self.historical_features = np.array(features_list)
                self.historical_outcomes['odor'] = odor_outcomes
                self.historical_outcomes['taste'] = taste_outcomes
                
                # 특성 벡터 정규화
                if len(features_list) > 1:
                    self.historical_features = self.scaler.fit_transform(self.historical_features)
                
                self.logger.info(f"히스토리 데이터 로드 완료: {len(mixtures)}개 혼합물")
            
        except Exception as e:
            self.logger.error(f"히스토리 데이터 로드 실패: {e}")
    
    def _extract_mixture_features(self, molecules: List[Dict]) -> np.ndarray:
        """혼합물에서 특성 벡터 추출"""
        try:
            # 각 분자의 특성을 가중평균으로 계산
            total_area = sum(mol.get('peak_area', 0) for mol in molecules)
            if total_area == 0:
                return np.zeros(50)  # 기본 특성 벡터 크기
            
            mixture_features = np.zeros(50)  # 특성 벡터 크기 조정 필요
            
            for mol in molecules:
                try:
                    smiles = mol.get('SMILES', '')
                    peak_area = mol.get('peak_area', 0)
                    weight = peak_area / total_area
                    
                    # 분자별 특성 추출 (FeatureVectorBuilder 사용)
                    mol_features = self.feature_builder.build_single_molecule_features(smiles)
                    if mol_features is not None and len(mol_features) > 0:
                        # 특성 벡터 크기 맞추기
                        features_array = np.array(mol_features[:50])
                        if len(features_array) < 50:
                            features_array = np.pad(features_array, (0, 50 - len(features_array)))
                        
                        mixture_features += weight * features_array
                    
                except Exception as e:
                    self.logger.warning(f"분자 특성 추출 실패 ({mol.get('SMILES', 'unknown')}): {e}")
                    continue
            
            return mixture_features
            
        except Exception as e:
            self.logger.error(f"혼합물 특성 추출 실패: {e}")
            return np.zeros(50)
    
    def find_similar_mixtures(
        self, 
        input_molecules: List[Dict], 
        mode: str = 'both',
        top_k: int = 5,
        similarity_method: str = 'cosine'
    ) -> Dict[str, Any]:
        """
        새로운 혼합물과 유사한 기존 혼합물들을 찾기
        
        Args:
            input_molecules: 입력 혼합물
            mode: 'odor', 'taste', 또는 'both'
            top_k: 반환할 유사 혼합물 수
            similarity_method: 'cosine' 또는 'euclidean'
            
        Returns:
            유사도 분석 결과
        """
        if self.historical_features is None or len(self.historical_mixtures) == 0:
            return {
                'similar_mixtures': [],
                'similarity_scores': [],
                'recommendations': {},
                'cluster_info': {},
                'message': '유사도 비교를 위한 히스토리 데이터가 없습니다.'
            }
        
        try:
            # 1. 입력 혼합물의 특성 벡터 추출
            input_features = self._extract_mixture_features(input_molecules)
            
            # 정규화 (기존 데이터와 동일한 스케일)
            if hasattr(self.scaler, 'transform'):
                input_features_scaled = self.scaler.transform(input_features.reshape(1, -1))[0]
            else:
                input_features_scaled = input_features
            
            # 2. 유사도 계산
            if similarity_method == 'cosine':
                similarities = cosine_similarity(
                    input_features_scaled.reshape(1, -1), 
                    self.historical_features
                )[0]
            else:  # euclidean
                distances = euclidean_distances(
                    input_features_scaled.reshape(1, -1), 
                    self.historical_features
                )[0]
                # 거리를 유사도로 변환 (0~1 범위)
                max_dist = np.max(distances) if np.max(distances) > 0 else 1
                similarities = 1 - (distances / max_dist)
            
            # 3. Top-K 유사 혼합물 선택
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            similar_mixtures = []
            similarity_scores = []
            
            for idx in top_indices:
                mixture = self.historical_mixtures[idx]
                score = similarities[idx]
                
                similar_mixtures.append({
                    'trial_id': mixture['trial_id'],
                    'molecules': mixture['molecules'],
                    'expert_scores': mixture['expert_scores'],
                    'mode': mixture['mode'],
                    'similarity_score': float(score)
                })
                similarity_scores.append(float(score))
            
            # 4. 추천 생성
            recommendations = self._generate_recommendations(
                similar_mixtures, mode
            )
            
            # 5. 클러스터 정보
            cluster_info = self._analyze_cluster_position(
                input_features_scaled, similarities
            )
            
            result = {
                'similar_mixtures': similar_mixtures,
                'similarity_scores': similarity_scores,
                'recommendations': recommendations,
                'cluster_info': cluster_info,
                'method': similarity_method,
                'total_historical': len(self.historical_mixtures)
            }
            
            self.logger.info(f"유사도 분석 완료: {len(similar_mixtures)}개 유사 혼합물 발견")
            return result
            
        except Exception as e:
            self.logger.error(f"유사도 분석 실패: {e}")
            return {
                'similar_mixtures': [],
                'similarity_scores': [],
                'recommendations': {},
                'cluster_info': {},
                'error': str(e)
            }
    
    def _generate_recommendations(
        self, 
        similar_mixtures: List[Dict], 
        mode: str
    ) -> Dict[str, Any]:
        """유사 혼합물 기반 추천 생성"""
        if not similar_mixtures:
            return {}
        
        try:
            recommendations = {
                'predicted_range': {},
                'confidence_factors': [],
                'adjustment_suggestions': []
            }
            
            # 유사 혼합물들의 결과 분석
            if mode in ['odor', 'both']:
                odor_scores = []
                for mixture in similar_mixtures:
                    if mixture['mode'] == 'odor' and mixture['expert_scores']:
                        odor_scores.append(mixture['expert_scores'])
                
                if odor_scores:
                    recommendations['predicted_range']['odor'] = self._calculate_score_range(odor_scores)
            
            if mode in ['taste', 'both']:
                taste_scores = []
                for mixture in similar_mixtures:
                    if mixture['mode'] == 'taste' and mixture['expert_scores']:
                        taste_scores.append(mixture['expert_scores'])
                
                if taste_scores:
                    recommendations['predicted_range']['taste'] = self._calculate_score_range(taste_scores)
            
            # 신뢰도 인자
            avg_similarity = np.mean([m['similarity_score'] for m in similar_mixtures])
            recommendations['confidence_factors'] = [
                f"평균 유사도: {avg_similarity:.3f}",
                f"유사 사례 수: {len(similar_mixtures)}개",
                f"최고 유사도: {max(m['similarity_score'] for m in similar_mixtures):.3f}"
            ]
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"추천 생성 실패: {e}")
            return {}
    
    def _calculate_score_range(self, score_list: List[Dict]) -> Dict[str, Any]:
        """점수 범위 계산"""
        if not score_list:
            return {}
        
        # 모든 descriptor 수집
        all_descriptors = set()
        for scores in score_list:
            all_descriptors.update(scores.keys())
        
        range_info = {}
        for desc in all_descriptors:
            values = [scores.get(desc, 0) for scores in score_list if desc in scores]
            if values:
                range_info[desc] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'samples': len(values)
                }
        
        return range_info
    
    def _analyze_cluster_position(
        self, 
        input_features: np.ndarray, 
        similarities: np.ndarray
    ) -> Dict[str, Any]:
        """입력 혼합물의 클러스터 위치 분석"""
        try:
            if len(self.historical_features) < 3:
                return {'message': '클러스터 분석을 위한 데이터 부족'}
            
            # K-means 클러스터링
            n_clusters = min(3, len(self.historical_features) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.historical_features)
            
            # 입력 혼합물이 속할 클러스터 예측
            input_cluster = kmeans.predict(input_features.reshape(1, -1))[0]
            
            # 클러스터별 특성 분석
            cluster_info = {
                'input_cluster': int(input_cluster),
                'total_clusters': n_clusters,
                'cluster_sizes': {},
                'cluster_characteristics': {}
            }
            
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_size = np.sum(cluster_mask)
                cluster_info['cluster_sizes'][i] = int(cluster_size)
                
                # 해당 클러스터의 유사도 분포
                cluster_similarities = similarities[cluster_mask]
                if len(cluster_similarities) > 0:
                    cluster_info['cluster_characteristics'][i] = {
                        'avg_similarity': float(np.mean(cluster_similarities)),
                        'max_similarity': float(np.max(cluster_similarities)),
                        'samples': int(cluster_size)
                    }
            
            return cluster_info
            
        except Exception as e:
            self.logger.warning(f"클러스터 분석 실패: {e}")
            return {'error': str(e)}
    
    def update_historical_data(self, new_mixture_data: Dict):
        """새로운 혼합물 데이터 추가"""
        try:
            # 실시간으로 히스토리 데이터 업데이트
            self._load_historical_data()
            self.logger.info("히스토리 데이터 업데이트 완료")
        except Exception as e:
            self.logger.error(f"히스토리 데이터 업데이트 실패: {e}")

# 전역 인스턴스 생성
similarity_analyzer = MixtureSimilarityAnalyzer()
