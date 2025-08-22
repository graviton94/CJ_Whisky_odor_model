from typing import Dict, List
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class StatisticalValidator:
    def __init__(self):
        self.historical_data = []
        self.confidence_level = 0.95
        
    def validate_prediction(self, prediction: Dict, actual: Dict = None, 
                          bootstrap_samples: int = 1000) -> Dict:
        """예측 결과의 통계적 검증"""
        validation_results = {
            'confidence_intervals': {},
            'outliers': {},
            'statistics': {}
        }
        
        # predict_hybrid의 중첩 딕셔너리 구조 처리
        flat_prediction = {}
        if isinstance(prediction, dict):
            for mode, mode_data in prediction.items():
                if isinstance(mode_data, dict) and 'corrected' in mode_data:
                    # predict_hybrid 형식: {mode: {'corrected': {...}}}
                    for desc, value in mode_data['corrected'].items():
                        flat_prediction[f"{mode}_{desc}"] = value
                elif isinstance(mode_data, dict):
                    # 일반 딕셔너리 형식
                    for desc, value in mode_data.items():
                        if isinstance(value, (int, float)):
                            flat_prediction[f"{mode}_{desc}"] = value
                elif isinstance(mode_data, (int, float)):
                    # 단순 값
                    flat_prediction[mode] = mode_data
        
        # 빈 예측 결과 처리
        if not flat_prediction:
            return {
                'confidence_intervals': {},
                'outliers': {},
                'statistics': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            }
        
        # 1. 신뢰구간 계산
        for descriptor, value in flat_prediction.items():
            if isinstance(value, (int, float)):
                ci = self._calculate_confidence_interval(value, bootstrap_samples)
                validation_results['confidence_intervals'][descriptor] = ci
        
        # 2. 이상치 탐지
        outliers = self._detect_outliers(flat_prediction)
        validation_results['outliers'] = outliers
        
        # 3. 기술 통계량
        validation_results['statistics'] = self._calculate_statistics(flat_prediction)
        
        # 4. 실제값과 비교 (가능한 경우)
        if actual is not None:
            validation_results['comparison'] = self._compare_with_actual(flat_prediction, actual)
            
        return validation_results
    
    def _calculate_confidence_interval(self, value: float, n_samples: int) -> Dict:
        """부트스트랩 방법을 사용한 신뢰구간 계산"""
        if isinstance(value, (int, float)) and not np.isnan(value) and np.isfinite(value):
            # 정규분포 가정 (표준편차 감소, 수치 안정성 개선)
            std_dev = max(value * 0.05, 0.01)  # 최소 표준편차 설정
            samples = np.random.normal(value, std_dev, n_samples)
            
            # 유효한 샘플만 사용
            valid_samples = samples[np.isfinite(samples)]
            if len(valid_samples) < 2:
                return {'lower': float(value), 'upper': float(value), 'mean': float(value), 'std': 0.0}
            
            try:
                lower, upper = stats.t.interval(self.confidence_level, len(valid_samples)-1,
                                              loc=np.mean(valid_samples),
                                              scale=stats.sem(valid_samples))
                return {
                    'lower': max(0.0, float(lower)),  # 음수 방지
                    'upper': min(10.0, float(upper)),  # 10 초과 방지
                    'mean': float(np.mean(valid_samples)),
                    'std': float(np.std(valid_samples))
                }
            except (ValueError, ZeroDivisionError):
                # 통계 계산 실패 시 기본값 반환
                return {'lower': float(value), 'upper': float(value), 'mean': float(value), 'std': 0.0}
        
        return {'lower': 0.0, 'upper': 0.0, 'mean': 0.0, 'std': 0.0}
    
    def _detect_outliers(self, prediction: Dict) -> Dict:
        """이상치 탐지"""
        values = np.array(list(prediction.values()))
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = {}
        for descriptor, value in prediction.items():
            if value < lower_bound or value > upper_bound:
                outliers[descriptor] = {
                    'value': value,
                    'z_score': (value - np.mean(values)) / np.std(values),
                    'is_low': value < lower_bound,
                    'is_high': value > upper_bound
                }
        
        return outliers
    
    def _calculate_statistics(self, prediction: Dict) -> Dict:
        """기술 통계량 계산"""
        values = np.array(list(prediction.values()))
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'n_samples': len(values)
        }
    
    def _compare_with_actual(self, prediction: Dict, actual: Dict) -> Dict:
        """예측값과 실제값 비교"""
        # 공통 키만 사용
        common_keys = set(prediction.keys()) & set(actual.keys())
        if not common_keys:
            return {}
            
        pred_values = [prediction[k] for k in common_keys]
        true_values = [actual[k] for k in common_keys]
        
        return {
            'mse': mean_squared_error(true_values, pred_values),
            'rmse': np.sqrt(mean_squared_error(true_values, pred_values)),
            'r2': r2_score(true_values, pred_values),
            'correlation': stats.pearsonr(true_values, pred_values)[0],
            'mae': np.mean(np.abs(np.array(true_values) - np.array(pred_values)))
        }
    
    def add_to_history(self, prediction: Dict, actual: Dict = None):
        """예측 결과를 히스토리에 추가"""
        self.historical_data.append({
            'prediction': prediction,
            'actual': actual,
            'timestamp': pd.Timestamp.now()
        })
    
    def analyze_trends(self, window: int = 10) -> Dict:
        """시계열 트렌드 분석"""
        if not self.historical_data:
            return {}
            
        recent_data = self.historical_data[-window:]
        
        trends = {}
        descriptors = recent_data[0]['prediction'].keys()
        
        for desc in descriptors:
            values = [d['prediction'][desc] for d in recent_data]
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(values)), values)
            
            trends[desc] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err
            }
        
        return trends
