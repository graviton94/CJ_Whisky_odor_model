"""
통합 예측 및 학습 시스템 (Integrated Prediction and Learning System)

이 시스템은 다음 워크플로우를 구현합니다:
1. Dataset 기반 초기 훈련
2. Ontology 규칙을 통한 미세조정
3. 예측값과 전문가 입력의 혼합 학습
4. 반복적 모델 개선

Author: CJ Whisky Odor Model Team
Version: 1.1 - Enhanced Ontology Integration
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

from config import (
    DRAVNIEKS_CSV, LEARN_JSONL, TRIAL_CSV,
    MODEL_DIR, ODOR_DESCRIPTORS, TASTE_DESCRIPTORS
)
from train_finetune import train_model
from predict_hybrid import predict_hybrid, build_description_support
from expert_module import save_trial_result, save_trial_long_format
from ontology_manager import OntologyManager
from statistical_validator import StatisticalValidator
from mixture_interactions import MixtureInteractionModel
from mixture_similarity import similarity_analyzer

try:
    from rules.engine import RuleEngine
    from rules.cache import CacheManager
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False
    logging.warning("Rules 모듈을 찾을 수 없습니다. 기본 온톨로지 시스템을 사용합니다.")

class IntegratedPredictionSystem:
    """통합 예측 및 학습 시스템 - 향상된 온톨로지 통합"""
    
    def __init__(self):
        """시스템 초기화"""
        self.logger = self._setup_logger()
        
        # 온톨로지 관리자 초기화 (기본 규칙 사용)
        self.ontology_manager = OntologyManager()
        
        # 기존 컴포넌트
        if RULES_AVAILABLE:
            self.rule_engine = RuleEngine()
            self.rule_engine.cache = CacheManager(max_size=50, expiration_time=1800)  # 30분 캐시
        else:
            self.rule_engine = None
            
        self.validator = StatisticalValidator()
        self.interaction_model = MixtureInteractionModel()
        self.performance_history = []
        
        # 모델 상태 추적
        self.model_version = {'odor': 1, 'taste': 1}
        self.last_training = {'odor': None, 'taste': None}
        
        # 기존 모델 상태와 훈련 이력 복원
        self._restore_model_state()
        
        self.logger.info("통합 예측 시스템 초기화 완료 (향상된 온톨로지 통합)")
    
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _restore_model_state(self):
        """모델 상태 및 학습 이력 복원"""
        try:
            from pathlib import Path
            import os
            
            # 모델 상태 파일 경로
            project_root = Path(__file__).parent.parent
            state_file = project_root / "model_state.json"
            
            # 상태 파일이 있으면 로드
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                self.model_version = state_data.get('model_version', {'odor': 1, 'taste': 1})
                
                # 날짜 문자열을 datetime 객체로 변환
                last_training_data = state_data.get('last_training', {'odor': None, 'taste': None})
                for mode in ['odor', 'taste']:
                    if last_training_data[mode]:
                        try:
                            self.last_training[mode] = datetime.fromisoformat(last_training_data[mode])
                        except:
                            self.last_training[mode] = None
                
                self.performance_history = state_data.get('performance_history', [])
                
                self.logger.info(f"모델 상태 복원 완료 - Odor v{self.model_version['odor']}, Taste v{self.model_version['taste']}")
                
            else:
                # 상태 파일이 없으면 모델 파일에서 추정
                self._restore_training_timestamps()
                self._estimate_model_versions()
                
        except Exception as e:
            self.logger.warning(f"모델 상태 복원 실패: {e}")
            # 기존 파일에서 훈련 시간만이라도 복원
            self._restore_training_timestamps()
    
    def _estimate_model_versions(self):
        """기존 학습 파일들을 기반으로 모델 버전 추정"""
        try:
            # mixture_trials_log.csv에서 학습 횟수 확인
            trial_csv_path = Path(TRIAL_CSV)
            if trial_csv_path.exists():
                df = pd.read_csv(trial_csv_path)
                if not df.empty:
                    # 각 모드별로 학습 횟수 계산
                    odor_count = len(df[df['mode'] == 'odor']) if 'mode' in df.columns else len(df)
                    taste_count = len(df[df['mode'] == 'taste']) if 'mode' in df.columns else len(df)
                    
                    # 최소 1, 학습 횟수에 따라 버전 설정
                    self.model_version['odor'] = max(1, odor_count + 1)
                    self.model_version['taste'] = max(1, taste_count + 1)
                    
                    self.logger.info(f"학습 이력 기반 버전 추정 - Odor: {odor_count}회 -> v{self.model_version['odor']}, Taste: {taste_count}회 -> v{self.model_version['taste']}")
            
        except Exception as e:
            self.logger.warning(f"모델 버전 추정 실패: {e}")
    
    def _save_model_state(self):
        """현재 모델 상태를 파일에 저장"""
        try:
            project_root = Path(__file__).parent.parent
            state_file = project_root / "model_state.json"
            
            state_data = {
                'model_version': self.model_version.copy(),
                'last_training': {
                    k: v.isoformat() if v else None 
                    for k, v in self.last_training.items()
                },
                'performance_history': self.performance_history[-50:],  # 최근 50개만 저장
                'timestamp': datetime.now().isoformat()
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"모델 상태 저장 완료: {state_file}")
            
        except Exception as e:
            self.logger.warning(f"모델 상태 저장 실패: {e}")
    
    def _restore_training_timestamps(self):
        """기존 모델 파일들에서 훈련 시간 복원"""
        try:
            import os
            from pathlib import Path
            
            # 프로젝트 루트 경로에서 모델 파일 찾기
            project_root = Path(__file__).parent.parent  # src의 상위 디렉토리
            odor_model_path = project_root / "odor_finetune.pth"
            taste_model_path = project_root / "taste_finetune.pth"
            
            self.logger.info(f"모델 파일 경로 확인 - Odor: {odor_model_path}, Taste: {taste_model_path}")
            
            # 모델 파일이 존재하면 수정 시간을 훈련 시간으로 설정
            if odor_model_path.exists():
                mtime = os.path.getmtime(odor_model_path)
                self.last_training['odor'] = datetime.fromtimestamp(mtime)
                self.logger.info(f"Odor 모델 훈련 시간 복원: {self.last_training['odor']}")
            else:
                self.logger.warning(f"Odor 모델 파일을 찾을 수 없음: {odor_model_path}")
            
            if taste_model_path.exists():
                mtime = os.path.getmtime(taste_model_path)
                self.last_training['taste'] = datetime.fromtimestamp(mtime)
                self.logger.info(f"Taste 모델 훈련 시간 복원: {self.last_training['taste']}")
            else:
                self.logger.warning(f"Taste 모델 파일을 찾을 수 없음: {taste_model_path}")
                
        except Exception as e:
            self.logger.warning(f"훈련 시간 복원 실패: {e}")
            # 기본값 유지
    
    def predict_mixture(
        self,
        input_molecules: List[Dict[str, Any]],
        mode: str = 'both',
        use_ontology: bool = True,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        혼합물 예측 (핵심 예측 메서드)
        
        Args:
            input_molecules: [{'SMILES': str, 'peak_area': float}, ...]
            mode: 'odor', 'taste', 또는 'both'
            use_ontology: 온톨로지 규칙 적용 여부
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            예측 결과 딕셔너리
        """
        self.logger.info(f"혼합물 예측 시작 - 분자 수: {len(input_molecules)}, 모드: {mode}")
        
        try:
            # 0. 유사 혼합물 분석 (새로 추가!)
            similarity_result = similarity_analyzer.find_similar_mixtures(
                input_molecules, mode=mode, top_k=3
            )
            
            # 1. 기본 하이브리드 예측
            hybrid_result = predict_hybrid(input_molecules, verbose=False)
            
            # 2. 상호작용 분석 (예외 처리 추가)
            molecules_for_interaction = [
                (mol['SMILES'], mol['peak_area']) 
                for mol in input_molecules
            ]
            
            interaction_results = {}
            try:
                if mode in ['odor', 'both']:
                    interaction_results['odor'] = self.interaction_model.calculate_interactions(
                        molecules_for_interaction, mode='odor'
                    )
                
                if mode in ['taste', 'both']:
                    interaction_results['taste'] = self.interaction_model.calculate_interactions(
                        molecules_for_interaction, mode='taste'
                    )
            except Exception as e:
                self.logger.warning(f"상호작용 분석 실패: {str(e)}, 기본 예측만 사용")
                # 상호작용 분석 실패 시 빈 결과로 대체
                interaction_results = {}
            
            # 3. 온톨로지 규칙 적용 (선택적)
            if use_ontology:
                hybrid_result = self._apply_ontology_rules(hybrid_result, interaction_results, input_molecules)
            
            # 4. 신뢰도 계산
            confidence_scores = self._calculate_confidence(hybrid_result, interaction_results)
            
            # 5. 통계적 검증
            validation_result = self.validator.validate_prediction(
                hybrid_result
            )
            
            # 6. description_support 생성 (분자별 기여도)
            try:
                support_data = {}
                if 'contributions' in hybrid_result.get('odor', {}):
                    support_data['odor'] = build_description_support(
                        hybrid_result['odor']['contributions'], mode='odor'
                    )
                if 'contributions' in hybrid_result.get('taste', {}):
                    support_data['taste'] = build_description_support(
                        hybrid_result['taste']['contributions'], mode='taste'
                    )
                
                # hybrid_result에 description_support 추가
                hybrid_result['description_support'] = support_data
                
                self.logger.info(f"분자별 기여도 분석 완료 - odor: {len(support_data.get('odor', {}))}, taste: {len(support_data.get('taste', {}))}")
            except Exception as e:
                self.logger.warning(f"분자별 기여도 생성 실패: {str(e)}")
                hybrid_result['description_support'] = {'odor': {}, 'taste': {}}
            
            # 7. 결과 통합 (유사도 분석 포함)
            final_result = {
                'prediction': hybrid_result,
                'interactions': interaction_results,
                'confidence': confidence_scores,
                'validation': validation_result,
                'similarity': similarity_result,  # 새로 추가!
                'model_version': self.model_version.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("혼합물 예측 완료")
            return final_result
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            raise
    
    def learn_from_expert(
        self,
        trial_id: str,
        input_molecules: List[Dict[str, Any]],
        expert_scores: Dict[str, float],
        mode: str,
        prediction_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        전문가 입력값을 통한 학습
        
        Args:
            trial_id: 실험 ID
            input_molecules: 입력 분자들
            expert_scores: 전문가가 평가한 점수
            mode: 'odor' 또는 'taste'
            prediction_result: 이전 예측 결과 (선택적)
            
        Returns:
            학습 결과
        """
        self.logger.info(f"전문가 학습 시작 - Trial ID: {trial_id}, 모드: {mode}")
        
        try:
            # 1. 예측값 생성 (제공되지 않은 경우)
            if prediction_result is None:
                prediction_result = self.predict_mixture(input_molecules, mode=mode)
            
            # 예측 결과 구조 확인 및 적절한 값 추출
            if 'prediction' in prediction_result:
                # 새로운 구조: {'prediction': hybrid_result, 'confidence': ...}
                hybrid_result = prediction_result['prediction']
                if mode in hybrid_result and 'corrected' in hybrid_result[mode]:
                    predicted_scores = hybrid_result[mode]['corrected']
                else:
                    raise KeyError(f"예측 결과에서 {mode} 모드의 corrected 값을 찾을 수 없습니다")
            else:
                # 이전 구조: 직접 hybrid_result 형태
                if mode in prediction_result and 'corrected' in prediction_result[mode]:
                    predicted_scores = prediction_result[mode]['corrected']
                else:
                    raise KeyError(f"예측 결과에서 {mode} 모드의 corrected 값을 찾을 수 없습니다")
            
            # 2. 전문가 데이터 저장
            save_trial_result(
                trial_id=trial_id,
                mode=mode,
                input_molecules=input_molecules,
                predict_scores=predicted_scores,
                expert_scores=expert_scores,
                extra_info={
                    'model_version': self.model_version[mode],
                    'confidence': prediction_result.get('confidence', {}).get(mode, {})
                }
            )
            
            # 3. 학습 데이터 형식으로 변환
            learning_data = self._prepare_learning_data(
                input_molecules, expert_scores, mode
            )
            
            # 4. JSONL 파일에 추가
            self._append_to_learning_dataset(learning_data, mode)
            
            # 5. 성능 분석
            performance_metrics = self._analyze_performance(
                predicted_scores, expert_scores, mode
            )
            
            # 6. 재훈련 필요성 판단 및 실행
            retrain_needed = self._should_retrain(performance_metrics, mode)
            retrain_result = None
            
            # 무조건 자동 재훈련 실행 (관능평가 데이터가 입력되면 항상 학습)
            try:
                self.logger.info(f"자동 재훈련 시작 - 모드: {mode} (관능평가 데이터 학습)")
                retrain_result = self.retrain_model(mode, force_retrain=True)
                self.logger.info(f"자동 재훈련 완료 - 성공: {retrain_result.get('retrained', False)}")
            except Exception as retrain_error:
                self.logger.warning(f"자동 재훈련 실패: {str(retrain_error)}")
                retrain_result = {'retrained': False, 'error': str(retrain_error)}
            
            learning_result = {
                'trial_id': trial_id,
                'mode': mode,
                'performance_metrics': performance_metrics,
                'retrain_recommended': retrain_needed,
                'auto_retrain_executed': True,  # 항상 실행
                'retrain_result': retrain_result,
                'learning_data_saved': True,
                'timestamp': datetime.now().isoformat()
            }
            
            status_msg = f"전문가 학습 완료 - 재훈련 필요: {retrain_needed}"
            if retrain_result:
                status_msg += f", 자동 재훈련: {'성공' if retrain_result.get('retrained') else '실패'}"
            self.logger.info(status_msg)
            return learning_result
            
        except Exception as e:
            self.logger.error(f"전문가 학습 중 오류 발생: {str(e)}")
            raise
    
    def retrain_model(
        self,
        mode: str,
        force_retrain: bool = False,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        모델 재훈련
        
        Args:
            mode: 'odor' 또는 'taste'
            force_retrain: 강제 재훈련 여부
            validation_split: 검증 데이터 비율
            
        Returns:
            훈련 결과
        """
        self.logger.info(f"모델 재훈련 시작 - 모드: {mode}")
        
        try:
            # 1. 재훈련 필요성 재확인
            if not force_retrain and not self._check_retrain_necessity(mode):
                return {
                    'retrained': False,
                    'reason': 'No significant improvement expected',
                    'current_version': self.model_version[mode]
                }
            
            # 2. 학습 데이터 로드
            training_data = self._load_training_data(mode)
            self.logger.info(f"로드된 학습 데이터: {len(training_data)}개")
            
            if len(training_data) < 2:  # 최소 데이터 요구사항을 2개로 낮춤
                return {
                    'retrained': False,
                    'reason': f'Insufficient training data: {len(training_data)} samples (minimum: 2)',
                    'current_version': self.model_version[mode]
                }
            
            # 3. 모델 훈련
            self.logger.info(f"모델 훈련 시작: {mode} 모드, {len(training_data)}개 데이터")
            try:
                training_result = train_model(mode=mode)
                self.logger.info(f"모델 훈련 완료: {training_result}")
                
                if training_result is None:
                    return {
                        'retrained': False,
                        'reason': 'Training function returned None',
                        'current_version': self.model_version[mode]
                    }
                
                # 훈련 성공 여부 확인
                if not training_result.get('success', False):
                    return {
                        'retrained': False,
                        'reason': 'Training completed but reported as failed',
                        'current_version': self.model_version[mode],
                        'training_result': training_result
                    }
                    
            except Exception as train_error:
                self.logger.error(f"모델 훈련 중 오류: {str(train_error)}")
                import traceback
                self.logger.error(f"훈련 오류 상세: {traceback.format_exc()}")
                return {
                    'retrained': False,
                    'reason': f'Training failed: {str(train_error)}',
                    'current_version': self.model_version[mode],
                    'error_details': str(train_error)
                }
            
            # 4. 모델 버전 업데이트
            self.model_version[mode] += 1
            self.last_training[mode] = datetime.now()
            
            # 5. 성능 비교 및 기록
            performance_improvement = self._evaluate_model_improvement(
                mode, training_result
            )
            
            retrain_result = {
                'retrained': True,
                'mode': mode,
                'new_version': self.model_version[mode],
                'training_result': training_result,
                'performance_improvement': performance_improvement,
                'training_samples': len(training_data),
                'timestamp': datetime.now().isoformat()
            }
            
            # 6. 모델 상태 저장
            self._save_model_state()
            
            # 7. 유사도 분석기 히스토리 데이터 업데이트 (새로 추가!)
            try:
                similarity_analyzer.update_historical_data({})
                self.logger.info("유사도 분석기 히스토리 데이터 업데이트 완료")
            except Exception as e:
                self.logger.warning(f"유사도 분석기 업데이트 실패: {e}")
            
            self.logger.info(f"모델 재훈련 완료 - 새 버전: {self.model_version[mode]}")
            return retrain_result
            
        except Exception as e:
            self.logger.error(f"모델 재훈련 중 전체 오류 발생: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                'retrained': False,
                'reason': f'Unexpected error: {str(e)}',
                'current_version': self.model_version[mode],
                'error_details': traceback.format_exc()
            }
    
    def optimize_ontology(
        self,
        mode: str,
        n_trials: int = 100,
        use_recent_data: bool = True
    ) -> Dict[str, Any]:
        """
        온톨로지 규칙 최적화
        
        Args:
            mode: 'odor' 또는 'taste'
            n_trials: Optuna 시행 횟수
            use_recent_data: 최근 데이터만 사용 여부
            
        Returns:
            최적화 결과
        """
        self.logger.info(f"온톨로지 최적화 시작 - 모드: {mode}, 시행 횟수: {n_trials}")
        
        try:
            # 최적화 수행 (기존 ontology_optuna.py 활용)
            import subprocess
            import sys
            
            result = subprocess.run([
                sys.executable, '-c',
                f"""
import sys
sys.path.append('{Path(__file__).parent}')
from ontology_optuna import optimize_ontology
optimize_ontology('{mode}', {n_trials})
"""
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"온톨로지 최적화 완료 - 모드: {mode}")
                return {
                    'optimized': True,
                    'mode': mode,
                    'n_trials': n_trials,
                    'output': result.stdout
                }
            else:
                self.logger.error(f"온톨로지 최적화 실패: {result.stderr}")
                return {
                    'optimized': False,
                    'error': result.stderr
                }
                
        except Exception as e:
            self.logger.error(f"온톨로지 최적화 중 오류 발생: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            # 데이터 통계
            dataset_stats = self._get_dataset_statistics()
            
            # 모델 정보
            model_info = {
                'versions': self.model_version.copy(),
                'last_training': {
                    k: v.isoformat() if v else None 
                    for k, v in self.last_training.items()
                }
            }
            
            # 캐시 정보
            cache_info = {
                'size': len(self.rule_engine.cache.cache),
                'max_size': self.rule_engine.cache.max_size
            }
            
            return {
                'system_ready': True,
                'dataset_statistics': dataset_stats,
                'model_info': model_info,
                'cache_info': cache_info,
                'performance_history': self.performance_history[-10:],  # 최근 10개
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'system_ready': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # 내부 헬퍼 메서드들
    def _apply_ontology_rules(self, hybrid_result, interaction_results, input_molecules=None):
        """온톨로지 규칙 적용"""
        try:
            enhanced_result = {}
            
            for mode in ['odor', 'taste']:
                if mode in hybrid_result:
                    mode_result = hybrid_result[mode]
                    corrected_scores = mode_result.get('corrected', {})
                    
                    self.logger.info(f"[DEBUG {mode}] Mode in hybrid_result: True")
                    self.logger.info(f"[DEBUG {mode}] Corrected scores keys: {list(corrected_scores.keys())}")
                    self.logger.info(f"[DEBUG {mode}] Corrected scores count: {len(corrected_scores)}")
                    
                    if corrected_scores:
                        # SMILES 리스트와 농도 정보 추출
                        smiles_list = []
                        concentrations = []
                        
                        # 우선 input_molecules에서 직접 추출 시도
                        if input_molecules:
                            total_area = sum(mol.get('peak_area', 1.0) for mol in input_molecules)
                            for mol_data in input_molecules:
                                smiles = mol_data.get('SMILES', mol_data.get('smiles', ''))
                                area = mol_data.get('peak_area', mol_data.get('Peak_Area', 1.0))
                                if smiles:
                                    smiles_list.append(smiles)
                                    concentrations.append(area / total_area if total_area > 0 else 1.0)
                            
                            self.logger.info(f"[DEBUG {mode}] SMILES extracted: {len(smiles_list)} molecules")
                            self.logger.info(f"[DEBUG {mode}] First 3 SMILES: {smiles_list[:3]}")
                        
                        # interaction_results에서 정보 추출 (보조)
                        if not smiles_list and mode in interaction_results and 'molecules' in interaction_results[mode]:
                            for mol_data in interaction_results[mode]['molecules']:
                                smiles_list.append(mol_data.get('SMILES', ''))
                                concentrations.append(mol_data.get('normalized_concentration', 1.0))
                        
                        # 기본값 설정 (정보가 없는 경우)
                        if not smiles_list:
                            smiles_list = ['CCO']  # 기본값
                            concentrations = [1.0]
                            self.logger.warning(f"[DEBUG {mode}] Using default SMILES - this indicates a problem!")
                        
                        self.logger.info(f"[DEBUG {mode}] Final SMILES count: {len(smiles_list)}")
                        self.logger.info(f"[DEBUG {mode}] Corrected scores count: {len(corrected_scores)}")
                        
                        # 온톨로지 관리자를 통한 규칙 적용
                        ontology_result = self.ontology_manager.apply_rules(
                            result_dict=corrected_scores,
                            smiles_list=smiles_list,
                            descriptor_type=mode,
                            detectability=None,  # 필요시 추가
                            extra_features=None,  # 필요시 추가
                            concentrations=concentrations
                        )
                        
                        self.logger.info(f"[DEBUG {mode}] Ontology result keys: {list(ontology_result.keys())}")
                        self.logger.info(f"[DEBUG {mode}] Rule log length: {len(ontology_result.get('rule_log', []))}")
                        if len(ontology_result.get('rule_log', [])) > 0:
                            self.logger.info(f"[DEBUG {mode}] First rule: {ontology_result.get('rule_log', [])[0]}")
                        
                        # 결과 업데이트
                        enhanced_mode_result = mode_result.copy()
                        enhanced_mode_result['corrected'] = ontology_result['corrected_scores']
                        enhanced_mode_result['ontology_log'] = ontology_result.get('rule_log', [])
                        enhanced_mode_result['mixture_interactions'] = ontology_result.get('mixture_interactions', {})
                        enhanced_mode_result['ontology_parameters'] = ontology_result.get('parameters', {})
                        
                        enhanced_result[mode] = enhanced_mode_result
                        
                        self.logger.info(f"온톨로지 규칙 적용 완료 - {mode}: {len(ontology_result.get('rule_log', []))}개 규칙 적용")
                    else:
                        self.logger.warning(f"[DEBUG {mode}] Corrected scores is empty - skipping ontology rules")
                        enhanced_result[mode] = mode_result
                else:
                    self.logger.warning(f"[DEBUG {mode}] Mode not in hybrid_result")
                    enhanced_result[mode] = hybrid_result.get(mode, {})
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"온톨로지 규칙 적용 실패: {e}")
            return hybrid_result
    
    def _calculate_confidence(self, hybrid_result, interaction_results):
        """신뢰도 계산 (온톨로지 적용 고려)"""
        confidence_scores = {}
        
        for mode in ['odor', 'taste']:
            if mode in hybrid_result:
                mode_result = hybrid_result[mode]
                corrected_scores = mode_result.get('corrected', {})
                ontology_log = mode_result.get('ontology_log', [])
                
                mode_confidence = {}
                
                for descriptor, score in corrected_scores.items():
                    base_confidence = 0.7
                    
                    # 온톨로지 규칙 적용 여부에 따른 신뢰도 조정
                    applied_rules = [r for r in ontology_log if r.get('to') == descriptor or r.get('descriptor') == descriptor]
                    if applied_rules:
                        # 규칙이 적용된 경우 신뢰도 증가
                        base_confidence += min(0.2, len(applied_rules) * 0.05)
                    
                    # 분자 상호작용 강도에 따른 조정
                    if mode in interaction_results:
                        interaction_data = interaction_results[mode]
                        interaction_strength = interaction_data.get('weighted_strength', 0)
                        base_confidence += min(0.15, interaction_strength * 0.3)
                    
                    # 예측값 범위에 따른 조정
                    if 0.3 <= score <= 8.0:
                        base_confidence += 0.1
                    elif score < 0.1 or score > 9.0:
                        base_confidence -= 0.1
                    
                    mode_confidence[descriptor] = min(0.95, max(0.3, base_confidence))
                
                confidence_scores[mode] = mode_confidence
        
        return confidence_scores
    
    def _prepare_learning_data(self, input_molecules, expert_scores, mode):
        """학습 데이터 준비"""
        from data_processor import build_input_vector
        
        smiles_list = [m['SMILES'] for m in input_molecules]
        peak_areas = np.array([m['peak_area'] for m in input_molecules])
        
        # Feature vector 생성
        features = build_input_vector(smiles_list, peak_areas, mode=mode)
        
        # Descriptor 리스트
        desc_list = ODOR_DESCRIPTORS if mode == 'odor' else TASTE_DESCRIPTORS
        
        # Expert scores를 순서에 맞게 정리
        labels = [expert_scores.get(desc, 0.0) for desc in desc_list]
        
        return {
            'smiles_list': smiles_list,
            'peak_area': peak_areas.tolist(),
            'features': features.tolist(),
            'labels': labels,
            'desc_list': desc_list
        }
    
    def _append_to_learning_dataset(self, learning_data, mode):
        """학습 데이터셋에 추가"""
        import json
        from datetime import datetime
        
        # Trial ID 생성
        timestamp = datetime.now()
        trial_id = f"expert_{timestamp.strftime('%Y%m%d_%H%M%S')}_{mode}"
        
        # JSONL 형식으로 데이터 준비
        jsonl_entry = {
            'trial_id': trial_id,
            'timestamp': timestamp.isoformat(),
            'mode': mode,
            **learning_data,
            'note': 'Expert learning data'
        }
        
        # JSONL 파일에 추가
        try:
            with open(LEARN_JSONL, 'a', encoding='utf-8') as f:
                f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
            self.logger.info(f"학습 데이터 추가 완료: {trial_id}")
        except Exception as e:
            self.logger.error(f"학습 데이터 저장 실패: {str(e)}")
            raise
    
    def _analyze_performance(self, predicted_scores, expert_scores, mode):
        """성능 분석"""
        
        # 공통 descriptor만 분석
        common_keys = set(predicted_scores.keys()) & set(expert_scores.keys())
        
        if not common_keys:
            return {'mae': 0.0, 'rmse': 0.0, 'error_count': 0}
        
        pred_values = [predicted_scores[k] for k in common_keys]
        expert_values = [expert_scores[k] for k in common_keys]
        
        # 성능 지표 계산
        errors = np.array(pred_values) - np.array(expert_values)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'error_count': len(errors),
            'max_error': float(np.max(np.abs(errors))),
            'common_descriptors': len(common_keys)
        }
    
    def _should_retrain(self, performance_metrics, mode):
        """재훈련 필요성 판단"""
        # MAE가 1.0 이상이면 재훈련 필요
        mae_threshold = 1.0
        
        # 최대 오차가 2.0 이상이면 재훈련 필요
        max_error_threshold = 2.0
        
        mae = performance_metrics.get('mae', 0.0)
        max_error = performance_metrics.get('max_error', 0.0)
        
        return mae > mae_threshold or max_error > max_error_threshold
    
    def _load_recent_trials(self, mode, hours=24):
        """최근 N시간 내 trial들 로드"""
        import json
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_trials = []
        
        try:
            with open(LEARN_JSONL, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('mode') == mode:
                        trial_time = datetime.fromisoformat(data.get('timestamp', ''))
                        if trial_time > cutoff_time:
                            recent_trials.append(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return recent_trials
    
    def _check_retrain_necessity(self, mode):
        """재훈련 필요성 확인"""
        # 최근 학습 데이터가 있으면 재훈련 필요
        recent_trials = self._load_recent_trials(mode, hours=24)
        return len(recent_trials) > 0
    
    def _load_training_data(self, mode):
        """훈련 데이터 로드"""
        import json
        
        training_data = []
        try:
            with open(LEARN_JSONL, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('mode') == mode:
                        training_data.append(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return training_data
    
    def _evaluate_model_improvement(self, mode, training_result):
        """모델 개선 평가"""
        if not training_result:
            return {'improvement': False, 'reason': 'Training failed'}
        
        # 기본적인 개선 평가
        return {
            'improvement': True,
            'training_loss': training_result.get('final_loss', 0.0),
            'validation_accuracy': training_result.get('val_accuracy', 0.0),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_dataset_statistics(self):
        """데이터셋 통계"""
        # 구현 필요
        return {}


# 싱글톤 인스턴스
integrated_system = IntegratedPredictionSystem()
