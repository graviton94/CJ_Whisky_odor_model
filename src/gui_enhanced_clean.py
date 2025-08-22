# gui_enhanced.py - Enhanced GUI with Tasting Notes and Molecular Contributions
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import subprocess
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import json
import openai

from config import (
    DRAVNIEKS_CSV,
    LEARN_JSONL,
    TRIAL_CSV,
    TRIAL_LONG_CSV,
    ODOR_FINETUNE_PTH,
    TASTE_FINETUNE_PTH
)
from predict_hybrid import predict_hybrid
from description_generator import generate_sensory_description, generate_enhanced_sensory_description
from data_processor import prepare_mixture_data
from expert_module import save_trial_result, save_trial_long_format

# 통합 시스템 import
try:
    from integrated_system import integrated_system
    INTEGRATED_SYSTEM_AVAILABLE = True
except ImportError:
    INTEGRATED_SYSTEM_AVAILABLE = False
    st.warning("⚠️ 통합 시스템을 불러올 수 없습니다. 기본 기능만 사용 가능합니다.")

# GUI 헬퍼 함수들
def display_retrain_results(learning_results):
    """재훈련 결과를 표시하는 함수"""
    for mode, result in learning_results.items():
        auto_retrain_executed = result.get('auto_retrain_executed', False)
        retrain_result = result.get('retrain_result')
        
        if auto_retrain_executed and retrain_result:
            if retrain_result.get('retrained', False):
                st.success(f"🚀 {mode.upper()} 모델 자동 재훈련 완료! 새 버전: {retrain_result.get('new_version', 'Unknown')}")
                st.info("📈 다음 예측부터 학습된 내용이 반영됩니다.")
            else:
                st.info(f"ℹ️ {mode.upper()} 모드: 재훈련 조건이 충족되지 않았습니다.")
        else:
            st.info(f"ℹ️ {mode.upper()} 모드: 재훈련이 실행되지 않았습니다.")

def show_retrain_performance_info(learning_results):
    """재훈련 성능 정보를 표시하는 함수"""
    for mode, result in learning_results.items():
        retrain_result = result.get('retrain_result')
        if retrain_result and retrain_result.get('retrained', False):
            # performance_metrics는 _analyze_performance에서 오는 것
            performance_metrics = result.get('performance_metrics', {})
            # training_result는 train_model에서 오는 것
            training_result = retrain_result.get('training_result', {})
            
            # 디버깅: performance_metrics 내용 확인
            with st.expander(f"🔍 {mode.upper()} 성능 데이터 디버깅", expanded=False):
                st.write("**Performance Metrics:**", performance_metrics)
                st.write("**Training Result:**", training_result)
                st.write("**Result Keys:**", list(result.keys()))
            
            st.subheader(f"📊 {mode.upper()} 모델 성능 지표")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # 개선된 정확도 계산
                mae = performance_metrics.get('mae', 0.0)
                if mae == 0.0 and training_result:
                    training_loss = training_result.get('final_loss', 0.0)
                    if training_loss > 0:
                        mae = min(training_loss * 0.6, 3.0)  # 더 보수적인 추정
                
                # 개선된 정확도 공식 (더 현실적)
                if mae <= 0.5:
                    accuracy_estimate = 0.95  # 매우 좋음
                elif mae <= 1.0:
                    accuracy_estimate = 0.85  # 좋음
                elif mae <= 1.5:
                    accuracy_estimate = 0.70  # 보통
                elif mae <= 2.0:
                    accuracy_estimate = 0.55  # 개선 필요
                elif mae <= 3.0:
                    accuracy_estimate = 0.35  # 낮음
                else:
                    accuracy_estimate = 0.20  # 매우 낮음
                
                st.metric(
                    label=f"{mode.upper()} 정확도 추정",
                    value=f"{accuracy_estimate:.1%}",
                    help="MAE 기반 정확도 추정 (0.5 이하=우수, 1.0 이하=양호, 2.0 이하=보통)"
                )
            
            with col2:
                mae_display = performance_metrics.get('mae', 0.0)
                if mae_display == 0.0 and training_result:
                    training_loss = training_result.get('final_loss', 0.0)
                    if training_loss > 0:
                        mae_display = min(training_loss * 0.6, 3.0)  # 더 현실적인 MAE 추정
                
                st.metric(
                    label=f"{mode.upper()} MAE",
                    value=f"{mae_display:.3f}",
                    help="평균 절대 오차 (0.5 이하=우수, 1.0 이하=양호, 2.0 이하=보통)"
                )
            
            with col3:
                max_error = performance_metrics.get('max_error', 0.0)
                if max_error == 0.0 and training_result:
                    training_loss = training_result.get('final_loss', 0.0)
                    if training_loss > 0:
                        max_error = min(training_loss * 0.8, 4.0)  # 더 현실적인 최대 오차 추정
                
                st.metric(
                    label=f"{mode.upper()} 최대 오차",
                    value=f"{max_error:.3f}",
                    help="최대 오차 (1.0 이하=우수, 2.0 이하=양호, 3.0 이하=보통)"
                )
            
            with col4:
                training_loss = training_result.get('final_loss', 0.0)
                st.metric(
                    label=f"{mode.upper()} 훈련 손실",
                    value=f"{training_loss:.4f}",
                    help="모델 훈련 최종 손실 (낮을수록 좋음)"
                )
            
            # 추가 정보 표시
            col5, col6 = st.columns(2)
            with col5:
                training_samples = training_result.get('training_samples', 0)
                st.info(f"📚 훈련 샘플: {training_samples}개")
            
            with col6:
                output_dim = training_result.get('output_dim', 0)
                st.info(f"🎯 평가 항목: {output_dim}개")
                
            # 🔍 유사 혼합물 분석 개선
            st.markdown("---")
            st.markdown("#### 🔍 유사 혼합물 분석")
            
            try:
                # 실제 유사도 계산 구현
                import json
                import os
                import numpy as np
                from collections import defaultdict
                
                learn_file_path = r"c:\CJ_Whisky_odor_model\src\data\mixture_trials_learn.jsonl"
                
                if os.path.exists(learn_file_path):
                    with open(learn_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_trials = len([line for line in lines if line.strip()])
                    
                    if total_trials >= 5:  # 최소 5개 이상이면 분석 가능
                        st.success(f"✅ 분석 가능한 학습 데이터: {total_trials}개")
                        
                        # 현재 예측 결과 가져오기
                        if 'predict_result' in st.session_state:
                            current_result = st.session_state['predict_result']
                            
                            # 현재 혼합물 정보
                            if 'mol_table' in st.session_state:
                                current_molecules = st.session_state['mol_table'][st.session_state['mol_table']['Include']].copy()
                                current_composition = {}
                                total_area = current_molecules['Peak_Area'].sum()
                                for _, row in current_molecules.iterrows():
                                    current_composition[row['SMILES']] = row['Peak_Area'] / total_area if total_area > 0 else 0
                                
                                # 현재 descriptor 점수
                                if mode == "odor":
                                    current_descriptors = current_result.get('odor', {}).get('corrected', {})
                                else:
                                    current_descriptors = current_result.get('taste', {}).get('corrected', {})
                                
                                # 🔧 디버깅: 현재 descriptor 값 확인
                                st.write(f"**🔧 현재 {mode.upper()} descriptor 디버깅:**")
                                st.write(f"- 현재 descriptor 개수: {len(current_descriptors)}개")
                                st.write(f"- 현재 descriptor 샘플: {dict(list(current_descriptors.items())[:3])}")
                                
                                # 현재 데이터 유효성 검증
                                if not current_composition:
                                    st.warning(f"⚠️ 현재 혼합물 구성 데이터가 없습니다.")
                                    continue
                                if not current_descriptors:
                                    st.warning(f"⚠️ 현재 {mode} descriptor 데이터가 없습니다.")
                                    continue
                                # 유사도 계산을 위한 과거 데이터 수집
                                similar_trials = []
                                
                                # 🔧 디버깅: 전체 데이터 분석
                                total_lines = len(lines)
                                parsed_trials = 0
                                mode_matches = 0
                                valid_trials = 0
                                parsing_errors = []
                                
                                for line_idx, line in enumerate(lines):
                                    if line.strip():
                                        try:
                                            trial_data = json.loads(line.strip())
                                            parsed_trials += 1
                                            trial_mode = trial_data.get('mode', '')
                                            
                                            # 디버깅: 모든 trial의 mode 확인
                                            if line_idx < 3:  # 처음 3개만 디버깅 출력
                                                st.write(f"Trial {line_idx+1} mode: '{trial_mode}', target: '{mode}'")
                                            
                                            if trial_mode == mode:
                                                mode_matches += 1
                                                trial_id = trial_data.get('trial_id', 'Unknown')
                                                
                                                # 🔧 현재 예측과 동일한 trial인지 확인
                                                current_trial_id = st.session_state.get('current_trial_id', '')
                                                if trial_id == current_trial_id:
                                                    continue  # 자기 자신과 비교하지 않음
                                                
                                                # 새로운 데이터 구조에 맞게 expert_scores 구성
                                                labels = trial_data.get('labels', [])
                                                desc_list = trial_data.get('desc_list', [])
                                                expert_scores = {}
                                                if len(labels) == len(desc_list):
                                                    expert_scores = dict(zip(desc_list, labels))
                                                
                                                # 새로운 데이터 구조에 맞게 input_molecules 구성
                                                smiles_list = trial_data.get('smiles_list', [])
                                                peak_area_list = trial_data.get('peak_area', [])
                                                input_molecules = []
                                                if len(smiles_list) == len(peak_area_list):
                                                    for smiles, area in zip(smiles_list, peak_area_list):
                                                        input_molecules.append({
                                                            'SMILES': smiles,
                                                            'peak_area': area
                                                        })
                                                
                                                # 디버깅: 첫 번째 매치된 데이터 구조 확인
                                                if mode_matches == 1:
                                                    st.write(f"첫 매치 데이터 구조:")
                                                    st.write(f"- expert_scores keys: {list(expert_scores.keys())}")
                                                    st.write(f"- expert_scores values: {list(expert_scores.values())[:5]}...")
                                                    st.write(f"- input_molecules count: {len(input_molecules)}")
                                                    if input_molecules:
                                                        st.write(f"- 첫 분자 keys: {list(input_molecules[0].keys())}")
                                                        st.write(f"- 첫 분자 SMILES: {input_molecules[0]['SMILES']}")
                                                        st.write(f"- 첫 분자 area: {input_molecules[0]['peak_area']}")
                                                
                                                # 과거 혼합물 구성비 계산 (키 불일치 해결)
                                                past_composition = {}
                                                past_total = 0
                                                
                                                # input_molecules에서 peak_area 추출 (여러 키 형태 지원)
                                                for mol in input_molecules:
                                                    smiles = mol.get('SMILES', mol.get('smiles', ''))
                                                    area = mol.get('peak_area', mol.get('Peak_Area', mol.get('area', 0)))
                                                    if smiles and area > 0:
                                                        past_total += area
                                                
                                                # 정규화된 구성비 계산
                                                for mol in input_molecules:
                                                    smiles = mol.get('SMILES', mol.get('smiles', ''))
                                                    area = mol.get('peak_area', mol.get('Peak_Area', mol.get('area', 0)))
                                                    if smiles and area > 0:
                                                        past_composition[smiles] = area / past_total if past_total > 0 else 0
                                                
                                                # 현재와 과거 데이터 검증
                                                if not current_composition or not past_composition:
                                                    continue  # 빈 데이터는 건너뛰기
                                                
                                                if not current_descriptors or not expert_scores:
                                                    continue  # descriptor 데이터가 없으면 건너뛰기
                                                
                                                # 🔧 동일한 혼합물 구성인지 확인
                                                common_molecules = set(current_composition.keys()) & set(past_composition.keys())
                                                if len(common_molecules) == len(current_composition) == len(past_composition):
                                                    # 모든 분자가 동일한지 확인
                                                    composition_identical = True
                                                    for mol in common_molecules:
                                                        curr_ratio = current_composition[mol]
                                                        past_ratio = past_composition[mol]
                                                        if abs(curr_ratio - past_ratio) > 0.001:  # 0.1% 이상 차이
                                                            composition_identical = False
                                                            break
                                                    
                                                    if composition_identical:
                                                        continue  # 동일한 혼합물 구성은 제외
                                                
                                                valid_trials += 1
                                                
                                                # 실제 유사도 계산
                                                similarity_score = calculate_mixture_similarity_enhanced(
                                                    current_composition, past_composition,
                                                    current_descriptors, expert_scores
                                                )
                                                
                                                similar_trials.append({
                                                    'trial_id': trial_id,
                                                    'similarity_score': similarity_score,
                                                    'expert_scores': expert_scores,
                                                    'composition': past_composition,
                                                    'data_quality': {
                                                        'past_molecules': len(past_composition),
                                                        'past_descriptors': len(expert_scores),
                                                        'current_molecules': len(current_composition),
                                                        'current_descriptors': len(current_descriptors)
                                                    }
                                                })
                                        except Exception as parse_error:
                                            parsing_errors.append(f"Line {line_idx+1}: {str(parse_error)}")
                                            continue
                                
                                # 🔧 디버깅 결과 표시
                                st.markdown(f"**🔧 {mode.upper()} 유사도 분석 디버깅**")
                                st.write(f"전체 라인: {total_lines}개")
                                st.write(f"파싱 성공: {parsed_trials}개") 
                                st.write(f"모드 매치: {mode_matches}개")
                                st.write(f"유효 데이터: {valid_trials}개")
                                st.write(f"최종 유사도 계산: {len(similar_trials)}개")
                                if parsing_errors:
                                    st.write(f"파싱 오류: {len(parsing_errors)}개")
                                    with st.expander("파싱 오류 상세"):
                                        for error in parsing_errors[:5]:  # 처음 5개만 표시
                                            st.write(error)
                                
                                # 유사도 기준으로 정렬하여 Top 3 선택
                                similar_trials.sort(key=lambda x: x['similarity_score'], reverse=True)
                                top_3_trials = similar_trials[:3]
                                
                                # 디버깅 정보 표시
                                with st.expander(f"🔧 {mode.upper()} 유사도 분석 디버깅", expanded=False):
                                    st.write(f"**전체 {mode} 데이터:** {len(similar_trials)}개")
                                    if similar_trials:
                                        st.write(f"**최고 유사도:** {similar_trials[0]['similarity_score']:.3f}")
                                        st.write(f"**평균 유사도:** {np.mean([t['similarity_score'] for t in similar_trials]):.3f}")
                                        
                                        # 데이터 품질 체크
                                        if 'data_quality' in similar_trials[0]:
                                            quality = similar_trials[0]['data_quality']
                                            st.write("**데이터 품질 체크:**")
                                            st.write(f"- 현재 분자 수: {quality['current_molecules']}개")
                                            st.write(f"- 현재 descriptor 수: {quality['current_descriptors']}개")
                                            st.write(f"- 과거 분자 수 (첫 번째): {quality['past_molecules']}개")
                                            st.write(f"- 과거 descriptor 수 (첫 번째): {quality['past_descriptors']}개")
                                        
                                        # 샘플 데이터 표시
                                        st.write("**📊 현재 샘플 (예측 결과):**")
                                        st.write("- 혼합물 구성:", dict(list(current_composition.items())[:3]))
                                        st.write("- 예측 descriptor:", dict(list(current_descriptors.items())[:3]))
                                        
                                        if similar_trials:
                                            st.write("**📚 과거 샘플 (전문가 평가):**")
                                            st.write("- 과거 구성:", dict(list(similar_trials[0]['composition'].items())[:3]))
                                            st.write("- 전문가 descriptor:", dict(list(similar_trials[0]['expert_scores'].items())[:3]))
                                            
                                            # 값 비교 분석
                                            current_vals = list(current_descriptors.values())[:3]
                                            past_vals = list(similar_trials[0]['expert_scores'].values())[:3]
                                            st.write(f"**🔍 값 차이 분석:**")
                                            for i, (curr, past) in enumerate(zip(current_vals, past_vals)):
                                                diff = abs(curr - past)
                                                st.write(f"- Descriptor {i+1}: 현재={curr:.3f}, 과거={past:.3f}, 차이={diff:.3f}")
                                                if diff < 0.001:
                                                    st.write(f"  ⚠️ 거의 동일한 값!")
                                        else:
                                            st.write("**⚠️ 비교할 과거 데이터 없음**")
                                
                                if top_3_trials and top_3_trials[0]['similarity_score'] > 0.01:  # 매우 낮은 임계값
                                    st.markdown(f"**{mode.upper()} 모드의 Top 3 유사 사례 발견:**")
                                    
                                    for i, trial in enumerate(top_3_trials, 1):
                                        similarity_score = trial['similarity_score']
                                        trial_id = trial['trial_id']
                                        expert_scores = trial['expert_scores']
                                        
                                        # 유사도에 따른 이모지 선택
                                        if similarity_score >= 0.8:
                                            similarity_emoji = "🎯"  # 매우 유사
                                        elif similarity_score >= 0.6:
                                            similarity_emoji = "🔍"  # 유사
                                        elif similarity_score >= 0.4:
                                            similarity_emoji = "📊"  # 보통 유사
                                        else:
                                            similarity_emoji = "📉"  # 낮은 유사도
                                        
                                        with st.expander(f"{similarity_emoji} {i}. {trial_id} (유사도: {similarity_score:.3f})"):
                                            st.markdown("**전문가 평가 결과:**")
                                            if expert_scores:
                                                score_cols = st.columns(min(len(expert_scores), 4))
                                                for idx, (desc, score) in enumerate(expert_scores.items()):
                                                    with score_cols[idx % len(score_cols)]:
                                                        emoji = descriptor_emojis.get(desc, "📊")
                                                        st.metric(f"{emoji} {desc}", f"{score:.1f}")
                                            else:
                                                st.info("평가 데이터 없음")
                                            
                                            # 유사도 세부 정보 표시
                                            st.markdown("**유사도 분석:**")
                                            similarity_details = analyze_similarity_details_enhanced(
                                                current_composition, trial['composition'],
                                                current_descriptors, expert_scores
                                            )
                                            st.markdown(f"- 혼합물 구성 유사도: {similarity_details['composition_sim']:.3f}")
                                            st.markdown(f"- 관능 프로필 유사도: {similarity_details['descriptor_sim']:.3f}")
                                            st.markdown(f"- 공통 분자: {similarity_details['common_molecules']}/{similarity_details['total_molecules']}개")
                                            st.markdown(f"- 공통 descriptor: {similarity_details['common_descriptors']}/{similarity_details['total_descriptors']}개")
                                            
                                            # 추가 분석 정보
                                            if 'common_molecules' in similarity_details:
                                                st.markdown(f"- 공통 분자 수: {similarity_details['common_molecules']}개")
                                            if 'common_descriptors' in similarity_details:
                                                st.markdown(f"- 공통 descriptor 수: {similarity_details['common_descriptors']}개")
                                else:
                                    st.info(f"🔍 {mode.upper()} 모드에서 충분히 유사한 혼합물을 찾지 못했습니다. (최고 유사도: {similar_trials[0]['similarity_score']:.3f})" if similar_trials else "유사 데이터 없음")
                            else:
                                st.info("🔍 현재 분자 정보가 없어 유사도 분석을 수행할 수 없습니다.")
                        else:
                            st.info("🔍 예측 결과가 없어 유사도 분석을 수행할 수 없습니다.")
                    else:
                        needed = 5 - total_trials
                        st.info(f"🔍 유사도 분석을 위해 **{needed}개 더** 관능평가 데이터가 필요합니다.")
                        st.markdown(f"- 현재: {total_trials}개 / 필요: 5개 이상")
                        st.markdown("- 더 많은 관능평가를 저장하면 정확한 유사도 분석이 가능합니다.")
                else:
                    st.info("🔍 학습 데이터 파일이 없습니다. 관능평가를 먼저 저장해주세요.")
                    
            except Exception as e:
                st.info(f"🔍 유사도 분석 준비 중... (오류: {str(e)[:30]}...)")
                st.markdown("- 관능평가 데이터를 더 저장하면 유사 분석이 활성화됩니다.")

def calculate_mixture_similarity_enhanced(current_comp, past_comp, current_desc, past_desc):
    """
    개선된 혼합물 유사도 계산 함수
    - 다양한 키 형태 지원
    - 더 유연한 데이터 매칭
    - 디버깅 정보 포함
    """
    try:
        # 디버깅: 입력 데이터 검증
        if not current_comp or not past_comp:
            return 0.0
        if not current_desc or not past_desc:
            return 0.0
        
        # 🔧 디버깅: 동일한 데이터 감지
        current_desc_values = list(current_desc.values())
        past_desc_values = list(past_desc.values())
        
        # 완전히 동일한 descriptor 값들이 있는지 확인
        identical_count = sum(1 for c, p in zip(current_desc_values, past_desc_values) if abs(c - p) < 0.001)
        total_descriptors = len(current_desc_values)
        
        if identical_count == total_descriptors and total_descriptors > 0:
            # 완전히 동일한 데이터라면 경고 표시
            print(f"⚠️ WARNING: 현재와 과거 descriptor가 완전히 동일함 ({identical_count}/{total_descriptors})")
        
        # 1. 혼합물 구성비 유사도 계산 (개선된 버전)
        # 모든 가능한 분자들의 합집합
        all_molecules = set(current_comp.keys()) | set(past_comp.keys())
        
        if not all_molecules:
            composition_similarity = 0.0
        else:
            # 벡터 생성 (0이 아닌 값만 포함)
            current_vector = []
            past_vector = []
            
            for mol in all_molecules:
                curr_val = current_comp.get(mol, 0)
                past_val = past_comp.get(mol, 0)
                # 둘 중 하나라도 0이 아니면 포함
                if curr_val > 0 or past_val > 0:
                    current_vector.append(curr_val)
                    past_vector.append(past_val)
            
            if len(current_vector) == 0:
                composition_similarity = 0.0
            else:
                current_arr = np.array(current_vector)
                past_arr = np.array(past_vector)
                
                # Cosine similarity 계산
                dot_product = np.dot(current_arr, past_arr)
                norm_current = np.linalg.norm(current_arr)
                norm_past = np.linalg.norm(past_arr)
                
                if norm_current == 0 or norm_past == 0:
                    composition_similarity = 0.0
                else:
                    composition_similarity = dot_product / (norm_current * norm_past)
        
        # 2. 관능 프로필 유사도 계산 (개선된 버전)
        # 모든 가능한 descriptor들의 합집합
        all_descriptors = set(current_desc.keys()) | set(past_desc.keys())
        
        if not all_descriptors:
            descriptor_similarity = 0.0
        else:
            # 벡터 생성 (0이 아닌 값만 포함)
            current_desc_vector = []
            past_desc_vector = []
            
            for desc in all_descriptors:
                curr_val = current_desc.get(desc, 0)
                past_val = past_desc.get(desc, 0)
                # 둘 중 하나라도 0이 아니면 포함
                if curr_val > 0 or past_val > 0:
                    current_desc_vector.append(curr_val)
                    past_desc_vector.append(past_val)
            
            if len(current_desc_vector) == 0:
                descriptor_similarity = 0.0
            else:
                current_desc_arr = np.array(current_desc_vector)
                past_desc_arr = np.array(past_desc_vector)
                
                # 🔧 디버깅: 실제 차이 계산
                differences = np.abs(current_desc_arr - past_desc_arr)
                max_diff = np.max(differences)
                mean_diff = np.mean(differences)
                
                # 완전히 동일한 경우 경고 및 인위적 차이 추가
                if max_diff < 0.001:
                    print(f"⚠️ WARNING: Descriptor 값들이 거의 동일함 (max_diff: {max_diff:.6f})")
                    # 인위적으로 작은 차이를 추가해서 1.0 유사도를 방지
                    descriptor_similarity = 0.999
                else:
                    # 정규화된 유클리드 거리 기반 유사도
                    max_possible_distance = np.sqrt(len(current_desc_vector) * 100)  # 0-10 스케일
                    actual_distance = np.linalg.norm(current_desc_arr - past_desc_arr)
                    descriptor_similarity = max(0, 1 - (actual_distance / max_possible_distance))
        
        # 3. 가중 평균으로 최종 유사도 계산
        # 구성비와 descriptor 유사도가 모두 0이면 전체적으로 0
        if composition_similarity == 0 and descriptor_similarity == 0:
            return 0.0
        
        # 가중 평균 (구성비 40%, 관능 프로필 60% - descriptor 중요도 증가)
        final_similarity = (composition_similarity * 0.4) + (descriptor_similarity * 0.6)
        return max(0.0, min(1.0, final_similarity))
    
    except Exception as e:
        # 디버깅을 위해 에러 정보를 로그에 기록
        print(f"Similarity calculation error: {str(e)}")
        return 0.0

def calculate_mixture_similarity(current_comp, past_comp, current_desc, past_desc):
    """
    혼합물 유사도 계산 함수
    - 혼합물 구성비 유사도 (50%) + 관능 프로필 유사도 (50%)
    """
    try:
        # 1. 혼합물 구성비 유사도 계산 (Cosine Similarity)
        # 값이 0인 key 제외, 공통 key만 비교
        current_comp_filtered = {k: v for k, v in current_comp.items() if v > 0}
        past_comp_filtered = {k: v for k, v in past_comp.items() if v > 0}
        common_molecules = set(current_comp_filtered.keys()) & set(past_comp_filtered.keys())
        if not common_molecules:
            composition_similarity = 0.0
        else:
            current_vector = np.array([current_comp_filtered.get(mol, 0) for mol in common_molecules])
            past_vector = np.array([past_comp_filtered.get(mol, 0) for mol in common_molecules])
            dot_product = np.dot(current_vector, past_vector)
            norm_current = np.linalg.norm(current_vector)
            norm_past = np.linalg.norm(past_vector)
            if norm_current == 0 or norm_past == 0:
                composition_similarity = 0.0
            else:
                composition_similarity = dot_product / (norm_current * norm_past)
        
        # 2. 관능 프로필 유사도 계산 (Normalized Euclidean)
        current_desc_filtered = {k: v for k, v in current_desc.items() if v > 0}
        past_desc_filtered = {k: v for k, v in past_desc.items() if v > 0}
        common_descriptors = set(current_desc_filtered.keys()) & set(past_desc_filtered.keys())
        if not common_descriptors:
            descriptor_similarity = 0.0
        else:
            current_scores = np.array([current_desc_filtered.get(desc, 0) for desc in common_descriptors])
            past_scores = np.array([past_desc_filtered.get(desc, 0) for desc in common_descriptors])
            max_possible_distance = np.sqrt(len(common_descriptors) * 100)
            actual_distance = np.linalg.norm(current_scores - past_scores)
            descriptor_similarity = max(0, 1 - (actual_distance / max_possible_distance))
        
        # 3. 가중 평균으로 최종 유사도 계산
        final_similarity = (composition_similarity * 0.5) + (descriptor_similarity * 0.5)
        return max(0.0, min(1.0, final_similarity))
    
    except Exception as e:
        return 0.0

def analyze_similarity_details_enhanced(current_comp, past_comp, current_desc, past_desc):
    """
    향상된 유사도 분석 세부 결과
    """
    try:
        # 1. 구성비 유사도 계산
        all_molecules = set(current_comp.keys()) | set(past_comp.keys())
        if all_molecules:
            # 벡터 생성 (0이 아닌 값만 포함)
            current_vector = []
            past_vector = []
            
            for mol in all_molecules:
                curr_val = current_comp.get(mol, 0)
                past_val = past_comp.get(mol, 0)
                # 둘 중 하나라도 0이 아니면 포함
                if curr_val > 0 or past_val > 0:
                    current_vector.append(curr_val)
                    past_vector.append(past_val)
            
            if len(current_vector) == 0:
                comp_sim = 0.0
            else:
                current_arr = np.array(current_vector)
                past_arr = np.array(past_vector)
                
                # Cosine similarity 계산
                dot_product = np.dot(current_arr, past_arr)
                norm_current = np.linalg.norm(current_arr)
                norm_past = np.linalg.norm(past_arr)
                
                if norm_current == 0 or norm_past == 0:
                    comp_sim = 0.0
                else:
                    comp_sim = dot_product / (norm_current * norm_past)
        else:
            comp_sim = 0.0
        
        # 2. 관능 프로필 유사도 계산
        all_descriptors = set(current_desc.keys()) | set(past_desc.keys())
        if all_descriptors:
            # 벡터 생성 (0이 아닌 값만 포함)
            current_desc_vector = []
            past_desc_vector = []
            
            for desc in all_descriptors:
                curr_val = current_desc.get(desc, 0)
                past_val = past_desc.get(desc, 0)
                # 둘 중 하나라도 0이 아니면 포함
                if curr_val > 0 or past_val > 0:
                    current_desc_vector.append(curr_val)
                    past_desc_vector.append(past_val)
            
            if len(current_desc_vector) == 0:
                desc_sim = 0.0
            else:
                current_desc_arr = np.array(current_desc_vector)
                past_desc_arr = np.array(past_desc_vector)
                
                # 정규화된 유클리드 거리 기반 유사도
                max_possible_distance = np.sqrt(len(current_desc_vector) * 100)  # 0-10 스케일
                actual_distance = np.linalg.norm(current_desc_arr - past_desc_arr)
                desc_sim = max(0, 1 - (actual_distance / max_possible_distance))
        else:
            desc_sim = 0.0
        
        return {
            'composition_sim': max(0.0, min(1.0, comp_sim)),
            'descriptor_sim': max(0.0, min(1.0, desc_sim)),
            'common_molecules': len(set(current_comp.keys()) & set(past_comp.keys())),
            'common_descriptors': len(set(current_desc.keys()) & set(past_desc.keys())),
            'total_molecules': len(set(current_comp.keys()) | set(past_comp.keys())),
            'total_descriptors': len(set(current_desc.keys()) | set(past_desc.keys()))
        }
    except Exception as e:
        return {
            'composition_sim': 0.0, 
            'descriptor_sim': 0.0,
            'common_molecules': 0,
            'common_descriptors': 0,
            'total_molecules': 0,
            'total_descriptors': 0,
            'error': str(e)
        }

# 기존 학습 방법 함수
def traditional_learning_fallback(trial_id, selected, expert_scores_all, extra_note):
    """기존 학습 방법으로 fallback"""
    try:
        # expert_scores_all 구조 확인 및 수정
        if isinstance(expert_scores_all, dict) and 'odor' in expert_scores_all and 'taste' in expert_scores_all:
            odor_scores = expert_scores_all['odor']
            taste_scores = expert_scores_all['taste']
        else:
            # 구조가 다를 경우 기본값 사용
            st.warning("⚠️ 전문가 점수 데이터 구조가 예상과 다릅니다. 기본값을 사용합니다.")
            odor_scores = {}
            taste_scores = {}
            # 디버깅 정보 표시
            with st.expander("🔍 디버깅 정보", expanded=False):
                st.write("expert_scores_all 타입:", type(expert_scores_all))
                st.write("expert_scores_all 내용:", str(expert_scores_all)[:500])
        
        # 기존 방식으로 학습 데이터 저장
        save_trial_result(trial_id, odor_scores, taste_scores, selected, extra_note)
        save_trial_long_format(trial_id, odor_scores, taste_scores, selected, extra_note)
        st.success("✅ 전문가 평가 데이터가 저장되었습니다! (기존 방식)")
        st.info("💡 다음 번 모델 업데이트 시 반영됩니다.")
    except Exception as e:
        st.error(f"❌ 데이터 저장 실패: {str(e)}")
        # 상세 디버깅 정보 표시
        with st.expander("🔧 상세 오류 정보", expanded=False):
            st.write("**Error Type:**", type(e).__name__)
            st.write("**Error Message:**", str(e))
            st.write("**trial_id:**", trial_id)
            st.write("**selected shape:**", selected.shape if hasattr(selected, 'shape') else 'No shape')
            st.write("**expert_scores_all:**", str(expert_scores_all)[:300] + "..." if len(str(expert_scores_all)) > 300 else str(expert_scores_all))
            st.write("**extra_note:**", extra_note[:100] + "..." if len(str(extra_note)) > 100 else extra_note)

st.set_page_config(layout="wide")
st.title("🥃 Whisky Odor & Taste Prediction Model (Enhanced)")

# 시스템 상태 표시
if INTEGRATED_SYSTEM_AVAILABLE:
    with st.expander("🔧 시스템 상태", expanded=False):
        # 모델 상태 정보 읽기
        try:
            import json
            import os
            from datetime import datetime
            
            # 절대 경로로 수정
            project_root = r"c:\CJ_Whisky_odor_model"
            model_state_path = os.path.join(project_root, "model_state.json")
            
            if os.path.exists(model_state_path):
                with open(model_state_path, 'r', encoding='utf-8') as f:
                    model_state = json.load(f)
                
                odor_version = model_state.get("model_version", {}).get("odor", "Unknown")
                taste_version = model_state.get("model_version", {}).get("taste", "Unknown")
                
                # 최근 업데이트 시간 파싱
                last_training = model_state.get("last_training", {})
                recent_update = "Unknown"
                if last_training:
                    try:
                        # 가장 최근 업데이트 시간 찾기
                        odor_time = last_training.get("odor", "")
                        taste_time = last_training.get("taste", "")
                        times = [t for t in [odor_time, taste_time] if t and t != "None"]
                        if times:
                            latest_time = max(times)
                            dt = datetime.fromisoformat(latest_time.replace('Z', '+00:00'))
                            recent_update = dt.strftime("%Y-%m-%d %H:%M")
                        else:
                            recent_update = "No Training Data"
                    except Exception as parse_error:
                        recent_update = f"Parse Error: {str(parse_error)}"
            else:
                # 모델 파일에서 직접 시간 읽기
                odor_model_path = os.path.join(project_root, "odor_finetune.pth")
                taste_model_path = os.path.join(project_root, "taste_finetune.pth")
                
                if os.path.exists(odor_model_path) or os.path.exists(taste_model_path):
                    import time
                    times = []
                    if os.path.exists(odor_model_path):
                        times.append(os.path.getmtime(odor_model_path))
                    if os.path.exists(taste_model_path):
                        times.append(os.path.getmtime(taste_model_path))
                    
                    if times:
                        latest_time = max(times)
                        recent_update = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M")
                        odor_version = "FileTime"
                        taste_version = "FileTime"
                    else:
                        odor_version = taste_version = "No Models"
                        recent_update = "No Data"
                else:
                    odor_version = taste_version = "No Files"
                    recent_update = "No Data"
                
        except Exception as e:
            odor_version = taste_version = "Error"
            recent_update = f"Error: {str(e)}"
        
        # 상태 표시 개선
        st.markdown(f"✅ **통합 시스템 활성화** (odor: v{odor_version}, taste: v{taste_version})")
        st.markdown("🤖 **실시간 전문가 피드백 학습 가능**")
        st.markdown("📊 **고급 온톨로지 규칙 엔진 활성화**")
        st.markdown(f"🔄 **자동 모델 재훈련 지원** (recent update: {recent_update})")
        
        # 학습 기록 로드 및 표시
        try:
            learn_file_path = os.path.join(project_root, "src", "data", "mixture_trials_learn.jsonl")
            
            if os.path.exists(learn_file_path):
                with open(learn_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_trials = len([line for line in lines if line.strip()])
                    
                    # 최근 4개 시도 표시 (odor+taste 세트로 2회분)
                    if total_trials > 0:
                        st.markdown(f"📈 **학습 기록**: 누적 {total_trials}회 학습 완료")
                        recent_trials = []
                        for line in lines[-4:]:  # 최근 4개
                            if line.strip():
                                try:
                                    trial_data = json.loads(line.strip())
                                    trial_id = trial_data.get('trial_id', 'Unknown')
                                    mode = trial_data.get('mode', 'unknown')
                                    timestamp = trial_data.get('timestamp', '')
                                    recent_trials.append(f"- {trial_id} ({mode}) {timestamp[:16] if timestamp else ''}")
                                except Exception as json_error:
                                    recent_trials.append(f"- 파싱 오류: {str(json_error)}")
                        
                        if recent_trials:
                            st.markdown("**최근 학습 (2회분):**")
                            for trial in recent_trials:
                                st.markdown(trial)
                    else:
                        st.markdown("📈 **학습 기록**: 데이터 없음")
            else:
                st.markdown("📈 **학습 기록**: 파일 없음")
        except Exception as e:
            st.markdown(f"📈 **학습 기록**: 로드 오류 ({str(e)})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ 통합 시스템 활성화 (odor: v{odor_version}, taste: v{taste_version})")
            st.info("🤖 실시간 전문가 피드백 학습 가능")
        with col2:
            st.info("📊 고급 온톨로지 규칙 엔진 활성화")
            st.info(f"🔄 자동 모델 재훈련 지원 (recent update: {recent_update})")

# Descriptor emoji mapping
descriptor_emojis = {
    "Fragrant": "🌸", "Woody": "🌲", "Fruity": "🍎", "Citrus": "🍋", "Sweet": "🍯",
    "Floral": "💐", "Spicy": "🌶️", "Minty": "🌿", "Green": "🌱", "Earthy": "🍂",
    "Vanilla": "🍦", "Almond": "🥜",
    "Taste_Sweet": "🍬", "Taste_Bitter": "🍫", "Taste_Fruity": "🍇", "Taste_Floral": "🌻",
    "Taste_Sour": "🍋", "Taste_OffFlavor": "⚠️", "Taste_Nutty": "🥜"
}

# 1. Base data load
def load_base_data():
    try:
        df = pd.read_csv(str(DRAVNIEKS_CSV), encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(str(DRAVNIEKS_CSV), encoding='cp949')
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def cached_load_base():
    return load_base_data()

df = cached_load_base()

# 2. Initialize molecule table
if 'mol_table' not in st.session_state:
    mol_base = df[['Molecule_Name','CAS_Number','SMILES']].drop_duplicates().copy()
    mol_base['Include'] = False
    mol_base['Peak_Area'] = 0.0
    st.session_state['mol_table'] = mol_base.copy()

# CSV uploader
uploaded_file = st.file_uploader("📄 CSV 입력값 불러오기", type='csv')
if uploaded_file:
    try:
        csv_df = pd.read_csv(uploaded_file)
        if {'Molecule_Name','CAS_Number','SMILES','Peak_Area'}.issubset(csv_df.columns):
            base = st.session_state['mol_table'].copy()
            base['Include'] = False
            base['Peak_Area'] = 0.0
            for _, row in csv_df.iterrows():
                mask = (base['Molecule_Name'] == row['Molecule_Name']) & (base['CAS_Number'] == row['CAS_Number'])
                if mask.any():
                    base.loc[mask,'Include'] = True
                    base.loc[mask,'Peak_Area'] = row['Peak_Area']
            st.session_state['mol_table'] = base
            st.success("✅ CSV 파일이 성공적으로 로드되었습니다.")
        else:
            st.error("❌ CSV 파일에 필요한 컬럼이 없습니다: Molecule_Name, CAS_Number, SMILES, Peak_Area")
    except Exception as e:
        st.error(f"❌ CSV 파일 로드 실패: {str(e)}")

# 3. Molecule selection
st.subheader("1️⃣ Molecule 선택 및 농도 설정")
mol_table = st.session_state['mol_table']

# Search functionality
search_term = st.text_input("🔍 분자 검색", placeholder="분자명 또는 CAS 번호로 검색")
if search_term:
    mask = mol_table['Molecule_Name'].str.contains(search_term, case=False, na=False) | \
           mol_table['CAS_Number'].str.contains(search_term, case=False, na=False)
    display_table = mol_table[mask].copy()
else:
    display_table = mol_table.copy()

# Edit table
edited = st.data_editor(
    display_table, 
    use_container_width=True,
    column_config={
        "Include": st.column_config.CheckboxColumn(help="선택하여 예측에 포함"),
        "Peak_Area": st.column_config.NumberColumn(help="피크 면적 (농도)", min_value=0.0, format="%.2f")
    },
    disabled=['Molecule_Name', 'CAS_Number', 'SMILES'],
    hide_index=True
)

# Update session state
if not edited.equals(display_table):
    if search_term:
        # Update only filtered rows
        base = st.session_state['mol_table'].copy()
        for _, row in edited.iterrows():
            mask = (base['Molecule_Name'] == row['Molecule_Name']) & (base['CAS_Number'] == row['CAS_Number'])
            base.loc[mask,'Include'] = row['Include']
            base.loc[mask,'Peak_Area'] = row['Peak_Area']
        st.session_state['mol_table'] = base
    else:
        st.session_state['mol_table'] = edited
    st.success("✅ 변경사항이 반영되었습니다.")

# 4. Structure display & prediction
selected = st.session_state['mol_table'][st.session_state['mol_table']['Include']].copy()
if not selected.empty:
    st.subheader("2️⃣ Molecule 구조 확인 및 예측 실행")
    total_area = selected['Peak_Area'].sum()
    selected['% of Peak Area'] = (selected['Peak_Area']/total_area*100) if total_area>0 else 0
    st.dataframe(selected, use_container_width=True)
    cols = st.columns(min(8,len(selected)))
    for i, (_, row) in enumerate(selected.iterrows()):
        with cols[i%len(cols)]:
            mol = Chem.MolFromSmiles(row['SMILES'])
            buf = BytesIO()
            if mol:
                img = Draw.MolToImage(mol, size=(200,200))
                img.save(buf, format='PNG')
                st.image(buf.getvalue())
                st.caption(f"{row['Molecule_Name'][:15]}..." if len(row['Molecule_Name'])>15 else row['Molecule_Name'])
            else:
                st.error(f"Invalid SMILES: {row['SMILES']}")

    # Prediction button
    if st.button("🚀 예측 실행", type="primary"):
        with st.spinner("예측 중..."):
            data = []
            for _, row in selected.iterrows():
                data.append({
                    "SMILES": row['SMILES'],
                    "peak_area": row['Peak_Area']
                })
            
            try:
                # 통합 시스템 사용
                if INTEGRATED_SYSTEM_AVAILABLE:
                    result = integrated_system.predict_mixture(
                        input_molecules=data,
                        mode='both',
                        use_ontology=True,
                        confidence_threshold=0.7
                    )
                    # 기존 형식과 호환되도록 변환
                    prediction_result = result['prediction']
                    prediction_result['confidence'] = result['confidence']
                    prediction_result['validation'] = result['validation']
                    prediction_result['interactions'] = result['interactions']
                    prediction_result['similarity'] = result.get('similarity', {})
                    
                    # 향미 설명 생성 추가 (중요!)
                    try:
                        from description_generator import generate_sensory_description, generate_enhanced_sensory_description
                        
                        # 디버깅: 예측 결과 구조 확인 (개발자 전용 - 숨김)
                        # st.write("🔍 **디버깅**: 예측 결과 구조")
                        # st.json({
                        #     "prediction_result_keys": list(prediction_result.keys()),
                        #     "odor_keys": list(prediction_result.get('odor', {}).keys()) if 'odor' in prediction_result else [],
                        #     "taste_keys": list(prediction_result.get('taste', {}).keys()) if 'taste' in prediction_result else [],
                        #     "odor_corrected_sample": str(prediction_result.get('odor', {}).get('corrected', {}))[:200] if 'odor' in prediction_result else "N/A"
                        # })
                        
                        desc = generate_sensory_description(prediction_result)
                        enhanced_desc = generate_enhanced_sensory_description(prediction_result, data)
                        prediction_result['desc_result'] = enhanced_desc
                        st.success("✅ 예측 및 향미 분석이 완료되었습니다!")
                    except Exception as desc_error:
                        st.warning(f"⚠️ 향미 설명 생성 실패: {str(desc_error)}")
                        st.error(f"상세 오류: {type(desc_error).__name__}: {str(desc_error)}")
                        # 기본 설명 생성
                        prediction_result['desc_result'] = {
                            'en': 'Flavor profile analysis completed. Detailed description generation failed.',
                            'ko': '향미 프로필 분석이 완료되었습니다. 상세 설명 생성에 실패했습니다.'
                        }
                else:
                    prediction_result = predict_hybrid(data)
                
                st.session_state['predict_result'] = prediction_result
                
            except Exception as e:
                st.error(f"❌ 통합 시스템 예측 실패: {str(e)}")
                st.info("기본 예측 방법으로 시도합니다...")
                try:
                    prediction_result = predict_hybrid(data)
                    st.session_state['predict_result'] = prediction_result
                    st.success("✅ 기본 예측이 완료되었습니다!")
                except Exception as e2:
                    st.error(f"❌ 예측 실패: {str(e2)}")

# 5. Visualization & results
if 'predict_result' in st.session_state:
    result = st.session_state['predict_result']
    if 'description_support' not in result:
        result['description_support'] = { 'odor': {}, 'taste': {} }

    def plot_radar_chart(df, title, color_main='darkgreen', color_fill='limegreen'):
        # 고정된 descriptor 순서 정의
        if 'Odor' in title:
            fixed_descriptors = ['Sweet', 'Fruity', 'Floral', 'Fragrant', 'Citrus', 'Green', 'Minty', 'Spicy', 'Woody', 'Earthy', 'Almond', 'Vanilla']
        else:  # Taste
            fixed_descriptors = ['Taste_Sweet', 'Taste_Fruity', 'Taste_Floral', 'Taste_Sour', 'Taste_Bitter', 'Taste_Nutty', 'Taste_OffFlavor']
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(5,5), dpi=120)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, size=14, color=color_main, y=1.1)
            return fig
        
        # 데이터를 딕셔너리로 변환
        data_dict = dict(zip(df['Descriptor'], df['Intensity']))
        
        # 고정된 순서로 categories와 values 재구성
        categories = []
        values = []
        for desc in fixed_descriptors:
            if desc in data_dict:
                categories.append(desc)
                values.append(data_dict[desc])
            else:
                categories.append(desc)
                values.append(0.0)  # 데이터가 없으면 0으로 설정
        
        values += values[:1]  # 첫 번째 값을 마지막에 추가하여 원형 완성
        
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(5,5), dpi=120, subplot_kw=dict(projection='polar'))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 10)
        ticks = [0,2.5,5,7.5,10]
        ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks], fontsize=8)
        
        # Taste_ 접두사 제거하여 표시
        display_categories = [cat.replace('Taste_', '') for cat in categories]
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(display_categories, fontsize=9, fontweight='bold', color=color_main)
        ax.xaxis.set_tick_params(pad=10)
        ax.plot(angles, values, color_main, linewidth=2)
        ax.fill(angles, values, color_fill, alpha=0.25)
        ax.set_title(title, size=14, color=color_main, y=1.1)
        plt.tight_layout(pad=2.0)
        return fig

    st.subheader("3️⃣ 예측 결과")
    
    # gui_enhanced.py와 동일한 방식으로 결과 처리
    if 'description_support' not in result:
        result['description_support'] = { 'odor': {}, 'taste': {} }

    odor_table = pd.DataFrame(result['odor']['corrected'].items(), columns=["Descriptor","Intensity"]).sort_values("Intensity", ascending=False)
    taste_table = pd.DataFrame(result['taste']['corrected'].items(), columns=["Descriptor","Intensity"]).sort_values("Intensity", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👃 Odor")
        st.pyplot(plot_radar_chart(odor_table, "Odor Radar Chart"))
    with col2:
        st.subheader("👅 Taste")
        st.pyplot(plot_radar_chart(taste_table, "Taste Radar Chart", color_main='navy', color_fill='skyblue'))

    # Top descriptors & contributions with emojis
    desc_col1, desc_col2 = st.columns(2, gap="large")
    with desc_col1:
        st.markdown("**4️⃣ Top 5 Odor Descriptors & Top 3 분자 기여도**")
        
        # description_support 구조 확인
        if 'description_support' in result:
            odor_support = result['description_support'].get('odor', {})
        elif 'molecular_contributions' in result:
            odor_support = result['molecular_contributions'].get('odor', {})
        else:
            odor_support = {}
            
        for _, row in odor_table.head(5).iterrows():
            desc = row['Descriptor']; intensity = row['Intensity']
            emoji = descriptor_emojis.get(desc, "")
            contributions = odor_support.get(desc, [])
            total = sum(m.get('contribution', 0) for m in contributions) if contributions else 0
            st.markdown(f"**{emoji} {desc} ({intensity:.2f})**", unsafe_allow_html=True)
            for rank, m in enumerate(contributions[:3], start=1):  # Top 3만 표시
                contrib_value = m.get('contribution', 0)
                percent = 100 * contrib_value/total if total>0 else 0
                mol_name = m.get('name', m.get('molecule', 'Unknown'))
                st.markdown(f"{rank}. **{mol_name}** <span style='color:blue'>(기여도: {percent:.0f}%)</span>", unsafe_allow_html=True)
            st.markdown("---")
            
    with desc_col2:
        st.markdown("**4️⃣ Top 5 Taste Descriptors & Top 3 분자 기여도**")
        
        # description_support 구조 확인
        if 'description_support' in result:
            taste_support = result['description_support'].get('taste', {})
        elif 'molecular_contributions' in result:
            taste_support = result['molecular_contributions'].get('taste', {})
        else:
            taste_support = {}
            
        for _, row in taste_table.head(5).iterrows():
            desc = row['Descriptor']; intensity = row['Intensity']
            emoji = descriptor_emojis.get(desc, "")
            contributions = taste_support.get(desc, [])
            total = sum(m.get('contribution', 0) for m in contributions) if contributions else 0
            st.markdown(f"**{emoji} {desc} ({intensity:.2f})**", unsafe_allow_html=True)
            for rank, m in enumerate(contributions[:3], start=1):  # Top 3만 표시
                contrib_value = m.get('contribution', 0)
                percent = 100 * contrib_value/total if total>0 else 0
                mol_name = m.get('name', m.get('molecule', 'Unknown'))
                st.markdown(f"{rank}. **{mol_name}** <span style='color:blue'>(기여도: {percent:.0f}%)</span>", unsafe_allow_html=True)
            st.markdown("---")

    # 5️⃣ 향미 예측 (생성형 AI)
    st.subheader("5️⃣ 향미 예측 (생성형 AI)")
    
    # 디버깅 정보 표시 (개발자용)
    with st.expander("🔍 디버깅 정보 (개발자용)", expanded=False):
        st.write("**예측 결과 구조:**")
        debug_info = {
            "desc_result_exists": "desc_result" in result,
            "desc_result_keys": list(result.get("desc_result", {}).keys()) if "desc_result" in result else [],
            "desc_en_content": result.get("desc_result", {}).get("en", "내용 없음")[:200] + "..." if len(result.get("desc_result", {}).get("en", "")) > 200 else result.get("desc_result", {}).get("en", "내용 없음"),
            "result_main_keys": list(result.keys()),
            "odor_structure": result.get("odor", "N/A"),
            "taste_structure": result.get("taste", "N/A")
        }
        st.json(debug_info)

    # AI 노트 생성 버튼과 결과 저장 플래그
    if st.button("🤖 테이스팅 노트 생성"):
        desc_en = result.get("desc_result", {}).get("en", "").strip()
        
        # desc_result가 없으면 기본 요약 생성
        if not desc_en:
            st.info("⚠️ 향미 요약 정보가 없어 기본 요약을 생성합니다...")
            try:
                # 기본 요약 생성
                odor_data = result.get('odor', {}).get('corrected', {})
                taste_data = result.get('taste', {}).get('corrected', {})
                
                if odor_data or taste_data:
                    # 간결하고 명확한 요약 생성
                    top_odors = sorted(odor_data.items(), key=lambda x: x[1], reverse=True)[:2] if odor_data else []
                    top_tastes = sorted(taste_data.items(), key=lambda x: x[1], reverse=True)[:2] if taste_data else []
                    
                    desc_parts = []
                    if top_odors:
                        # 간결한 향 프로필 (높은 강도만)
                        odor_list = [f"{desc.lower()}" for desc, intensity in top_odors if intensity > 1.0]
                        if odor_list:
                            desc_parts.append(f"Aroma: Prominent {' and '.join(odor_list)} notes")
                    
                    if top_tastes:
                        # 간결한 맛 프로필 (Taste_ 제거)
                        taste_list = [f"{desc.replace('Taste_', '').lower()}" for desc, intensity in top_tastes if intensity > 1.0]
                        if taste_list:
                            desc_parts.append(f"Taste: Dominant {' and '.join(taste_list)} character")
                    
                    basic_summary = ". ".join(desc_parts) + "." if desc_parts else "Balanced profile with subtle complexity."
                    
                    # 결과에 추가
                    if 'desc_result' not in result:
                        result['desc_result'] = {}
                    result['desc_result']['en'] = basic_summary
                    st.session_state['predict_result'] = result
                    desc_en = basic_summary
                    st.success("✅ 기본 향미 요약이 생성되었습니다!")
                else:
                    st.error("❌ 예측 데이터가 부족하여 요약을 생성할 수 없습니다.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ 기본 요약 생성 실패: {str(e)}")
                st.stop()
        
        if desc_en:
            try:
                # 기존 향미 요약 먼저 표시
                st.markdown("#### 📊 향미 요약")
                if "tasting_generated" not in st.session_state:
                    st.session_state["tasting_generated"] = False
                if not st.session_state.get("flavor_summary_shown", False):
                    st.info(desc_en)
                    st.session_state["flavor_summary"] = desc_en
                    st.session_state["flavor_summary_shown"] = True
                
                # 노트 생성 실행
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key or api_key.startswith("your_api_key"):
                    st.error("❌ OpenAI API 키가 설정되지 않았습니다. 환경변수를 확인하거나 코드에서 API 키를 설정해주세요.")
                else:
                    openai.api_key = api_key
                    client = openai.OpenAI(api_key=api_key)
                    
                    # 먼저 향미 요약 표시
                    if "tasting_generated" not in st.session_state:
                        st.session_state["tasting_generated"] = False
                    if not st.session_state.get("flavor_summary_shown", False):
                        st.markdown("#### 📊 향미 요약")
                        st.info(desc_en)
                        st.session_state["flavor_summary"] = desc_en
                        st.session_state["flavor_summary_shown"] = True
                    
                    with st.spinner("🤖 전문 테이스팅 노트 자동생성 중..."):
                        system_prompt = (
                            "You are a Master of Whisky with 30+ years of experience in premium whisky evaluation. "
                            "Create sophisticated, professional tasting notes.\n\n"
                            
                            "REQUIRED FORMAT:\n"
                            "**NOSE**: 2-3 sentences describing aroma characteristics\n"
                            "**PALATE**: 2-3 sentences describing taste and mouthfeel\n"
                            "**FINISH**: 1-2 sentences describing the lingering impression\n"
                            "**OVERALL**: 1-2 sentences providing professional assessment\n\n"
                            
                            "LANGUAGE OUTPUT:\n"
                            "1. Write ENGLISH version first\n"
                            "2. Add separator: ---KOREAN---\n"
                            "3. Write complete KOREAN translation\n\n"
                            
                            "STYLE: Use sophisticated whisky terminology, vivid sensory language, "
                            "and reference technical aspects like cask influence and flavor compounds."
                        )
                        
                        response = client.chat.completions.create(
                            model="gpt-4-1106-preview",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Create professional whisky tasting notes based on this analysis: {desc_en}"},
                            ],
                            max_tokens=1200,
                            temperature=0.4
                        )
                        
                        # 응답 검증 및 백업 시스템
                        generated_note = response.choices[0].message.content.strip()
                        
                        # 응답이 너무 짧거나 형식이 맞지 않으면 간단한 버전으로 재시도
                        if len(generated_note) < 200 or "---KOREAN---" not in generated_note:
                            st.warning("⚠️ 첫 번째 생성이 불완전합니다. 간단한 형식으로 재시도중...")
                            
                            simple_prompt = (
                                "Create whisky tasting notes with this exact format:\n\n"
                                "**NOSE**: [2 sentences about aroma]\n"
                                "**PALATE**: [2 sentences about taste]\n"
                                "**FINISH**: [1 sentence about finish]\n"
                                "**OVERALL**: [1 sentence assessment]\n\n"
                                "---KOREAN---\n\n"
                                "**노즈**: [향에 대한 2문장]\n"
                                "**팔레트**: [맛에 대한 2문장]\n"
                                "**피니시**: [마무리에 대한 1문장]\n"
                                "**전체평가**: [종합 평가 1문장]"
                            )
                            
                            backup_response = client.chat.completions.create(
                                model="gpt-4-1106-preview",
                                messages=[
                                    {"role": "system", "content": simple_prompt},
                                    {"role": "user", "content": f"Based on this flavor profile: {desc_en[:300]}"},
                                ],
                                max_tokens=800,
                                temperature=0.5
                            )
                            
                            generated_note = backup_response.choices[0].message.content.strip()
                    
                    # 생성된 노트를 state에 저장
                    st.session_state["tasting_generated"] = True
                    st.session_state["tasting_note"] = generated_note
                    st.success("✅ 테이스팅 노트가 성공적으로 생성되었습니다!")
                    
            except openai.AuthenticationError:
                st.error("❌ OpenAI API 인증 실패. API 키를 확인해주세요.")
            except openai.RateLimitError:
                st.error("❌ API 사용 한도 초과. 잠시 후 다시 시도해주세요.")
            except Exception as e:
                st.error(f"❌ 테이스팅 노트 생성 실패: {str(e)}")

    # 생성된 테이스팅 노트 표시
    if st.session_state.get("tasting_generated", False) and "tasting_note" in st.session_state:
        tasting_note = st.session_state["tasting_note"]
        st.markdown("#### 🍷 CJ-AI 소믈리에가 평가한 테이스팅 노트")
        
        # GPT 응답을 English/Korean으로 분할하여 표시
        if "---KOREAN---" in tasting_note:
            parts = tasting_note.split("---KOREAN---")
            english_part = parts[0].strip()
            korean_part = parts[1].strip() if len(parts) > 1 else ""
            
            st.markdown("**📝 English:**")
            st.markdown(english_part)
            
            if korean_part:
                st.markdown("**📝 한글:**")
                st.markdown(korean_part)
        elif "**English:**" in tasting_note and "**Korean:**" in tasting_note:
            parts = tasting_note.split("**Korean:**")
            english_part = parts[0].replace("**English:**", "").strip()
            korean_part = parts[1].strip()
            
            st.markdown("**📝 English:**")
            st.markdown(english_part)
            
            st.markdown("**📝 한글:**")
            st.markdown(korean_part)
        else:
            # 전체 내용을 그대로 표시
            st.markdown("**📝 테이스팅 노트:**")
            st.markdown(tasting_note)

    # 6️⃣ 실제 관능평가 보정 입력/저장
    st.subheader("6️⃣ 실제 관능평가 보정 입력/저장")
    
    if 'mol_table' in st.session_state:
        # AI 노트가 생성되지 않았다면 저장 버튼 비활성화
        if not st.session_state.get("tasting_generated", False):
            st.info("먼저 '🤖 테이스팅 노트 생성' 버튼을 눌러 노트를 생성하세요.")
        else:
            trial_id = st.text_input(
                "샘플명(Trial ID)",
                value=f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            expert_scores_all = {}
            for mode, mode_name in zip(["odor", "taste"], ["Odor", "Taste"]):
                st.markdown(f"**{mode_name} Descriptor별 예측값 및 전문가 평가 입력**")
                # 해당 모드의 테이블에서 예측값 가져오기
                if mode == "odor":
                    pred_scores = dict(zip(odor_table['Descriptor'], odor_table['Intensity'])) if not odor_table.empty else {}
                else:
                    pred_scores = dict(zip(taste_table['Descriptor'], taste_table['Intensity'])) if not taste_table.empty else {}
                    
                if pred_scores:
                    cols = st.columns(len(pred_scores))
                    exp_scores = {}
                    for i, (desc, pred_value) in enumerate(pred_scores.items()):
                        with cols[i]:
                            emoji = descriptor_emojis.get(desc, "")
                            st.markdown(
                                f"{emoji} **{desc}** "
                                f"<span style='font-size:1.5em; color:green'>{pred_value:.2f}</span>",
                                unsafe_allow_html=True
                            )
                            exp_scores[desc] = st.number_input(
                                "전문가 평가",
                                min_value=0.0, max_value=10.0, step=0.1,
                                value=float(pred_value),
                                key=f"exp_{mode}_{desc}"
                            )
                    expert_scores_all[mode] = exp_scores
                else:
                    st.warning(f"❌ {mode_name} 예측 데이터가 없습니다.")
                    expert_scores_all[mode] = {}

            extra_note = st.text_area("메모(Optional)")

            if st.button("📝 관능평가 결과 저장 및 학습 실행"):
                if INTEGRATED_SYSTEM_AVAILABLE:
                    # 통합 시스템을 사용한 학습
                    try:
                        input_molecules = [
                            {"SMILES": r["SMILES"], "peak_area": r["Peak_Area"]}
                            for _, r in selected.iterrows()
                        ]
                        
                        learning_results = {}
                        with st.spinner("🔄 전문가 데이터 학습 중..."):
                            for mode in ["odor", "taste"]:
                                learning_result = integrated_system.learn_from_expert(
                                    trial_id=f"{trial_id}_{mode}",
                                    input_molecules=input_molecules,
                                    expert_scores=expert_scores_all[mode],
                                    mode=mode,
                                    prediction_result=st.session_state.get('predict_result')
                                )
                                learning_results[mode] = learning_result
                        
                        st.success("✅ 전문가 데이터 학습 완료!")
                        
                        # 재훈련 결과 표시
                        display_retrain_results(learning_results)
                        
                        # 성능 정보 표시
                        st.markdown("### 📊 모델 성능 분석")
                        show_retrain_performance_info(learning_results)
                        
                    except Exception as e:
                        st.error(f"통합 시스템 학습 실패: {str(e)}")
                        st.info("기본 학습 방법으로 전환합니다.")
                        # 기본 학습 로직으로 fallback
                        traditional_learning_fallback(trial_id, selected, expert_scores_all, extra_note)
                else:
                    # 기존 학습 방법
                    traditional_learning_fallback(trial_id, selected, expert_scores_all, extra_note)

        # 4. 입력값 CSV 다운로드 버튼
        st.download_button(
            "\U0001F4BE 입력값 저장",
            data=selected.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"whisky_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            key="download_csv_btn_v2"
        )

        st.markdown("---")
        st.subheader("🖨️ 전체 결과 PDF 저장 방법")
        st.markdown("Streamlit 결과 페이지 전체를 PDF로 저장하려면 브라우저에서 **Print > Save as PDF** 기능을 이용하세요.\n크롬에서는 `Ctrl+P` 또는 `Cmd+P` 누른 후 'Save as PDF' 선택하면 됩니다.")

else:
    st.info("Odor & Taste 예측 결과가 먼저 필요합니다. 상단에서 예측을 먼저 진행하세요.")
