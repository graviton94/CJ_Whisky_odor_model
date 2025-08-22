#!/usr/bin/env python3
"""
모델 재훈련 스크립트 - 현재 특성 벡터 차원에 맞춤
"""

import sys
import os
import torch
import json
import numpy as np
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(__file__))

from train_finetune import train_model

def retrain_models():
    """현재 특성 벡터 차원에 맞춰 모델 재훈련"""
    
    print("🔄 모델 재훈련을 시작합니다...")
    print("=" * 50)
    
    # 1. Odor 모델 훈련
    print("\n1️⃣ Odor 모델 훈련 중...")
    try:
        train_model('odor')
        print("✅ Odor 모델 훈련 완료!")
    except Exception as e:
        print(f"❌ Odor 모델 훈련 실패: {str(e)}")
    
    # 2. Taste 모델 훈련  
    print("\n2️⃣ Taste 모델 훈련 중...")
    try:
        train_model('taste')
        print("✅ Taste 모델 훈련 완료!")
    except Exception as e:
        print(f"❌ Taste 모델 훈련 실패: {str(e)}")
    
    # 3. 훈련 결과 확인
    print("\n📋 훈련 결과 확인:")
    
    models_to_check = [
        ('odor', 'src/saved_models/odor_finetune.pth', 'src/saved_models/model_meta_odor.json'),
        ('taste', 'src/saved_models/taste_finetune.pth', 'src/saved_models/model_meta_taste.json')
    ]
    
    for mode, model_path, meta_path in models_to_check:
        try:
            # 모델 파일 확인
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                input_dim = state_dict['net.0.weight'].shape[1]
                output_dim = state_dict['net.6.weight'].shape[0]
                print(f"  {mode.upper()}: ✅ 모델 저장됨 (입력: {input_dim}차원, 출력: {output_dim}차원)")
                
                # 메타데이터 확인
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    print(f"    메타데이터: 입력 {meta['input_dim']}차원, 출력 {meta['output_dim']}차원")
                else:
                    print(f"    ⚠️ 메타데이터 파일 없음: {meta_path}")
            else:
                print(f"  {mode.upper()}: ❌ 모델 파일 없음: {model_path}")
                
        except Exception as e:
            print(f"  {mode.upper()}: ❌ 확인 실패: {str(e)}")
    
    print("\n🎉 모델 재훈련 프로세스 완료!")

if __name__ == "__main__":
    retrain_models()
