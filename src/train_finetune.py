import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from data_processor import build_input_vector
from config import (
    LEARN_JSONL,
    ODOR_FINETUNE_PTH,
    TASTE_FINETUNE_PTH,
    MODEL_META_ODOR,
    MODEL_META_TASTE,
    TF_BATCH_SIZE,
    TF_EPOCHS,
    TF_LR
)

# Ensure model directory exists
for path in [ODOR_FINETUNE_PTH, TASTE_FINETUNE_PTH, MODEL_META_ODOR, MODEL_META_TASTE]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

class FinetuneMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Hardtanh(min_val=0.0, max_val=10.0)
        )
    def forward(self, x):
        return self.net(x)


def detect_style_conflicts(X, y, desc_list):
    """
    학습 데이터에서 스타일 충돌 감지
    Returns: (has_conflict, clusters, cluster_info)
    """
    if len(X) < 6:  # 최소 데이터 수
        return False, None, "Insufficient data for conflict detection"
    
    try:
        # 특성 벡터로 클러스터링
        kmeans = KMeans(n_clusters=min(3, len(X)//2), random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # 클러스터별 descriptor 평균 계산
        cluster_profiles = {}
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_y = y[cluster_mask]
            avg_scores = np.mean(cluster_y, axis=0)
            cluster_profiles[cluster_id] = {
                'scores': avg_scores,
                'descriptors': {desc_list[i]: avg_scores[i] for i in range(len(desc_list))},
                'count': np.sum(cluster_mask)
            }
        
        # 클러스터 간 차이 분석
        max_diff = 0
        conflicting_descriptors = []
        
        for i, cluster1 in cluster_profiles.items():
            for j, cluster2 in cluster_profiles.items():
                if i < j:  # 중복 방지
                    diff = np.abs(cluster1['scores'] - cluster2['scores'])
                    max_diff = max(max_diff, np.max(diff))
                    
                    # 큰 차이를 보이는 descriptor 찾기
                    for idx, d in enumerate(diff):
                        if d > 3.0:  # 3점 이상 차이
                            conflicting_descriptors.append({
                                'descriptor': desc_list[idx],
                                'cluster1_score': cluster1['scores'][idx],
                                'cluster2_score': cluster2['scores'][idx],
                                'difference': d
                            })
        
        has_conflict = max_diff > 3.0 and len(conflicting_descriptors) > 0
        
        conflict_info = {
            'max_difference': max_diff,
            'conflicting_descriptors': conflicting_descriptors,
            'cluster_profiles': cluster_profiles,
            'clusters': clusters
        }
        
        return has_conflict, clusters, conflict_info
        
    except Exception as e:
        return False, None, f"Conflict detection failed: {str(e)}"


def calculate_sample_weights(X, y, clusters=None):
    """
    샘플별 가중치 계산 - 최근 데이터와 유사한 스타일에 더 높은 가중치
    다중 스타일 적응: A → B → C 스타일 변화에 대응
    """
    weights = np.ones(len(X))
    
    if clusters is not None:
        # 최근 20개 샘플 분석 (더 긴 패턴 감지)
        recent_samples = min(20, len(X))
        recent_clusters = clusters[-recent_samples:]
        recent_cluster_counts = np.bincount(recent_clusters)
        
        # 시간 가중치: 더 최근일수록 높은 가중치
        time_decay = np.exp(-np.arange(len(X))[::-1] / (len(X) * 0.3))  # 지수 감소
        
        # 클러스터 진화 패턴 감지
        cluster_evolution = analyze_cluster_evolution(clusters)
        
        for i, cluster in enumerate(clusters):
            if cluster < len(recent_cluster_counts):
                # 1. 최근 등장 빈도
                recent_weight = recent_cluster_counts[cluster] / recent_samples
                
                # 2. 시간 가중치
                temporal_weight = time_decay[i]
                
                # 3. 클러스터 진화 가중치
                evolution_weight = cluster_evolution.get(cluster, 1.0)
                
                # 최종 가중치 (0.2~2.0 범위)
                final_weight = 0.2 + (recent_weight * temporal_weight * evolution_weight) * 1.8
                weights[i] = final_weight
                
        print(f"   📊 가중치 분포: 평균={np.mean(weights):.2f}, 범위={np.min(weights):.2f}~{np.max(weights):.2f}")
    
    return weights


def analyze_cluster_evolution(clusters):
    """
    클러스터 진화 패턴 분석: 새로운 스타일의 등장과 변화 감지
    """
    if len(clusters) < 10:
        return {cluster: 1.0 for cluster in np.unique(clusters)}
    
    # 시간 구간별 클러스터 분포 분석
    n_segments = min(5, len(clusters) // 4)
    segment_size = len(clusters) // n_segments
    
    segment_distributions = []
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < n_segments - 1 else len(clusters)
        segment = clusters[start_idx:end_idx]
        
        # 각 세그먼트의 클러스터 분포
        unique, counts = np.unique(segment, return_counts=True)
        distribution = {cluster: count / len(segment) for cluster, count in zip(unique, counts)}
        segment_distributions.append(distribution)
    
    # 진화 가중치 계산
    evolution_weights = {}
    all_clusters = np.unique(clusters)
    
    for cluster in all_clusters:
        appearances = []
        for dist in segment_distributions:
            appearances.append(dist.get(cluster, 0.0))
        
        # 패턴 분석
        if len(appearances) >= 3:
            recent_trend = np.mean(appearances[-2:])  # 최근 2개 구간 평균
            early_trend = np.mean(appearances[:2])    # 초기 2개 구간 평균
            
            if recent_trend > early_trend:
                # 상승 트렌드 (새로운 스타일) - 높은 가중치
                evolution_weights[cluster] = 1.0 + (recent_trend - early_trend) * 2.0
            elif recent_trend < early_trend * 0.5:
                # 급격한 하락 (구식 스타일) - 낮은 가중치
                evolution_weights[cluster] = 0.5
            else:
                # 안정적 패턴
                evolution_weights[cluster] = 1.0
        else:
            evolution_weights[cluster] = 1.0
    
    return evolution_weights


def multi_style_conflict_analysis(X, y, desc_list):
    """
    다중 스타일 충돌 분석 및 전략 결정
    """
    has_conflict, clusters, conflict_info = detect_style_conflicts(X, y, desc_list)
    
    if not has_conflict:
        return {
            'strategy': 'unified',
            'description': '스타일 일관성 확인 - 통합 학습',
            'clusters': None,
            'weights': np.ones(len(X))
        }
    
    n_clusters = len(np.unique(clusters))
    cluster_evolution = analyze_cluster_evolution(clusters)
    
    # 전략 결정
    if n_clusters == 2:
        strategy = 'binary_adaptive'
        description = f'A/B 스타일 충돌 - 최근 스타일 우선'
    elif n_clusters == 3:
        strategy = 'triple_evolution'
        description = f'A/B/C 삼중 스타일 - 진화 패턴 기반'
    else:
        strategy = 'multi_cluster'
        description = f'{n_clusters}개 스타일 혼재 - 복합 적응'
    
    weights = calculate_sample_weights(X, y, clusters)
    
    return {
        'strategy': strategy,
        'description': description,
        'clusters': clusters,
        'cluster_evolution': cluster_evolution,
        'weights': weights,
        'conflict_info': conflict_info
    }


def train_model(mode: str):
    assert mode in ['odor', 'taste'], "mode must be 'odor' or 'taste'"

    if not os.path.exists(LEARN_JSONL):
        print(f"🚫 학습 데이터 파일이 없습니다: {LEARN_JSONL}")
        return {
            'success': False,
            'reason': f'Learning data file not found: {LEARN_JSONL}'
        }

    # Load JSONL data
    with open(LEARN_JSONL, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    rows_mode = [r for r in rows if r.get('mode') == mode]
    if not rows_mode:
        print(f"🚫 '{mode}' 모드의 학습 데이터가 없습니다.")
        return {
            'success': False,
            'reason': f'No training data found for mode: {mode}'
        }

    X_list, y_list = [], []
    desc_list_final = None
    bad_idx = []
    max_dim = 0
    
    # 1차: 모든 row를 처리하여 최대 차원 찾기
    temp_vectors = []
    dimension_info = {}
    
    for i, row in enumerate(rows_mode):
        try:
            smiles_list = row['smiles_list']
            peak_area = row['peak_area']
            labels = row['labels']
            desc_list = row['desc_list']
            if desc_list_final is None:
                desc_list_final = desc_list
            elif desc_list != desc_list_final:
                raise ValueError(f"Descriptor 순서 불일치! (row {i})")

            # 실제 feature vector 생성 (항상 현재 알고리즘 사용)
            vec = build_input_vector(smiles_list, peak_area, mode=mode)
            
            # 차원 정보 수집
            if vec.shape[0] not in dimension_info:
                dimension_info[vec.shape[0]] = 0
            dimension_info[vec.shape[0]] += 1
            
            temp_vectors.append((i, vec, labels))
            max_dim = max(max_dim, vec.shape[0])
            
        except Exception as e:
            print(f"⚠️ row {i} 오류: {e}")
            bad_idx.append(i)
    
    print(f"📊 차원 분포: {dimension_info}")
    print(f"📏 최대 차원: {max_dim}")
    
    # JSONL의 저장된 features와 실제 계산된 features 비교
    mismatch_count = 0
    for i, row in enumerate(rows_mode):
        if 'features' in row and i < len(temp_vectors):
            stored_features = row['features']
            _, actual_vec, _ = temp_vectors[i]
            if len(stored_features) != actual_vec.shape[0]:
                mismatch_count += 1
                print(f"⚠️ 차원 불일치 발견 row {i}: 저장된={len(stored_features)}, 실제={actual_vec.shape[0]}")
    
    if mismatch_count > 0:
        print(f"🔧 총 {mismatch_count}개 row에서 차원 불일치 감지 - 실제 계산값 사용")
    
    # 2차: 차원 통일하여 최종 데이터 구성
    for i, vec, labels in temp_vectors:
        # 차원 통일 (부족하면 0으로 패딩, 초과하면 자르기)
        if vec.shape[0] < max_dim:
            padding = np.zeros(max_dim - vec.shape[0])
            vec = np.concatenate([vec, padding])
        elif vec.shape[0] > max_dim:
            vec = vec[:max_dim]
        
        X_list.append(vec)
        y_list.append(labels)

    if not X_list:
        print("🚫 유효한 학습 row가 없습니다.")
        return {
            'success': False,
            'reason': 'No valid training rows found'
        }

    print(f"📊 차원 통일 완료: {len(X_list)}개 샘플, 차원={max_dim}")
    X = np.stack(X_list)
    y = np.stack(y_list)
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    # 🧪 스타일 충돌 감지
    has_conflict, clusters, conflict_info = detect_style_conflicts(X, y, desc_list_final)
    
    if has_conflict:
        print("⚠️ 스타일 충돌 감지!")
        print(f"   최대 차이: {conflict_info['max_difference']:.2f}")
        print(f"   충돌 Descriptor: {[d['descriptor'] for d in conflict_info['conflicting_descriptors']]}")
        
        # 적응적 가중치 계산
        sample_weights = calculate_sample_weights(X, y, clusters)
        print(f"   가중치 적용: 최근 스타일 우선 (범위: {np.min(sample_weights):.2f}~{np.max(sample_weights):.2f})")
    else:
        print("✅ 스타일 일관성 확인됨")
        sample_weights = np.ones(len(X))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 가중치도 train/val로 분할
    if len(sample_weights) == len(X):
        train_indices = np.arange(len(X_train))
        val_indices = np.arange(len(X_train), len(X))
        weights_train = sample_weights[:len(X_train)]
        weights_val = sample_weights[len(X_train):]
    else:
        weights_train = np.ones(len(X_train))
        weights_val = np.ones(len(X_val))

    # 가중치를 적용한 DataLoader 생성
    if has_conflict:
        # WeightedRandomSampler 사용
        sampler = WeightedRandomSampler(weights_train, len(weights_train), replacement=True)
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            ), batch_size=TF_BATCH_SIZE, sampler=sampler
        )
    else:
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            ), batch_size=TF_BATCH_SIZE, shuffle=True
        )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ), batch_size=TF_BATCH_SIZE, shuffle=False
    )

    model = FinetuneMLP(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=TF_LR)
    criterion = nn.MSELoss()

    for epoch in range(1, TF_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            val_loss = np.mean([
                criterion(model(xb), yb).item() for xb, yb in val_loader
            ])
            print(f"[Epoch {epoch}] Val Loss = {val_loss:.4f}")

    # Save model and metadata
    save_path = ODOR_FINETUNE_PTH if mode == 'odor' else TASTE_FINETUNE_PTH
    meta_path = MODEL_META_ODOR if mode == 'odor' else MODEL_META_TASTE
    torch.save(model.state_dict(), save_path)
    print(f"✅ Finetuned model saved to {save_path}")

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input_dim': input_dim,
            'output_dim': output_dim,
            'y_cols': desc_list_final,
            'feature_dim': input_dim
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Model meta saved to {meta_path}")

    if bad_idx:
        print(f"⚠️ 학습 제외된 row index: {bad_idx}")
    
    # 훈련 성공 결과 반환
    return {
        'success': True,
        'final_loss': val_loss,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'training_samples': len(X_list),
        'excluded_samples': len(bad_idx),
        'model_path': str(save_path),
        'meta_path': str(meta_path)
    }


if __name__ == '__main__':
    import sys
    # 받을 인자가 있으면 그 모드만, 없으면 odor와 taste 모두 실행
    modes = sys.argv[1:] if len(sys.argv) > 1 else ['odor', 'taste']
    for mode in modes:
        print(f"\n▶ Running train_model for mode: {mode}\n")
        train_model(mode)