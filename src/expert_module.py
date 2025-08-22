import pandas as pd
import datetime
from pathlib import Path

from config import TRIAL_CSV, TRIAL_LONG_CSV


def save_trial_result(trial_id, mode, input_molecules, predict_scores, expert_scores, extra_info=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    record = {'trial_id': trial_id, 'timestamp': timestamp, 'mode': mode}

    for i, m in enumerate(input_molecules, start=1):
        record[f'SMILES_{i}'] = m['SMILES']
        record[f'PeakArea_{i}'] = m['peak_area']
    for k, v in predict_scores.items():
        record[f'predict_{k}'] = v
    for k, v in expert_scores.items():
        record[f'expert_{k}'] = v
    if extra_info:
        record.update(extra_info)

    df_new = pd.DataFrame([record])
    path = Path(TRIAL_CSV)
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"[Expert] Trial 저장 완료: {trial_id} ({mode})")


def save_trial_long_format(trial_id, mode, input_molecules, predict_scores, expert_scores, extra_info=None, path=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    records = []
    for idx, mol in enumerate(input_molecules, start=1):
        for desc, pred in predict_scores.items():
            record = {
                'trial_id': trial_id,
                'timestamp': timestamp,
                'mode': mode,
                'molecule_idx': idx,
                'SMILES': mol['SMILES'],
                'PeakArea': mol['peak_area'],
                'descriptor': desc,
                'predict_score': pred,
                'expert_score': expert_scores.get(desc)
            }
            if extra_info:
                record.update(extra_info)
            records.append(record)
    df_new = pd.DataFrame(records)
    out_path = Path(path or TRIAL_LONG_CSV)
    if out_path.exists():
        df = pd.read_csv(out_path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[Expert] Long format trial 저장 완료: {trial_id} ({mode})")


def load_trials(trial_id=None, mode=None):
    path = Path(TRIAL_CSV)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if trial_id:
        df = df[df['trial_id'] == trial_id]
    if mode:
        df = df[df['mode'] == mode]
    return df


def get_expert_vs_predict(trial_id, mode):
    df = load_trials(trial_id, mode)
    if df.empty:
        return []
    desc_cols = [c for c in df.columns if c.startswith('predict_')]
    results = []
    for _, row in df.iterrows():
        entry = {'trial_id': row['trial_id'], 'mode': row['mode']}
        for col in desc_cols:
            desc = col.replace('predict_', '')
            entry[f'predict_{desc}'] = row[col]
            expert_val = row.get(f'expert_{desc}')
            entry[f'expert_{desc}'] = expert_val
            if pd.notnull(expert_val):
                entry[f'gap_{desc}'] = expert_val - row[col]
        results.append(entry)
    return results
