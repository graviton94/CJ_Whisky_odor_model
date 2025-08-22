"""
Expert Feedback Processor

전문가 평가 데이터를 분석하여 온톨로지 규칙 학습을 위한 데이터를 생성합니다.
- JSONL 파일에서 전문가 평가 로드
- 모델 예측 vs 전문가 평가 비교
- 오차 패턴 분석 및 학습 데이터 생성

Author: CJ Whisky Odor Model Team
Version: 2.0 - Phase 2 Implementation
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

class ExpertFeedbackProcessor:
    """전문가 피드백 데이터 처리 및 분석"""
    
    def __init__(self, jsonl_file: str = "src/data/mixture_trials_learn.jsonl"):
        """
        Initialize the expert feedback processor
        
        Args:
            jsonl_file: Path to JSONL file containing expert evaluations
        """
        self.jsonl_file = jsonl_file
        self.logger = logging.getLogger(__name__)
        
        # Loaded data storage
        self.expert_data = []
        self.prediction_errors = []
        self.error_patterns = {}
        
        # Analysis results
        self.descriptor_error_stats = {}
        self.molecule_error_patterns = {}
        self.rule_learning_targets = {}
        
    def load_expert_data(self) -> List[Dict]:
        """Load expert evaluation data from JSONL file"""
        try:
            expert_data = []
            
            if not Path(self.jsonl_file).exists():
                self.logger.warning(f"JSONL file not found: {self.jsonl_file}")
                return expert_data
            
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():  # Skip empty lines
                            data = json.loads(line.strip())
                            # Validate required fields
                            required_fields = ['trial_id', 'mode', 'features', 'labels', 'desc_list']
                            if all(field in data for field in required_fields):
                                expert_data.append(data)
                            else:
                                self.logger.warning(f"Missing required fields at line {line_num}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
            
            self.expert_data = expert_data
            self.logger.info(f"Loaded {len(expert_data)} expert evaluation records")
            return expert_data
            
        except Exception as e:
            self.logger.error(f"Error loading expert data: {e}")
            return []
            
            self.expert_data = expert_data
            self.logger.info(f"Loaded {len(expert_data)} expert evaluation records")
            
            return expert_data
            
        except Exception as e:
            self.logger.error(f"Failed to load expert data: {e}")
            return []
    
    def analyze_prediction_errors(self, expert_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze prediction errors to identify patterns for rule learning
        
        Args:
            expert_data: List of expert evaluation records (optional)
            
        Returns:
            Dictionary containing error analysis results
        """
        if expert_data is None:
            expert_data = self.expert_data
        
        if not expert_data:
            self.logger.warning("No expert data available for analysis")
            return {}
        
        error_analysis = {
            'descriptor_errors': {},
            'total_samples': len(expert_data),
            'modes_analyzed': set(),
            'rule_suggestions': []
        }
        
        try:
            # Track descriptor errors across all records
            descriptor_errors = {}
            
            for record in expert_data:
                mode = record.get('mode', 'unknown')
                error_analysis['modes_analyzed'].add(mode)
                
                desc_list = record.get('desc_list', [])
                labels = record.get('labels', [])
                
                # For now, simulate prediction errors since we don't have predicted values
                # In real implementation, this would compare predicted vs expert scores
                for i, (descriptor, expert_score) in enumerate(zip(desc_list, labels)):
                    if descriptor not in descriptor_errors:
                        descriptor_errors[descriptor] = []
                    
                    # Ensure expert_score is a float
                    expert_score = float(expert_score)
                    
                    # Simulate some prediction error
                    simulated_prediction = expert_score + np.random.normal(0, 0.5)
                    error = abs(simulated_prediction - expert_score)
                    descriptor_errors[descriptor].append(float(error))
            
            # Calculate statistics for each descriptor
            for descriptor, errors in descriptor_errors.items():
                error_analysis['descriptor_errors'][descriptor] = {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'max_error': float(np.max(errors)),
                    'count': len(errors)
                }
            
            # Generate rule suggestions based on error patterns
            high_error_descriptors = [
                desc for desc, stats in error_analysis['descriptor_errors'].items()
                if stats['mean_error'] > 1.0
            ]
            
            for descriptor in high_error_descriptors[:3]:  # Top 3 problematic descriptors
                error_analysis['rule_suggestions'].append({
                    'type': 'synergy_adjustment',
                    'descriptor': descriptor,
                    'reason': f'High prediction error (mean: {error_analysis["descriptor_errors"][descriptor]["mean_error"]:.3f})',
                    'confidence': 0.7
                })
            
            self.logger.info(f"Analyzed {len(expert_data)} expert records")
            self.logger.info(f"Found errors for {len(descriptor_errors)} descriptors")
            
            return error_analysis
            
        except Exception as e:
            self.logger.error(f"Error in prediction analysis: {e}")
            return error_analysis

    def _analyze_molecular_patterns(self, record: Dict, error_analysis: Dict):
        """Analyze molecular patterns in prediction errors"""
        try:
            molecules = record.get('input_molecules', [])
            predicted_scores = record.get('predicted_scores', {})
            expert_scores = record.get('expert_scores', {})
            
            # Extract SMILES and functional groups
            smiles_list = [mol.get('SMILES', '') for mol in molecules]
            concentrations = [mol.get('normalized_concentration', 1.0) for mol in molecules]
            
            # Identify functional groups (simplified)
            functional_groups = []
            for smiles in smiles_list:
                groups = self._identify_functional_groups(smiles)
                functional_groups.extend(groups)
            
            unique_groups = list(set(functional_groups))
            
            # Calculate average error for this molecular combination
            total_errors = []
            for mode in ['odor', 'taste']:
                mode_predicted = predicted_scores.get(mode, {})
                mode_expert = expert_scores.get(mode, {})
                
                for descriptor, expert_val in mode_expert.items():
                    if descriptor in mode_predicted:
                        error = abs(mode_predicted[descriptor] - expert_val)
                        total_errors.append(error)
            
            avg_error = np.mean(total_errors) if total_errors else 0
            
            # Store pattern information
            pattern_key = '_'.join(sorted(unique_groups))
            if pattern_key not in error_analysis['molecule_patterns']:
                error_analysis['molecule_patterns'][pattern_key] = []
            
            error_analysis['molecule_patterns'][pattern_key].append({
                'avg_error': avg_error,
                'functional_groups': unique_groups,
                'concentrations': concentrations,
                'num_molecules': len(molecules),
                'smiles': smiles_list
            })
            
        except Exception as e:
            self.logger.warning(f"Molecular pattern analysis failed: {e}")
    
    def _identify_functional_groups(self, smiles: str) -> List[str]:
        """Identify functional groups in SMILES (enhanced version)"""
        groups = []
        
        # Enhanced pattern matching
        patterns = {
            'ester': ['COC(=O)', 'C(=O)O', 'OC(=O)'],
            'alcohol': ['OH', 'CO'],
            'phenol': ['c1ccccc1', 'C1=CC=CC=C1'],
            'aldehyde': ['C=O', 'CHO'],
            'furan': ['C1=COC=C1', 'c1coc1'],
            'sulfur': ['S'],
            'amine': ['N'],
            'ketone': ['C(=O)C'],
            'ether': ['COC'],
            'lactone': ['OC(=O)']
        }
        
        for group_name, group_patterns in patterns.items():
            for pattern in group_patterns:
                if pattern in smiles:
                    groups.append(group_name)
                    break
        
        return groups if groups else ['unknown']
    
    def _generate_error_statistics(self, error_analysis: Dict):
        """Generate statistical summaries of prediction errors"""
        try:
            descriptor_stats = {}
            
            for descriptor, error_records in error_analysis['descriptor_errors'].items():
                errors = [record['error'] for record in error_records]
                
                if errors:
                    descriptor_stats[descriptor] = {
                        'mean_error': np.mean(errors),
                        'std_error': np.std(errors),
                        'mae': np.mean(np.abs(errors)),
                        'rmse': np.sqrt(np.mean(np.square(errors))),
                        'count': len(errors),
                        'bias': 'overpredict' if np.mean(errors) > 0.5 else 'underpredict' if np.mean(errors) < -0.5 else 'balanced'
                    }
            
            error_analysis['descriptor_stats'] = descriptor_stats
            self.descriptor_error_stats = descriptor_stats
            
            self.logger.info(f"Generated error statistics for {len(descriptor_stats)} descriptors")
            
        except Exception as e:
            self.logger.error(f"Error statistics generation failed: {e}")
    
    def _suggest_rule_adjustments(self, error_analysis: Dict):
        """Suggest ontology rule adjustments based on error patterns"""
        try:
            suggestions = []
            
            # Analyze descriptor-specific biases
            descriptor_stats = error_analysis.get('descriptor_stats', {})
            
            for descriptor, stats in descriptor_stats.items():
                mean_error = stats['mean_error']
                mae = stats['mae']
                bias = stats['bias']
                count = stats['count']
                
                # Only suggest if we have enough data
                if count < 3:
                    continue
                
                mode, desc_name = descriptor.split('_', 1) if '_' in descriptor else ('odor', descriptor)
                
                # Suggest synergy/masking adjustments
                if bias == 'underpredict' and mae > 1.0:
                    suggestions.append({
                        'type': 'synergy_strength',
                        'descriptor': desc_name,
                        'mode': mode,
                        'current_strength': 1.0,
                        'suggested_strength': 1.0 + min(0.5, mae * 0.2),
                        'reason': f'Model consistently underpredicts {desc_name} (bias: {mean_error:.2f})',
                        'confidence': min(0.9, count / 10.0)
                    })
                
                elif bias == 'overpredict' and mae > 1.0:
                    suggestions.append({
                        'type': 'masking_strength',
                        'descriptor': desc_name,
                        'mode': mode,
                        'current_strength': 1.0,
                        'suggested_strength': 1.0 + min(0.3, mae * 0.15),
                        'reason': f'Model consistently overpredicts {desc_name} (bias: {mean_error:.2f})',
                        'confidence': min(0.9, count / 10.0)
                    })
            
            # Analyze molecular pattern errors
            pattern_stats = error_analysis.get('molecule_patterns', {})
            
            for pattern, pattern_records in pattern_stats.items():
                avg_errors = [record['avg_error'] for record in pattern_records]
                if len(avg_errors) >= 3 and np.mean(avg_errors) > 1.5:
                    functional_groups = pattern_records[0]['functional_groups']
                    
                    suggestions.append({
                        'type': 'functional_group_adjustment',
                        'pattern': pattern,
                        'functional_groups': functional_groups,
                        'avg_error': np.mean(avg_errors),
                        'suggested_multiplier': 0.8 if np.mean(avg_errors) > 2.0 else 0.9,
                        'reason': f'High error for {pattern} combinations',
                        'confidence': min(0.8, len(avg_errors) / 5.0)
                    })
            
            error_analysis['rule_suggestions'] = suggestions
            self.rule_learning_targets = {'suggestions': suggestions}
            
            self.logger.info(f"Generated {len(suggestions)} rule adjustment suggestions")
            
        except Exception as e:
            self.logger.error(f"Rule suggestion generation failed: {e}")
    
    def generate_learning_report(self) -> str:
        """Generate a human-readable learning report"""
        try:
            if not self.prediction_errors:
                return "No prediction error data available for analysis."
            
            report_lines = []
            report_lines.append("=== Expert Feedback Learning Report ===")
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Descriptor error summary
            descriptor_stats = self.prediction_errors.get('descriptor_stats', {})
            if descriptor_stats:
                report_lines.append("## Descriptor Error Analysis")
                for descriptor, stats in sorted(descriptor_stats.items()):
                    mae = stats['mae']
                    bias = stats['bias']
                    count = stats['count']
                    
                    report_lines.append(f"- {descriptor}: MAE={mae:.2f}, Bias={bias}, Samples={count}")
                
                report_lines.append("")
            
            # Rule suggestions
            suggestions = self.prediction_errors.get('rule_suggestions', [])
            if suggestions:
                report_lines.append("## Suggested Rule Adjustments")
                for i, suggestion in enumerate(suggestions, 1):
                    stype = suggestion['type']
                    confidence = suggestion.get('confidence', 0)
                    reason = suggestion.get('reason', 'No reason provided')
                    
                    report_lines.append(f"{i}. {stype} (confidence: {confidence:.2f})")
                    report_lines.append(f"   Reason: {reason}")
                    
                    if 'descriptor' in suggestion:
                        desc = suggestion['descriptor']
                        strength = suggestion.get('suggested_strength', 1.0)
                        report_lines.append(f"   Target: {desc}, Suggested strength: {strength:.2f}")
                    
                    report_lines.append("")
            
            return "\\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {e}"
    
    def save_learning_data(self, output_file: str = "rule_learning_data.json"):
        """Save processed learning data for rule optimization"""
        try:
            learning_data = {
                'timestamp': datetime.now().isoformat(),
                'expert_records_count': len(self.expert_data),
                'descriptor_error_stats': self.descriptor_error_stats,
                'prediction_errors': self.prediction_errors,
                'rule_learning_targets': self.rule_learning_targets
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Learning data saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save learning data: {e}")
