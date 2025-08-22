"""규칙 처리를 담당하는 엔진 모듈"""
from typing import Dict, List, Any, Optional
from collections import OrderedDict
import time
from pathlib import Path
import json

from .base import Rule, RuleSet, RuleContext, RuleResult
from .cache import CacheManager
from .validator import RuleValidator
from .synergy import SynergyRule
from .masking import MaskingRule
from .functional import FunctionalRule

class RuleEngine:
    """규칙 엔진 메인 클래스"""
    
    def __init__(self):
        self.rule_sets: Dict[str, RuleSet] = {
            'synergy': RuleSet('synergy'),
            'masking': RuleSet('masking'),
            'functional': RuleSet('functional')
        }
        self.cache = CacheManager()
        self.validator = RuleValidator()
        
    def load_rules(self, path: Path) -> None:
        """JSON 파일에서 규칙 로드
        
        Args:
            path: 규칙 JSON 파일 경로
        """
        # 캐시 확인
        cached_rules = self.cache.get(str(path))
        if cached_rules:
            # 캐시된 데이터로 규칙 재구성
            self._load_from_cache(cached_rules)
            return
            
        # JSON 파일 로드
        with open(path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
            
        # 규칙 유효성 검사
        if not self.validator.validate_rules(rules_data):
            raise ValueError("Invalid rules configuration")
            
        # 규칙 간 의존성 맵 생성
        self.dependencies = {}
        for set_name, rules in rules_data.items():
            if set_name in self.rule_sets:
                for rule_name, rule_config in rules.items():
                    self.dependencies[rule_name] = rule_config.get('depends_on', [])
                self._load_rule_set(set_name, rules)
                
        # 캐시에 저장 (원본 데이터 저장)
        self.cache.set(str(path), rules_data)
    
    def _load_from_cache(self, cached_rules: Dict) -> None:
        """캐시된 데이터에서 규칙 로드
        
        Args:
            cached_rules: 캐시된 규칙 데이터
        """
        # 규칙 집합 초기화
        for rule_set in self.rule_sets.values():
            rule_set.rules.clear()
            
        # 의존성 맵 재생성
        self.dependencies = {}
        for set_name, rules in cached_rules.items():
            if set_name in self.rule_sets:
                for rule_name, rule_config in rules.items():
                    self.dependencies[rule_name] = rule_config.get('depends_on', [])
                self._load_rule_set(set_name, rules)
    
    def _load_rule_set(self, set_name: str, rules: Dict) -> None:
        """규칙 집합에 규칙 로드
        
        Args:
            set_name: 규칙 집합 이름
            rules: 규칙 데이터
        """
        rule_set = self.rule_sets[set_name]
        for rule_name, rule_data in rules.items():
            rule = self._create_rule(set_name, rule_name, rule_data)
            if rule:
                rule_set.add_rule(rule)
    
    def _create_rule(self, set_name: str, rule_name: str, rule_data: Dict) -> Optional[Rule]:
        """규칙 객체 생성
        
        Args:
            set_name: 규칙 집합 이름
            rule_name: 규칙 이름
            rule_data: 규칙 데이터
            
        Returns:
            생성된 규칙 객체
        """
        try:
            if set_name == 'synergy':
                return SynergyRule(
                    name=rule_name,
                    target=rule_data['target'],
                    factor=rule_data['factor'],
                    conditions=rule_data.get('conditions', {})
                )
            elif set_name == 'masking':
                return MaskingRule(
                    name=rule_name,
                    target=rule_data['target'],
                    masker=rule_data['masker'],
                    threshold=rule_data['threshold'],
                    reduction=rule_data['reduction']
                )
            elif set_name == 'functional':
                return FunctionalRule(
                    name=rule_name,
                    target=rule_data['target'],
                    required_groups=rule_data['required_groups'],
                    effect=rule_data['effect']
                )
            return None
        except KeyError as e:
            raise ValueError(f"Missing required field in rule {rule_name}: {str(e)}")
    
    def apply_rules(
        self,
        data: Dict[str, float],
        context: RuleContext
    ) -> Dict[str, Any]:
        """규칙 적용
        
        Args:
            data: 입력 데이터
            context: 규칙 실행 컨텍스트
            
        Returns:
            규칙이 적용된 결과
        """
        result = data.copy()
        rule_log = []
        
        # 규칙 실행 순서 결정
        executed = set()
        all_rules = []
        
        # 모든 규칙 수집 및 이름으로 정렬
        for set_name in ['synergy', 'masking', 'functional']:
            rule_set = self.rule_sets[set_name]
            rules = sorted(rule_set.get_rules(), key=lambda r: r.name)
            all_rules.extend(rules)
                
        def can_execute(rule: Rule) -> bool:
            if rule.name in executed:
                return False
            return all(dep in executed for dep in self.dependencies.get(rule.name, []))
        
        # 의존성을 고려하여 규칙 실행
        while len(executed) < len(all_rules):
            executed_this_round = False
            
            for rule in all_rules:
                if not can_execute(rule):
                    continue
                
                rule_result = rule.apply(result, context)
                if rule_result.applied:
                    result = rule_result.modified_data
                    rule_log.append({
                        'rule': rule.name,
                        'message': rule_result.message
                    })
                    executed.add(rule.name)
                    executed_this_round = True
            
            if not executed_this_round:
                # 더 이상 실행할 수 있는 규칙이 없음
                break
        
        return {
            'result': result,
            'rule_log': rule_log
        }
