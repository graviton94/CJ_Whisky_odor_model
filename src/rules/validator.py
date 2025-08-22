"""규칙 검증을 담당하는 모듈"""
from typing import Dict, Set, List
from collections import defaultdict

class RuleValidator:
    """규칙 유효성 검사기"""
    
    def validate_rules(self, rules: Dict) -> bool:
        """규칙 설정 전체 검증
        
        Args:
            rules: 규칙 설정 데이터
            
        Returns:
            검증 통과 여부
        """
        # 1. 필수 규칙 타입 확인
        if not self._validate_rule_types(rules):
            print(f"Rule validation failed: missing required types. Found types: {set(rules.keys())}")
            return False
            
        # 2. 규칙 간 충돌 검사
        if not self._check_rule_conflicts(rules):
            return False
            
        # 3. 순환 참조 검사
        if not self._check_circular_dependencies(rules):
            return False
            
        return True
    
    def _validate_rule_types(self, rules: Dict) -> bool:
        """필수 규칙 타입 존재 여부 확인
        
        Args:
            rules: 규칙 설정 데이터
            
        Returns:
            검증 통과 여부
        """
        required_types = {'synergy', 'masking', 'functional'}
        return all(rule_type in rules for rule_type in required_types)
    
    def _check_rule_conflicts(self, rules: Dict) -> bool:
        """규칙 간 충돌 검사
        
        Args:
            rules: 규칙 설정 데이터
            
        Returns:
            충돌 없음 여부
        """
        # 각 규칙 타입 내의 충돌 검사
        for rule_type, rule_set in rules.items():
            if not isinstance(rule_set, dict):
                return False
                
            # 동일한 타겟에 대한 충돌 규칙 검사
            targets = defaultdict(list)
            for rule_name, rule_data in rule_set.items():
                if 'target' in rule_data:
                    targets[rule_data['target']].append(rule_name)
                    
            # 동일 타겟에 대해 충돌하는 규칙이 있는지 확인
            for target, rule_names in targets.items():
                if len(rule_names) > 1 and not self._are_rules_compatible(rule_set, rule_names):
                    return False
                    
        return True
    
    def _are_rules_compatible(self, rule_set: Dict, rule_names: List[str]) -> bool:
        """규칙들 간의 호환성 검사
        
        Args:
            rule_set: 규칙 집합
            rule_names: 검사할 규칙 이름 목록
            
        Returns:
            호환성 여부
        """
        for i, name1 in enumerate(rule_names):
            rule1 = rule_set[name1]
            for name2 in rule_names[i+1:]:
                rule2 = rule_set[name2]
                
                # 상충되는 효과 검사
                if 'effect_type' in rule1 and 'effect_type' in rule2:
                    if rule1['effect_type'] == 'set' and rule2['effect_type'] == 'set':
                        # set 타입 효과는 동일 대상에 중복 적용 불가
                        return False
                        
                # 최소/최대값 제약 충돌 검사
                if ('min_value' in rule1 and 'max_value' in rule2 and 
                    rule1['min_value'] > rule2['max_value']):
                    return False
                if ('max_value' in rule1 and 'min_value' in rule2 and 
                    rule1['max_value'] < rule2['min_value']):
                    return False
        
        return True
    
    def _check_circular_dependencies(self, rules: Dict) -> bool:
        """순환 참조 검사
        
        Args:
            rules: 규칙 설정 데이터
            
        Returns:
            순환 참조 없음 여부
        """
        dependencies = defaultdict(set)
        
        # 의존성 그래프 구축
        for rule_type, rule_set in rules.items():
            for rule_name, rule_data in rule_set.items():
                if 'depends_on' in rule_data:
                    dependencies[rule_name].update(rule_data['depends_on'])
        
        # 순환 참조 검사
        visited = set()
        path = []
        
        def has_cycle(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
                
            visited.add(node)
            path.append(node)
            
            for dep in dependencies[node]:
                if has_cycle(dep):
                    return True
                    
            path.pop()
            return False
        
        nodes = list(dependencies.keys())
        return not any(has_cycle(node) for node in nodes)
