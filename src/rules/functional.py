"""작용기 기반 규칙 구현"""
from typing import Dict, Any, List
from .base import Rule, RuleContext, RuleResult

class FunctionalRule(Rule[Dict[str, Any], Dict[str, float]]):
    """작용기 기반 규칙"""
    
    def __init__(self, name: str, target: str, required_groups: List[str], effect: Dict[str, Any]):
        """
        Args:
            name: 규칙 이름
            target: 대상 설명자
            required_groups: 필요한 작용기 목록
            effect: 적용할 효과 설정
        """
        super().__init__(name)
        self.target = target
        self.required_groups = required_groups
        self.effect = effect
    
    def apply(self, data: Dict[str, float], context: RuleContext) -> RuleResult:
        """규칙 적용
        
        Args:
            data: 설명자별 강도 데이터
            context: 규칙 실행 컨텍스트
            
        Returns:
            규칙 적용 결과
        """
        # 작용기 존재 여부 확인
        if not self._check_functional_groups(context):
            return RuleResult(False, data)
            
        # 효과 적용
        result = data.copy()
        current_value = result.get(self.target, 0.0)
        
        if self.effect['type'] == 'multiply':
            result[self.target] = current_value * self.effect['factor']
        elif self.effect['type'] == 'add':
            result[self.target] = current_value + self.effect['value']
        elif self.effect['type'] == 'set':
            result[self.target] = self.effect['value']
            
        return RuleResult(
            True,
            result,
            f"Applied functional group effect on {self.target}"
        )
    
    def validate(self) -> bool:
        """규칙 유효성 검사"""
        valid_effect_types = {'multiply', 'add', 'set'}
        
        return (
            isinstance(self.target, str) and
            isinstance(self.required_groups, list) and
            len(self.required_groups) > 0 and
            isinstance(self.effect, dict) and
            'type' in self.effect and
            self.effect['type'] in valid_effect_types
        )
    
    def _check_functional_groups(self, context: RuleContext) -> bool:
        """작용기 존재 여부 확인
        
        Args:
            context: 규칙 실행 컨텍스트
            
        Returns:
            모든 필요 작용기 존재 여부
        """
        if 'functional_groups' not in context.parameters:
            return False
            
        groups = context.parameters['functional_groups']
        return all(group in groups for group in self.required_groups)
