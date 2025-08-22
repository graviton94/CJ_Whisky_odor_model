"""시너지 규칙 구현"""
from typing import Dict, Any
from .base import Rule, RuleContext, RuleResult

class SynergyRule(Rule[Dict[str, Any], Dict[str, float]]):
    """시너지 효과 규칙"""
    
    def __init__(self, name: str, target: str, factor: float, conditions: Dict[str, Any]):
        """
        Args:
            name: 규칙 이름
            target: 대상 설명자
            factor: 시너지 계수
            conditions: 적용 조건
        """
        super().__init__(name)
        self.target = target
        self.factor = factor
        self.conditions = conditions
    
    def apply(self, data: Dict[str, float], context: RuleContext) -> RuleResult:
        """규칙 적용
        
        Args:
            data: 설명자별 강도 데이터
            context: 규칙 실행 컨텍스트
            
        Returns:
            규칙 적용 결과
        """
        # 조건 확인
        if not self._check_conditions(data, context):
            return RuleResult(False, data)
            
        # 시너지 효과 적용
        result = data.copy()
        current_value = result.get(self.target, 0.0)
        result[self.target] = current_value * self.factor
        
        return RuleResult(
            True,
            result,
            f"Applied synergy effect on {self.target} with factor {self.factor}"
        )
    
    def validate(self) -> bool:
        """규칙 유효성 검사"""
        return (
            isinstance(self.target, str) and
            isinstance(self.factor, (int, float)) and
            self.factor > 0
        )
    
    def _check_conditions(self, data: Dict[str, float], context: RuleContext) -> bool:
        """적용 조건 확인
        
        Args:
            data: 설명자별 강도 데이터
            context: 규칙 실행 컨텍스트
            
        Returns:
            조건 만족 여부
        """
        if 'min_values' in self.conditions:
            for desc, min_val in self.conditions['min_values'].items():
                if data.get(desc, 0.0) < min_val:
                    return False
                    
        if 'required_parameters' in self.conditions:
            for param, value in self.conditions['required_parameters'].items():
                if context.parameters.get(param) != value:
                    return False
                    
        return True
