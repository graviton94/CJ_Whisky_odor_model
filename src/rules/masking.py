"""마스킹 규칙 구현"""
from typing import Dict, Any
from .base import Rule, RuleContext, RuleResult

class MaskingRule(Rule[Dict[str, Any], Dict[str, float]]):
    """마스킹 효과 규칙"""
    
    def __init__(self, name: str, target: str, masker: str, threshold: float, reduction: float):
        """
        Args:
            name: 규칙 이름
            target: 마스킹될 설명자
            masker: 마스킹하는 설명자
            threshold: 마스킹 발생 임계값
            reduction: 감소 계수
        """
        super().__init__(name)
        self.target = target
        self.masker = masker
        self.threshold = threshold
        self.reduction = reduction
    
    def apply(self, data: Dict[str, float], context: RuleContext) -> RuleResult:
        """규칙 적용
        
        Args:
            data: 설명자별 강도 데이터
            context: 규칙 실행 컨텍스트
            
        Returns:
            규칙 적용 결과
        """
        # target이나 masker가 데이터에 없으면 규칙 적용 안 함
        if self.target not in data or self.masker not in data:
            return RuleResult(False, data)
            
        # 마스킹 조건 확인
        masker_value = data[self.masker]
        if masker_value < self.threshold:
            return RuleResult(False, data)
            
        # 마스킹 효과 적용
        result = data.copy()
        target_value = result[self.target]
        result[self.target] = target_value * (1 - self.reduction)
        
        return RuleResult(
            True,
            result,
            f"Applied masking effect on {self.target} by {self.masker}"
        )
    
    def validate(self) -> bool:
        """규칙 유효성 검사"""
        return (
            isinstance(self.target, str) and
            isinstance(self.masker, str) and
            isinstance(self.threshold, (int, float)) and
            isinstance(self.reduction, (int, float)) and
            0 <= self.reduction <= 1 and
            self.threshold > 0
        )
