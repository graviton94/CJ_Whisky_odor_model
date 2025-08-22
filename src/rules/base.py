"""규칙 엔진의 기본 인터페이스와 타입을 정의하는 모듈"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime

RuleType = TypeVar('RuleType')
DataType = TypeVar('DataType')

@dataclass
class RuleContext:
    """규칙 실행 컨텍스트"""
    descriptor_type: str
    parameters: Dict[str, Any]
    timestamp: datetime = datetime.now()

@dataclass
class RuleResult:
    """규칙 실행 결과"""
    applied: bool
    modified_data: Any
    message: str = ""

class Rule(ABC, Generic[RuleType, DataType]):
    """규칙 기본 클래스"""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
        
    @abstractmethod
    def apply(self, data: DataType, context: RuleContext) -> RuleResult:
        """규칙 적용
        
        Args:
            data: 규칙을 적용할 데이터
            context: 규칙 실행 컨텍스트
            
        Returns:
            규칙 적용 결과
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """규칙 유효성 검사"""
        pass

class RuleSet(Generic[RuleType]):
    """규칙 집합"""
    
    def __init__(self, name: str):
        self.name = name
        self.rules: Dict[str, Rule] = {}
        
    def add_rule(self, rule: Rule) -> None:
        """규칙 추가"""
        if rule.validate():
            self.rules[rule.name] = rule
        else:
            raise ValueError(f"Invalid rule: {rule.name}")
    
    def get_rules(self) -> List[Rule]:
        """우선순위 순으로 정렬된 규칙 목록 반환"""
        return sorted(self.rules.values(), key=lambda x: x.priority, reverse=True)
