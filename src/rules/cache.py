"""규칙 캐시를 관리하는 모듈"""
import time
from typing import Dict, Any, Optional
from collections import OrderedDict

class CacheManager:
    """JSON 규칙 파일의 캐시를 관리하는 클래스
    
    이 클래스는 GUI 애플리케이션에서 규칙 파일의 로드 성능을 개선하기 위해 사용됩니다.
    프로그램 시작 시 로드된 규칙을 메모리에 캐시하여 반복적인 파일 I/O와 JSON 파싱을 방지합니다.
    """
    
    def __init__(self, max_size: int = 100, expiration_time: float = 3600):
        """캐시 매니저 초기화
        
        Args:
            max_size: 캐시에 저장할 최대 항목 수 (기본값: 100)
            expiration_time: 캐시 항목의 만료 시간(초) (기본값: 3600초 = 1시간)
        """
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.expiration_time = expiration_time
    
    def get(self, key: str) -> Optional[Any]:
        """캐시된 값을 조회
        
        Args:
            key: 캐시 키 (보통 규칙 파일의 경로)
            
        Returns:
            캐시된 값이 있고 만료되지 않았으면 그 값을, 없거나 만료되었으면 None
        """
        if key not in self.cache:
            return None
            
        cached_item = self.cache[key]
        current_time = time.time()
        
        # 만료 시간 체크
        if current_time - cached_item['timestamp'] > self.expiration_time:
            del self.cache[key]
            return None
            
        # LRU 업데이트: 최근 사용된 항목을 끝으로 이동
        self.cache.move_to_end(key)
        return cached_item['data']
    
    def set(self, key: str, value: Any) -> None:
        """값을 캐시에 저장
        
        Args:
            key: 캐시 키 (보통 규칙 파일의 경로)
            value: 캐시할 값 (파싱된 규칙 데이터)
        """
        current_time = time.time()
        
        # 기존 항목이 있으면 업데이트
        if key in self.cache:
            self.cache[key] = {'data': value, 'timestamp': current_time}
            self.cache.move_to_end(key)
        else:
            # 새 항목 추가
            self.cache[key] = {'data': value, 'timestamp': current_time}
            
            # 최대 크기 초과 시 오래된 항목 제거
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # FIFO: 첫 번째(가장 오래된) 항목 제거
        
    def clear(self) -> None:
        """캐시 전체 초기화
        
        프로그램 종료 시 또는 규칙을 완전히 리로드해야 할 때 사용
        """
        self.cache.clear()
