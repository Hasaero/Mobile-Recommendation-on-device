"""
데이터 모델 정의 모듈: GLP 시스템에서 사용되는 핵심 데이터 구조들

이 모듈은 다음 데이터 클래스들을 정의합니다:
- EventLogEntry: 사용자 활동 이벤트 로그
- RoutineModel: 루틴 패턴 확률 모델
- POI: 관심지점(Point of Interest) 정보
- RecommendationCandidate: 추천 후보와 점수
- AppIntentSignal: 앱 사용 의도 신호
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass(frozen=True)
class EventLogEntry:
    """
    사용자 활동 이벤트를 나타내는 데이터 클래스

    사용자의 일상 활동을 시간, 활동, 장소, 앱 카테고리로 기록합니다.
    frozen=True로 설정하여 불변 객체로 만들어 데이터 무결성을 보장합니다.
    """
    timestamp: datetime  # 활동이 발생한 시간
    activity: str        # 활동 종류 (예: wake_up, coffee_break, exercise)
    location: str        # 활동이 발생한 장소 ID
    app_category: str    # 관련 앱 카테고리 (예: alarm, maps, fitness)


@dataclass
class RoutineModel:
    """
    사용자의 루틴 패턴을 담은 확률 모델

    프로세스 마이닝을 통해 추출된 두 가지 확률 분포를 저장합니다:
    1. 시간대별 활동 확률: 특정 시간대에 특정 활동을 할 확률
    2. 활동 전이 확률: 특정 활동 후 다른 활동을 할 확률
    """
    # 시간대 -> 활동 -> 확률 (예: {"morning": {"coffee_break": 0.8}})
    time_slot_activity_probabilities: Dict[str, Dict[str, float]]

    # 현재활동 -> 다음활동 -> 확률 (예: {"wake_up": {"coffee_break": 0.9}})
    activity_transition_probabilities: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class POI:
    """
    관심지점(Point of Interest) 정보를 담는 데이터 클래스

    지도 상의 특정 장소에 대한 기본 정보와 메타데이터를 저장합니다.
    frozen=True로 설정하여 POI 정보의 변경을 방지합니다.
    """
    poi_id: str      # 고유 식별자
    name: str        # 장소 이름 (예: "Bukbu Fitness")
    category: str    # 장소 카테고리 (예: "gym", "coffee_shop")
    latitude: float  # 위도 좌표
    longitude: float # 경도 좌표
    is_open: bool    # 현재 영업 여부


@dataclass
class RecommendationCandidate:
    """
    추천 후보 장소와 그 점수 정보를 담는 데이터 클래스

    추천 엔진에서 생성된 각 후보의 최종 점수와
    점수 계산에 사용된 각 신호별 기여도를 저장합니다.
    """
    poi: POI                           # 추천 대상 POI
    score: float                       # 최종 추천 점수 (0.0 ~ 1.0+)
    reasoning_tokens: Dict[str, float] # 신호별 점수 (routine, personal, intent, distance, llm)


@dataclass
class AppIntentSignal:
    """
    앱 사용 패턴에서 추출된 사용자 의도 신호

    최근 앱 사용 이력을 분석하여 사용자의 현재 의도를
    의도 종류와 강도로 나타냅니다.
    """
    intent: str  # 의도 종류 (예: exercise, meal, social, commute)
    score: float # 의도 강도 (높을수록 강한 의도)
