"""앱 신호 분석 모듈: 사용자의 최근 앱 사용 패턴에서 의도 신호를 추출하는 모듈

이 모듈은 다음 기능들을 제공합니다:
- 앱 카테고리별 사용자 의도 매핑
- 시간 창 기반 최근 앱 사용 분석
- 최근성 가중치를 적용한 의도 점수 계산
- 의도별 점수 순으로 정렬된 신호 목록 생성
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, List

from .data_models import AppIntentSignal, EventLogEntry

# 앱 카테고리에서 사용자 의도로의 매핑 테이블
# 각 앱 카테고리 사용 시 추론할 수 있는 사용자 의도와 그 강도
APP_CATEGORY_TO_INTENTS = {
    "maps": {"commute": 1.0, "explore": 0.4},           # 지도: 이동 100%, 탐색 40%
    "messenger": {"social": 1.0},                       # 메신저: 사회적 활동 100%
    "fitness": {"exercise": 1.0, "hydrate": 0.3},      # 피트니스: 운동 100%, 수분섭취 30%
    "food_delivery": {"meal": 1.0},                    # 음식 배달: 식사 100%
    "calendar": {"planning": 1.0},                     # 캘린더: 계획 수립 100%
    "news": {"inform": 1.0},                           # 뉴스: 정보 습등 100%
    "finance": {"errand": 0.8},                        # 금융: 심부름/경제 활동 80%
}


def infer_intent_signals(
    events: Iterable[EventLogEntry],
    reference_time: datetime,
    window_minutes: int = 120,
) -> List[AppIntentSignal]:
    """
    사용자의 최근 앱 사용 이력에서 의도 신호를 추출하는 함수

    지정된 시간 창 내의 앱 사용 이벤트들을 분석하여
    사용자의 현재 의도를 추론하고 점수화합니다.
    최근 이벤트일수록 더 높은 가중치를 부여합니다.

    Args:
        events: 분석할 사용자 이벤트 목록
        reference_time: 의도 추출의 기준 시간
        window_minutes: 분석 대상 시간 창 (기본 120분 = 2시간)

    Returns:
        점수 내림차순으로 정렬된 의도 신호 목록
    """
    # 시간 창 시작점 계산 (기준 시간에서 window_minutes 전)
    window_start = reference_time - timedelta(minutes=window_minutes)

    # 의도별 누적 점수를 저장할 딕셔너리
    intent_scores = defaultdict(float)

    # 시간 창 내의 모든 이벤트 분석
    for event in events:
        # 시간 창 이전 이벤트는 무시
        if event.timestamp < window_start:
            continue

        # 앱 카테고리에 매핑된 의도 가중치 조회
        weights = APP_CATEGORY_TO_INTENTS.get(event.app_category)
        if not weights:
            continue  # 매핑되지 않은 카테고리는 스킵

        # 최근성 계산: 기준 시간에서 얼마나 오래되었는지
        delta_minutes = (reference_time - event.timestamp).total_seconds() / 60.0

        # 최근성 가중치: 최근 이벤트일수록 높은 가중치 (0.2 ~ 1.0)
        recency_factor = max(0.2, 1.0 - delta_minutes / window_minutes)

        # 각 의도에 대해 가중치 * 최근성 점수 누적
        for intent, weight in weights.items():
            intent_scores[intent] += weight * recency_factor

    # 의도 신호 객체 생성 및 점수 내림차순 정렬
    signals = [AppIntentSignal(intent=intent, score=score) for intent, score in intent_scores.items()]
    return sorted(signals, key=lambda signal: signal.score, reverse=True)
