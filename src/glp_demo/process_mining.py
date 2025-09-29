"""
프로세스 마이닝 모듈: 사용자의 이벤트 로그에서 일상 루틴 패턴을 추출하는 모듈

이 모듈은 타임스탬프가 있는 사용자 활동 로그를 분석하여 다음을 생성합니다:
1. 시간대별 활동 확률 분포 (예: 아침에 커피 마실 확률)
2. 활동 간 전이 확률 분포 (예: 기상 후 커피 마실 확률)

Markov 체인 기반의 루틴 모델을 구축하여 추천 시스템의 루틴 점수 계산에 활용됩니다.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List

from .data_models import EventLogEntry, RoutineModel

# 하루를 6개 시간대로 구분하는 정의
# 각 튜플: (시작시간, 종료시간, 시간대_이름)
TIME_SLOT_DEFINITIONS = (
    (5, 10, "morning"),     # 아침 5시-10시
    (10, 14, "midday"),     # 오전 10시-오후 2시
    (14, 18, "afternoon"),  # 오후 2시-6시
    (18, 22, "evening"),    # 저녁 6시-10시
    (22, 24, "late_night"), # 심야 10시-자정
    (0, 5, "overnight"),    # 새벽 자정-5시
)


def infer_time_slot(timestamp: datetime) -> str:
    """
    주어진 시간을 시간대로 분류하는 함수

    Args:
        timestamp: 분류할 시간 정보

    Returns:
        해당하는 시간대 문자열 (morning, midday, afternoon, evening, late_night, overnight)
    """
    hour = timestamp.hour
    for start, end, label in TIME_SLOT_DEFINITIONS:
        # 일반적인 경우: 시작시간 <= 종료시간 (예: 5시-10시)
        if start <= end and start <= hour < end:
            return label
        # 자정을 넘나드는 경우: 시작시간 > 종료시간 (예: 22시-5시)
        if start > end and (hour >= start or hour < end):
            return label
    return "unknown"


def build_routine_model(events: Iterable[EventLogEntry]) -> RoutineModel:
    """
    사용자의 이벤트 로그에서 루틴 모델을 구축하는 함수

    이 함수는 두 가지 확률 분포를 계산합니다:
    1. 시간대별 활동 확률: 특정 시간대에 특정 활동을 할 확률
    2. 활동 전이 확률: 특정 활동 후 다른 활동을 할 확률

    Args:
        events: 시간순으로 정렬된 사용자 활동 이벤트 목록

    Returns:
        시간대별 활동 확률과 활동 전이 확률을 포함하는 루틴 모델
    """
    # 시간 순으로 이벤트를 정렬
    sorted_events: List[EventLogEntry] = sorted(events, key=lambda item: item.timestamp)

    # 시간대별 활동 횟수를 저장할 중첩 딕셔너리
    time_slot_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # 활동 전이 횟수를 저장할 중첩 딕셔너리 (활동A -> 활동B 횟수)
    transition_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # 이전 활동과 날짜를 추적하기 위한 변수들
    previous_activity = None
    previous_day = None

    for event in sorted_events:
        # 현재 이벤트의 시간대 분류
        slot = infer_time_slot(event.timestamp)
        # 시간대별 활동 횟수 증가
        time_slot_counts[slot][event.activity] += 1.0

        # 활동 전이 패턴 분석 (같은 날 내에서만)
        day_key = event.timestamp.date()
        if previous_activity and previous_day == day_key:
            # 이전 활동에서 현재 활동으로의 전이 횟수 증가
            transition_counts[previous_activity][event.activity] += 1.0

        # 다음 이벤트를 위해 현재 정보를 저장
        previous_activity = event.activity
        previous_day = day_key

    # 횟수를 확률로 정규화
    time_slot_probabilities = {
        slot: _normalize_distribution(activity_counts)
        for slot, activity_counts in time_slot_counts.items()
    }
    transition_probabilities = {
        activity: _normalize_distribution(targets)
        for activity, targets in transition_counts.items()
    }

    return RoutineModel(
        time_slot_activity_probabilities=time_slot_probabilities,
        activity_transition_probabilities=transition_probabilities,
    )


def _normalize_distribution(counts: Dict[str, float]) -> Dict[str, float]:
    """
    횟수 딕셔너리를 확률 분포로 정규화하는 내부 함수

    각 값을 전체 합으로 나누어 확률 분포를 만듭니다.
    예: {A: 3, B: 1} -> {A: 0.75, B: 0.25}

    Args:
        counts: 항목별 횟수가 담긴 딕셔너리

    Returns:
        항목별 확률이 담긴 딕셔너리 (합이 1.0이 됨)
    """
    total = float(sum(counts.values()))
    # 총 횟수가 0이면 빈 딕셔너리 반환
    if total == 0.0:
        return {}
    # 각 횟수를 총 횟수로 나누어 확률 계산
    return {label: value / total for label, value in counts.items()}
