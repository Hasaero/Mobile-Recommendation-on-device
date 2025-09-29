"""개인 지식 그래프 모듈: 사용자 개인의 선호도와 행동 패턴을 학습하고 관리하는 모듈

이 모듈은 사용자의 개인화된 정보를 저장하고 분석합니다:
- 장소별 방문 빈도 및 최근 방문 시간
- 앱 카테고리별 사용 패턴
- 활동과 장소 간의 연관 관계
- 개인 선호도 점수 계산
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, Optional

from .data_models import EventLogEntry, RoutineModel


class PersonalKnowledgeGraph:
    """
    개인 지식 그래프 클래스: 사용자의 개인화된 행동 패턴과 선호도를 관리

    사용자의 과거 활동 이력을 분석하여 개인 선호도를 학습하고,
    이를 바탕으로 추천 시 개인화 점수를 계산합니다.
    """

    def __init__(self) -> None:
        # 장소별 방문 횟수 (장소_ID -> 방문_횟수)
        self.place_visit_counts: Dict[str, int] = defaultdict(int)

        # 장소별 마지막 방문 시간 (장소_ID -> 마지막_방문_시간)
        self.place_last_visit: Dict[str, datetime] = {}


        # 장소별로 수행한 활동들 (장소_ID -> {활동들})
        self.activity_by_location: Dict[str, set[str]] = defaultdict(set)

        # 활동별로 수행한 장소들 (활동_종류 -> {장소들})
        self.location_by_activity: Dict[str, set[str]] = defaultdict(set)

    def ingest_events(self, events: Iterable[EventLogEntry]) -> None:
        """
        사용자 이벤트 로그를 수집하여 개인 지식 그래프를 구축하는 함수

        각 이벤트에서 장소, 활동, 앱 사용 정보를 추출하여
        개인 선호도 분석에 필요한 통계 정보를 업데이트합니다.

        Args:
            events: 분석할 사용자 활동 이벤트 목록
        """
        for event in events:
            location_key = event.location

            # 장소 방문 정보 업데이트
            self.place_visit_counts[location_key] += 1
            self.place_last_visit[location_key] = event.timestamp


            # 장소-활동 연관 관계 업데이트
            self.activity_by_location[location_key].add(event.activity)
            self.location_by_activity[event.activity].add(location_key)

    def link_routine(self, routine: RoutineModel) -> None:
        """
        루틴 모델과 개인 지식 그래프를 연결하는 함수

        루틴 모델의 활동 전이 패턴을 바탕으로
        장소-활동 연관 관계를 보강합니다.

        Args:
            routine: 프로세스 마이닝으로 구축된 루틴 모델
        """
        # 활동 전이 확률에서 연관 관계 추출
        for activity, targets in routine.activity_transition_probabilities.items():
            for target_activity in targets:
                # 대상 활동을 수행하는 장소가 있다면
                if target_activity in self.location_by_activity:
                    # 해당 장소들에 현재 활동도 연결
                    for location in self.location_by_activity[target_activity]:
                        self.activity_by_location[location].add(target_activity)

    def preference_score(self, location_key: str, reference_time: Optional[datetime]) -> float:
        """
        특정 장소에 대한 개인 선호도 점수를 계산하는 함수

        방문 빈도, 최근성, 탐색 보너스를 종합하여
        0.0-1.2 범위의 선호도 점수를 반환합니다.

        Args:
            location_key: 점수를 계산할 장소 ID
            reference_time: 최근성 계산의 기준 시간

        Returns:
            개인 선호도 점수 (높을수록 선호)
        """
        visit_count = self.place_visit_counts.get(location_key, 0)

        # 방문 빈도 점수 계산 (가장 자주 방문한 곳을 1.0으로 정규화)
        if not self.place_visit_counts:
            most_frequent = 0
        else:
            most_frequent = max(self.place_visit_counts.values())
        frequency_component = visit_count / most_frequent if most_frequent else 0.0

        # 최근성 점수 계산 (최근 방문할수록 높은 점수)
        recency_component = 0.0
        if reference_time and location_key in self.place_last_visit:
            delta = reference_time - self.place_last_visit[location_key]
            days = max(delta.total_seconds() / 86400.0, 0.0)
            # 지수적 감소: 1일 후 0.5, 2일 후 0.33, ...
            recency_component = 1.0 / (1.0 + days)

        # 탐색 보너스: 미방문 장소에 대한 발견 인센티브
        exploration_bonus = 0.2 if visit_count == 0 else 0.0

        # 가중합: 빈도 60% + 최근성 40% + 탐색 보너스
        return 0.6 * frequency_component + 0.4 * recency_component + exploration_bonus

