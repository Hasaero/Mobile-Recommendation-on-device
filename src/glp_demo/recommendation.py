"""추천 엔진 모듈: 다중 신호를 결합하여 개인화된 POI 추천을 생성하는 모듈

이 모듈은 다음 신호들을 종합하여 추천 점수를 계산합니다:
- 루틴 신호: 시간대별 활동 패턴에 기반한 점수
- 개인 신호: 사용자 개인의 선호도와 방문 이력
- 의도 신호: 최근 앱 사용에서 추출된 의도
- 거리 신호: 사용자 위치로부터의 거리
- LLM 신호: 제로샷 상황에서 LLM이 계산하는 보너스 점수
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, TYPE_CHECKING

from .app_signals import AppIntentSignal
from .ckg import CommonKnowledgeGraph, _haversine_distance
from .data_models import RecommendationCandidate
from .pkg import PersonalKnowledgeGraph
from .process_mining import infer_time_slot

if TYPE_CHECKING:
    from .llm import ZeroShotLLMScorer

# POI 카테고리를 사용자 활동으로 매핑하는 테이블
# 루틴 모델에서 시간대별 활동 확률을 조회할 때 사용
CATEGORY_TO_ACTIVITY = {
    "coffee_shop": "coffee_break",     # 카페 -> 커피 브레이크
    "bakery": "coffee_break",          # 베이커리 -> 커피 브레이크
    "korean_restaurant": "meal",       # 한식당 -> 식사
    "italian_restaurant": "meal",      # 이탈리안 레스토랑 -> 식사
    "gym": "exercise",                 # 헬스장 -> 운동
    "pilates": "exercise",             # 필라테스 -> 운동
    "convenience_store": "errand",     # 편의점 -> 심부름/용무
}

# POI 카테고리를 사용자 의도로 매핑하는 테이블
# 앱 사용 패턴에서 추출된 의도와 POI를 매칭할 때 사용
CATEGORY_TO_INTENT = {
    "coffee_shop": "social",           # 카페 -> 사회적 활동
    "bakery": "meal",                 # 베이커리 -> 식사
    "korean_restaurant": "meal",       # 한식당 -> 식사
    "italian_restaurant": "meal",      # 이탈리안 레스토랑 -> 식사
    "gym": "exercise",                 # 헬스장 -> 운동
    "pilates": "exercise",             # 필라테스 -> 운동
    "convenience_store": "errand",     # 편의점 -> 심부름/용무
}

# 사용자 의도에 따른 관련 POI 카테고리 목록
# 특정 의도가 감지되었을 때 우선적으로 고려할 카테고리들
INTENT_TO_TARGET_CATEGORIES = {
    "meal": ["korean_restaurant", "italian_restaurant", "bakery"],  # 식사 의도
    "commute": [],                                                   # 통근 의도 (POI 매핑 없음)
    "exercise": ["gym", "pilates"],                                 # 운동 의도
    "social": ["coffee_shop", "bakery"],                           # 사회적 활동 의도
    "explore": ["coffee_shop", "korean_restaurant"],               # 탐색 의도
    "inform": [],                                                    # 정보 의도 (POI 매핑 없음)
    "errand": ["convenience_store"],                               # 심부름 의도
}

# 제로샷 LLM 점수의 가중치
# 사용자가 방문한 적 없는 POI에 대해 LLM이 계산한 점수를 최종 점수에 반영하는 비율
ZERO_SHOT_LLM_WEIGHT = 0.5


@dataclass
class RankedRecommendation:
    """
    순위가 매겨진 추천 결과를 담는 데이터 클래스

    추천 후보와 함께 추천 설명 생성에 필요한 컨텍스트 정보를 포함합니다.
    """
    candidate: RecommendationCandidate  # 추천 후보 POI와 점수 정보
    explanation_inputs: dict             # 추천 설명 생성용 입력 데이터


class RecommendationEngine:
    """
    추천 엔진 클래스: 다중 신호를 결합하여 개인화된 POI 추천을 생성

    루틴 모델, 개인 지식 그래프, 공통 지식 그래프, LLM 스코러를
    종합하여 사용쪐의 컨텍스트에 맞는 최적의 장소를 추천합니다.
    """

    def __init__(
        self,
        routine_model,
        knowledge_graph: PersonalKnowledgeGraph,
        common_graph: CommonKnowledgeGraph,
        zero_shot_scorer: ZeroShotLLMScorer | None = None,
    ) -> None:
        """
        추천 엔진을 초기화하는 함수

        Args:
            routine_model: 사용자의 루틴 패턴 모델
            knowledge_graph: 개인 지식 그래프 (선호도 및 방문 이력)
            common_graph: 공통 지식 그래프 (POI 데이터베이스)
            zero_shot_scorer: 제로샷 LLM 스코러 (선택사항)
        """
        self.routine_model = routine_model        # 루틴 모델 참조
        self.pkg = knowledge_graph               # 개인 지식 그래프 참조
        self.ckg = common_graph                  # 공통 지식 그래프 참조
        self.zero_shot_scorer = zero_shot_scorer # LLM 스코러 참조

    def recommend(
        self,
        latitude: float,
        longitude: float,
        reference_time: datetime,
        intents: Sequence[AppIntentSignal],
        radius_km: float = 1.5,
        limit: int = 3,
    ) -> List[RankedRecommendation]:
        """
        사용자의 컨텍스트에 기반하여 개인화된 POI 추천 목록을 생성하는 함수

        다중 신호(루틴, 개인, 의도, 거리, LLM)를 결합하여
        최종 추천 점수를 계산하고 순위를 매깁니다.

        Args:
            latitude: 사용자의 현재 위도
            longitude: 사용자의 현재 경도
            reference_time: 추천 기준 시간 (루틴 분석용)
            intents: 앱 사용에서 추출된 사용자 의도 목록
            radius_km: 검색 반경 (킬로미터, 기본 1.5km)
            limit: 최대 추천 개수 (기본 3개)

        Returns:
            점수 순으로 정렬된 추천 POI 목록
        """
        # 기준 시간에서 시간대 추출
        slot = infer_time_slot(reference_time)

        # 사용자 의도에 따른 우선 카테고리 수집
        categories = self._collect_target_categories(intents)

        # 근처 POI 검색
        nearby = self.ckg.query_nearby(
            latitude,
            longitude,
            radius_km,
            categories if categories else None,
        )

        ranked: List[RankedRecommendation] = []
        if not nearby:
            return ranked

        # 의도 점수를 딕셔너리로 변환 (빠른 조회를 위해)
        intent_scores = {intent.intent: intent.score for intent in intents}

        # 각 신호의 가중치 정의
        weights = {
            "routine": 0.35,   # 루틴 신호 35%
            "pkg": 0.3,        # 개인 선호도 30%
            "intent": 0.25,    # 의도 신호 25%
            "distance": 0.1,   # 거리 신호 10%
        }

        # 제로샷 LLM 스코링을 위한 데이터 준비
        zero_shot_features: list[dict] = []
        zero_shot_index: dict[str, int] = {}

        # 각 근처 POI에 대해 다중 신호 점수 계산
        for poi in nearby:
            # POI 카테고리를 활동으로 매핑 (기본값: "explore")
            activity = CATEGORY_TO_ACTIVITY.get(poi.category, "explore")

            # 1. 루틴 신호: 현재 시간대에 해당 활동을 할 확률
            routine_score = self._routine_alignment(slot, activity)

            # 2. 개인 신호: 사용자의 개인적 선호도 (방문 빈도, 최근성 등)
            pkg_score = self.pkg.preference_score(poi.poi_id, reference_time)

            # 3. 의도 신호: 사용자의 현재 의도와 POI 카테고리의 일치도
            intent_score = self._intent_alignment(poi.category, intent_scores)

            # 4. 거리 신호: 사용자 위치로부터의 근접성 보너스
            distance_score = self._distance_bonus(
                latitude,
                longitude,
                poi.latitude,
                poi.longitude,
                radius_km,
            )

            # 가중 합으로 최종 점수 계산
            score = (
                routine_score * weights["routine"]
                + pkg_score * weights["pkg"]
                + intent_score * weights["intent"]
                + distance_score * weights["distance"]
            )

            # 추천 결과 객체 생성
            recommendation = RankedRecommendation(
                candidate=RecommendationCandidate(
                    poi=poi,
                    score=score,
                    reasoning_tokens={
                        "routine": routine_score,
                        "personal": pkg_score,
                        "intent": intent_score,
                        "distance": distance_score,
                    },
                ),
                explanation_inputs={
                    "time_slot": slot,
                    "activity": activity,
                    "intent_scores": intent_scores,
                },
            )
            ranked.append(recommendation)

            # 제로샷 POI(미방문 장소)에 대한 LLM 스코링 준비
            if (
                self.zero_shot_scorer
                and self.pkg.place_visit_counts.get(poi.poi_id, 0) == 0
            ):
                # LLM 점수 초기화
                recommendation.candidate.reasoning_tokens.setdefault("llm", 0.0)
                # 인덱스 매핑 저장
                zero_shot_index[poi.poi_id] = len(ranked) - 1
                # LLM 입력 데이터 준비
                zero_shot_features.append(
                    {
                        "poi_id": poi.poi_id,
                        "name": poi.name,
                        "category": poi.category,
                        "reasoning_tokens": dict(
                            recommendation.candidate.reasoning_tokens
                        ),
                    }
                )

        # LLM 스코링을 위한 컨텍스트 정보 구성
        context = {
            "time_slot": slot,
            "intent_scores": intent_scores,
            "reference_time": reference_time.isoformat(),
            "user_location": {"latitude": latitude, "longitude": longitude},
        }

        # 제로샷 POI들에 LLM 점수 적용
        self._apply_zero_shot_scores(
            ranked,
            zero_shot_features,
            zero_shot_index,
            context,
        )

        # 최종 점수 내림차순 정렬 후 상위 limit개 반환
        ranked.sort(key=lambda item: item.candidate.score, reverse=True)
        return ranked[:limit]

    def _collect_target_categories(self, intents: Sequence[AppIntentSignal]) -> List[str]:
        """
        사용자 의도에 따른 대상 POI 카테고리 목록을 수집하는 내부 함수

        여러 의도에서 겹치는 카테고리들을 중복 제거하여 반환합니다.

        Args:
            intents: 사용자 의도 신호 목록

        Returns:
            중복 제거된 대상 카테고리 목록
        """
        categories: List[str] = []
        for intent in intents:
            # 각 의도에 매핑된 카테고리들을 수집
            categories.extend(INTENT_TO_TARGET_CATEGORIES.get(intent.intent, []))
        # dict.fromkeys()를 사용하여 순서 보장하며 중복 제거
        return list(dict.fromkeys(categories))

    def _routine_alignment(self, time_slot: str, activity: str) -> float:
        """
        루틴 모델에서 특정 시간대에 특정 활동을 할 확률을 조회하는 내부 함수

        사용자의 과거 루틴 패턴에 기반하여 현재 시간대에
        해당 활동을 할 가능성을 0.0-1.0 범위로 반환합니다.

        Args:
            time_slot: 시간대 (morning, midday, afternoon, evening, late_night, overnight)
            activity: 활동 종류 (coffee_break, meal, exercise, errand, explore)

        Returns:
            루틴 일치도 점수 (0.0-1.0)
        """
        # 해당 시간대의 활동 확률 분포 조회
        distribution = self.routine_model.time_slot_activity_probabilities.get(
            time_slot,
            {},
        )
        if not distribution:
            return 0.0
        # 해당 활동의 확률 반환 (없으면 0.0)
        return distribution.get(activity, 0.0)

    def _intent_alignment(self, category: str, intent_scores: dict) -> float:
        """
        POI 카테고리와 사용자 의도의 일치도를 계산하는 내부 함수

        POI 카테고리가 사용자의 현재 의도와 얼마나 잘 매칭되는지를
        정규화된 점수로 반환합니다.

        Args:
            category: POI 카테골0리
            intent_scores: 사용자 의도별 점수 딕셔너리

        Returns:
            의도 일치도 점수 (0.0-1.0)
        """
        # POI 카테고리에 매핑된 의도 조회
        intent = CATEGORY_TO_INTENT.get(category)
        if not intent:
            # 매핑되지 않는 카테고리는 기본 점수 부여
            return 0.1

        # 사용자 의도 중 최대 점수 찾기 (정규화용)
        max_score = max(intent_scores.values(), default=0.0)
        if max_score == 0:
            return 0.0

        # 해당 의도의 점수를 최대 점수로 정규화
        return intent_scores.get(intent, 0.0) / max_score

    def _distance_bonus(
        self,
        user_lat: float,
        user_lon: float,
        poi_lat: float,
        poi_lon: float,
        radius_km: float,
    ) -> float:
        """
        사용자 위치로부터 POI까지의 거리에 따른 보너스 점수를 계산하는 내부 함수

        가까운 곳일수록 높은 점수를, 맨 바깥지에 있는 곳은
        낮은 점수를 부여하여 0.0-1.0 범위로 정규화합니다.

        Args:
            user_lat: 사용자 위도
            user_lon: 사용자 경도
            poi_lat: POI 위도
            poi_lon: POI 경도
            radius_km: 검색 반경 (정규화 기준)

        Returns:
            거리 보너스 점수 (0.0-1.0, 가까울수록 높음)
        """
        # 하버사인 공식으로 정확한 거리 계산
        distance = _haversine_distance(user_lat, user_lon, poi_lat, poi_lon)

        # 거리를 반경으로 정규화하여 0.0-1.0 범위로 변환
        # 1.0 - (distance / radius) => 가까울수록 1.0에 가까움
        normalized = max(0.0, 1.0 - distance / max(radius_km, 0.1))
        return normalized

    def _apply_zero_shot_scores(
        self,
        ranked: List[RankedRecommendation],
        zero_shot_features: list[dict],
        zero_shot_index: dict[str, int],
        context: dict,
    ) -> None:
        """
        제로샷 POI들에 LLM 점수를 적용하는 내부 함수

        사용자가 방문한 적 없는 POI들에 대해 LLM이 컨텍스트를
        분석하여 계산한 보너스 점수를 최종 점수에 반영합니다.

        Args:
            ranked: 추천 결과 목록 (바로 수정됨)
            zero_shot_features: LLM 입력용 POI 특징 데이터
            zero_shot_index: POI ID에서 추천 목록 인덱스로의 매핑
            context: LLM 판단용 컨텍스트 정보
        """
        # LLM 스코러가 없거나 제로샷 POI가 없으면 스킵
        if not self.zero_shot_scorer or not zero_shot_features:
            return

        # LLM에서 POI별 보너스 점수 계산
        llm_scores = self.zero_shot_scorer.score(context, zero_shot_features)

        # 각 POI의 최종 점수에 LLM 보너스 추가
        for poi_id, bonus in llm_scores.items():
            idx = zero_shot_index.get(poi_id)
            if idx is None:
                continue
            recommendation = ranked[idx]
            # 이유 토큰에 LLM 점수 기록
            recommendation.candidate.reasoning_tokens["llm"] = bonus
            # 최종 점수에 가중치를 적용하여 LLM 점수 반영
            recommendation.candidate.score += ZERO_SHOT_LLM_WEIGHT * bonus


