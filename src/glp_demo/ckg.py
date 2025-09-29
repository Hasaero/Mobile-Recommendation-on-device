"""공통 지식 그래프 모듈: POI 데이터베이스와 지리적 검색 기능을 제공하는 모듈

이 모듈은 다음 기능들을 제공합니다:
- POI(관심지점) 데이터베이스 관리
- 위치 기반 근거리 POI 검색
- 카테고리 일반화를 통한 유연한 검색
- 하버사인 공식을 이용한 정확한 거리 계산
"""
from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt
from typing import Iterable, List, Optional, Sequence

from .data_models import POI

# 지구 반지름 (킬로미터 단위)
EARTH_RADIUS_KM = 6371.0

# 카테고리 일반화 매핑: 세부 카테고리를 상위 카테고리로 그룹화
# 예: "korean_restaurant", "italian_restaurant" -> "restaurant"
CATEGORY_GENERALIZATION = {
    "korean_restaurant": "restaurant",
    "japanese_restaurant": "restaurant",
    "italian_restaurant": "restaurant",
    "bakery": "cafe",
    "coffee_shop": "cafe",
    "gym": "fitness",
    "pilates": "fitness",
    "convenience_store": "retail",
}


class CommonKnowledgeGraph:
    """
    공통 지식 그래프 클래스: POI 데이터베이스와 지리적 검색을 담당

    모든 사용자가 공유하는 POI 정보를 관리하고,
    사용자의 위치 기반으로 근처 POI들을 효율적으로 검색합니다.
    """

    def __init__(self, pois: Iterable[POI]) -> None:
        """
        POI 목록으로 공통 지식 그래프를 초기화하는 함수

        Args:
            pois: 관리할 POI(관심지점) 목록
        """
        # 내부 POI 저장소: 모든 POI 정보를 리스트로 저장
        self._pois: List[POI] = list(pois)

    def query_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        categories: Optional[Sequence[str]] = None,
        include_unopen: bool = False,
    ) -> List[POI]:
        """
        지정된 위치 근처의 POI들을 검색하는 함수

        하버사인 공식을 사용하여 정확한 거리를 계산하고,
        반경, 카테고리, 영업상태 등의 조건으로 필터링합니다.

        Args:
            latitude: 검색 중심점의 위도
            longitude: 검색 중심점의 경도
            radius_km: 검색 반경 (킬로미터)
            categories: 찾고자 하는 POI 카테고리 목록 (None이면 모든 카테고리)
            include_unopen: 영업 종료된 POI도 포함할지 여부

        Returns:
            조건에 맞는 POI 목록
        """
        matches: List[POI] = []

        for poi in self._pois:
            # 영업 상태 확인: 닫힌 곳은 제외 (include_unopen=False인 경우)
            if not include_unopen and not poi.is_open:
                continue

            # 카테고리 필터링: 지정된 카테고리에 맞는지 확인
            if categories and not self._category_matches(poi.category, categories):
                continue

            # 거리 계산: 하버사인 공식으로 정확한 지구 표면 거리 계산
            if _haversine_distance(latitude, longitude, poi.latitude, poi.longitude) <= radius_km:
                matches.append(poi)

        return matches

    def _category_matches(self, category: str, targets: Sequence[str]) -> bool:
        """
        POI의 카테고리가 대상 카테고리 목록과 일치하는지 확인하는 내부 함수

        카테고리 일반화를 통해 유연한 매칭을 제공합니다.
        예: "korean_restaurant"는 "restaurant" 검색에도 매칭됨

        Args:
            category: 확인할 POI의 카테고리
            targets: 대상 카테고리 목록

        Returns:
            일치하면 True, 아니면 False
        """
        # POI 카테고리를 상위 카테고리로 일반화
        normalized_category = CATEGORY_GENERALIZATION.get(category, category)

        # 대상 카테고리들도 모두 일반화
        normalized_targets = {CATEGORY_GENERALIZATION.get(target, target) for target in targets}

        # 정확한 매칭 또는 일반화된 카테고리 매칭 확인
        return category in targets or normalized_category in normalized_targets


def _haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    하버사인 공식을 사용하여 두 지점 간의 거리를 계산하는 함수

    지구를 구체로 가정하고 두 위경도 좌표 간의 최단 거리를
    킬로미터 단위로 정확하게 계산합니다.

    Args:
        lat1: 첫 번째 지점의 위도
        lon1: 첫 번째 지점의 경도
        lat2: 두 번째 지점의 위도
        lon2: 두 번째 지점의 경도

    Returns:
        두 지점 간의 거리 (킬로미터)
    """
    # 위도를 라디안으로 변환
    phi1, phi2 = radians(lat1), radians(lat2)

    # 위도와 경도 차이를 라디안으로 변환
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    # 하버사인 공식의 핵심 계산
    # a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2

    # 중심각 계산: c = 2 × atan2(√a, √(1−a))
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # 최종 거리 = 지구 반지름 × 중심각
    return EARTH_RADIUS_KM * c
