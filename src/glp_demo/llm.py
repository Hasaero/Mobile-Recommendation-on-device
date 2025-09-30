"""LLM 통합 모듈: 추천 설명 생성 (템플릿 기반)

이 모듈은 템플릿 기반 추천 설명 생성 기능을 제공합니다.
Zero-shot 점수 계산은 openai_scorer.py를 사용하세요.
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .recommendation import RankedRecommendation

# 영어 설명용 이유 라벨 매핑
EN_REASON_LABELS = {
    "routine": "fits your usual schedule",
    "personal": "matches your recent visits",
    "intent": "aligns with your current app signals",
    "distance": "is close to your location",
}

# 기본 영어 설명
DEFAULT_EN_REASON = "good chance to explore somewhere new"

# 영어 설명 템플릿
EN_EXPLANATION_TEMPLATE = "I suggest {poi_name}. Reason: {reasons}."


class ExplanationGenerator:
    """
    추천 설명 생성기 클래스 (템플릿 기반)

    이유 토큰을 기반으로 사용자 친화적인 추천 설명을 생성합니다.
    """

    def __init__(self, generator=None, **kwargs) -> None:
        """
        추천 설명 생성기 초기화

        Args:
            generator: 사용하지 않음 (하위 호환성 유지용)
            **kwargs: 사용하지 않음 (하위 호환성 유지용)
        """
        # 템플릿 기반으로만 동작 (LLM 사용 안 함)
        pass

    def build_message(self, ranked: RankedRecommendation) -> str:
        """
        순위가 매겨진 추천 결과에 대한 설명 메시지 생성

        Args:
            ranked: 순위가 매겨진 추천 결과

        Returns:
            사용자 친화적인 추천 설명 문자열
        """
        poi = ranked.candidate.poi
        scores = ranked.candidate.reasoning_tokens
        picked: List[str] = []

        # 유의미한 점수(0.2 이상)를 가진 이유들 선별
        for key in ("routine", "personal", "intent", "distance"):
            if scores.get(key, 0.0) > 0.2 and key in EN_REASON_LABELS:
                picked.append(EN_REASON_LABELS[key])

        # 선별된 이유가 없으면 기본 이유 사용
        if not picked:
            picked.append(DEFAULT_EN_REASON)

        # 이유들을 세미콜론으로 연결하여 설명 문장 생성
        reason_sentence = "; ".join(picked)
        return EN_EXPLANATION_TEMPLATE.format(poi_name=poi.name, reasons=reason_sentence)