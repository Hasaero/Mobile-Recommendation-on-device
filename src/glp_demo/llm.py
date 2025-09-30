"""LLM 통합 모듈: 추천 설명 생성 및 OpenAI 기반 제로샷 스코링"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .recommendation import RankedRecommendation

logger = logging.getLogger(__name__)

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

# OpenAI API를 사용한 제로샷 스코링용 프롬프트
OPENAI_PROMPT = """You are an on-device recommendation model assistant.
Given the user context and POI candidates, return a JSON object that maps each poi_id to a relevance score between 0 and 1.

Context:
{context}

Candidates:
{candidates}

Return ONLY a valid JSON object with poi_id as keys and scores (0.0-1.0) as values. Do not include any explanation or additional text.

Example format:
{{"poi_id_1": 0.85, "poi_id_2": 0.65}}

JSON:"""


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


class OpenAIZeroShotScorer:
    """
    OpenAI API를 사용한 제로샷 POI 스코러

    로컬 모델 대신 OpenAI GPT API를 사용하여 더 정확한 점수 계산
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> None:
        """
        OpenAI Zero-Shot Scorer 초기화

        Args:
            api_key: OpenAI API 키 (None이면 환경변수 OPENAI_API_KEY 사용)
            model: 사용할 OpenAI 모델 (gpt-3.5-turbo, gpt-4, etc.)
            temperature: 생성 온도 (낮을수록 일관적)
            max_tokens: 최대 토큰 수
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # OpenAI 클라이언트 초기화
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required. Install it with `pip install openai`."
            ) from exc

    def score(self, context: Dict[str, Any], poi_features: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        컨텍스트와 POI 특징을 바탕으로 제로샷 POI들의 점수를 계산

        Args:
            context: 사용자 컨텍스트 (시간, 위치, 의도 등)
            poi_features: 점수를 매길 POI 특징 목록

        Returns:
            POI ID에서 점수로의 매핑 딕셔너리
        """
        if not poi_features:
            return {}

        # 프롬프트 구성
        prompt = self._build_prompt(context, poi_features)

        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # 응답에서 텍스트 추출
            raw_text = response.choices[0].message.content.strip()
            logger.info(f"OpenAI response: {raw_text}")

            # JSON 파싱
            parsed_scores = self._extract_scores(raw_text)

            # 각 POI에 대해 점수 반환 (0.0-1.0 범위로 제한)
            return {
                features["poi_id"]: self._clamp(parsed_scores.get(features["poi_id"], 0.0))
                for features in poi_features
            }

        except Exception as exc:
            logger.error(f"OpenAI API call failed: {exc}")
            # 실패 시 모든 POI에 0.5 기본값 부여
            return {features["poi_id"]: 0.5 for features in poi_features}

    def _build_prompt(self, context: Dict[str, Any], poi_features: List[Dict[str, Any]]) -> str:
        """프롬프트 구성"""
        # 컨텍스트 정보
        context_lines = [
            f"- time_slot: {context.get('time_slot', 'unknown')}",
            f"- reference_time: {context.get('reference_time', 'unknown')}",
            f"- user_location: {context.get('user_location', {})}",
        ]

        intents = context.get("intent_scores", {})
        if intents:
            formatted = ", ".join(f"{intent}:{score:.3f}" for intent, score in intents.items())
            context_lines.append(f"- intents: {formatted}")

        # POI 후보 정보
        candidate_lines = []
        for features in poi_features:
            reasoning = features.get("reasoning_tokens", {})
            reasoning_str = ", ".join(
                f"{key}={value:.3f}" if isinstance(value, (int, float)) else f"{key}={value}"
                for key, value in reasoning.items()
            )
            candidate_lines.append(
                f"- {features['poi_id']} | name={features.get('name')} | category={features.get('category')}"
                + (f" | signals={reasoning_str}" if reasoning_str else "")
            )

        return OPENAI_PROMPT.format(
            context="\n".join(context_lines),
            candidates="\n".join(candidate_lines),
        )

    def _extract_scores(self, generated_text: str) -> Dict[str, float]:
        """생성된 텍스트에서 JSON 점수 추출"""
        # JSON 블록 찾기
        import re

        # 코드 블록 제거 (```json ... ```)
        text = re.sub(r'```json\s*', '', generated_text)
        text = re.sub(r'```\s*', '', text)

        # JSON 파싱 시도
        try:
            # 전체 텍스트를 JSON으로 파싱
            data = json.loads(text)

            # 숫자 값들만 추출
            scores: Dict[str, float] = {}
            for key, value in data.items():
                try:
                    scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    logger.debug(f"Discarding non-numeric score for {key}: {value}")

            return scores

        except json.JSONDecodeError:
            # JSON 파싱 실패 시 정규표현식으로 재시도
            match = re.search(r'\{[^}]+\}', text)
            if match:
                try:
                    data = json.loads(match.group(0))
                    scores = {}
                    for key, value in data.items():
                        try:
                            scores[str(key)] = float(value)
                        except (TypeError, ValueError):
                            pass
                    return scores
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Could not parse JSON from: {text}")
            return {}

    @staticmethod
    def _clamp(value: float) -> float:
        """점수를 0.0-1.0 범위로 제한"""
        return max(0.0, min(1.0, value))