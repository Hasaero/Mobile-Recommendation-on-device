"""LLM 통합 모듈: 추천 설명 생성과 제로샷 POI 스코링을 위한 LLM 기능들

이 모듈은 다음 기능들을 제공합니다:
- 오픈소스 LLM을 통한 자연어 추천 설명 생성
- 사용자가 방문한 적 없는 POI에 대한 제로샷 점수 계산
- 컨텍스트 기반 POI 관련성 평가
- Hugging Face transformers 파이프라인 지원
- 강건한 오류 처리 및 폴백 메시지 생성
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .recommendation import RankedRecommendation

# 제로샷 LLM 스코링용 기본 프롬프트
# 사용자 컨텍스트와 POI 후보들을 바탕으로 0-1 범위의 관련성 점수를 JSON으로 반환
DEFAULT_PROMPT = (
    "You are an on-device recommendation model. "
    "Given the user context and the provided POI candidates, "
    "return a JSON object that maps each poi_id to a relevance score between 0 and 1. "
    "Do not include any fields that were not provided.\n"
    "Context:\n{context}\nCandidates:\n{candidates}\nJSON:"
)
# 추천 설명 생성용 프롬프트
# 이유 토큰을 바탕으로 40단어 이내의 친근한 영어 설명 생성
EXPLANATION_PROMPT = (
    "You are a recommendation assistant. Respond in English with one friendly sentence explaining why the POI suits the user. "
    "Mention the place name and cite up to two reasons based on the reasoning tokens. Stay under 40 words.\n"
    "Context:\n{context}\nReasoning tokens:\n{reasoning}\nResponse:"
)

# 영어 설명용 이유 라벨 매핑
# 각 신호 유형에 대한 사용자 친화적 설명 텍스트
EN_REASON_LABELS = {
    "routine": "fits your usual schedule",         # 루틴 신호
    "personal": "matches your recent visits",      # 개인 선호도 신호
    "intent": "aligns with your current app signals",  # 의도 신호
    "distance": "is close to your location",      # 거리 신호
}

# 기본 영어 설명 (다른 이유가 없을 때)
DEFAULT_EN_REASON = "good chance to explore somewhere new"

# 영어 설명 템플릿
EN_EXPLANATION_TEMPLATE = "I suggest {poi_name}. Reason: {reasons}."

# 로거 설정
logger = logging.getLogger(__name__)



class ExplanationGenerator:
    """
    추천 설명 생성기 클래스: 오픈소스 LLM을 사용하여 영어 추천 메시지 생성

    이유 토큰과 컨텍스트를 바탕으로 사용자 친화적인 추천 설명을 생성.
    LLM이 사용 불가한 경우 FallBack 메시지로 대체
    """

    def __init__(
        self,
        generator: Any = None,
        *,
        model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens: int = 80,
        temperature: float = 0.3,
        top_p: float = 0.9,
        prompt_template: str | None = None,
        device_map: str | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        """
        추천 설명 생성기를 초기화하는 함수

        Args:
            generator: 사전 설정된 텍스트 생성기 (선택사항)
            model_id: Hugging Face 모델 ID
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 제어 (0.0-1.0)
            top_p: nucleus sampling 매개변수
            prompt_template: 사용자 정의 프롬프트 템플릿
            device_map: GPU/CPU 디바이스 매핑
            **pipeline_kwargs: 추가 파이프라인 매개변수
        """
        # 생성 매개변수 저장
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_template = prompt_template or EXPLANATION_PROMPT

        if generator is not None:
            # 사전 설정된 생성기 사용
            self._generator = generator
        else:
            # transformers 라이브러리를 사용한 생성기 초기화
            try:
                from transformers import pipeline as hf_pipeline  # type: ignore import
            except Exception as exc:  # pragma: no cover - import guard
                logger.warning("transformers is not available: %s", exc)
                self._generator = None
                return

            if device_map is not None:
                pipeline_kwargs.setdefault("device_map", device_map)
            pipeline_kwargs.setdefault("return_full_text", False)
            self._generator = hf_pipeline(
                "text-generation",
                model=model_id,
                **pipeline_kwargs,
            )

    def build_message(self, ranked: RankedRecommendation) -> str:
        """
        순위가 매겨진 추천 결과에 대한 설명 메시지를 생성하는 함수

        LLM을 사용하여 자연스러운 설명을 생성하거나,
        실패 시 폴백 메시지를 반환합니다.

        Args:
            ranked: 순위가 매겨진 추천 결과

        Returns:
            사용자 친화적인 추천 설명 문자열
        """
        # LLM이 사용 불가하면 폴백 메시지 반환
        if self._generator is None:
            return self._fallback_message(ranked)

        # 프롬프트 구성
        prompt = self._build_prompt(ranked)
        try:
            # LLM으로 텍스트 생성
            outputs = self._generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ExplanationGenerator LLM call failed: %s", exc)
            return self._fallback_message(ranked)

        # 첫 번째 문장 추출 및 검증
        message = self._extract_first_sentence(outputs)
        if message:
            return message
        return self._fallback_message(ranked)

    def _build_prompt(self, ranked: RankedRecommendation) -> str:
        """
        추천 결과로부터 LLM용 프롬프트를 구성하는 내부 함수

        컨텍스트와 이유 토큰을 포맷팅하여 LLM이 이해할 수 있는
        형태의 프롬프트를 생성합니다.

        Args:
            ranked: 순위가 매겨진 추천 결과

        Returns:
            LLM 입력용 프롬프트 문자열
        """
        inputs = ranked.explanation_inputs
        candidate = ranked.candidate
        intent_scores = inputs.get("intent_scores", {})

        # 컨텍스트 정보 라인들 구성
        context_lines = [
            f"- poi_name: {candidate.poi.name}",
            f"- poi_category: {candidate.poi.category}",
            f"- time_slot: {inputs.get('time_slot', 'unknown')}",
            f"- activity: {inputs.get('activity', 'unknown')}",
        ]

        # 사용자 의도 정보 추가 (있는 경우)
        if intent_scores:
            formatted_intents = ", ".join(
                f"{intent}:{score:.3f}" for intent, score in intent_scores.items()
            )
            context_lines.append(f"- intents: {formatted_intents}")

        # 이유 토큰 라인들 구성
        reasoning_tokens = ranked.candidate.reasoning_tokens
        reasoning_lines = [
            f"{key}={value:.3f}" if isinstance(value, (int, float)) else f"{key}={value}"
            for key, value in reasoning_tokens.items()
        ]

        # 템플릿에 컨텍스트와 이유 정보 삽입
        return self.prompt_template.format(
            context="\n".join(context_lines),
            reasoning="\n".join(reasoning_lines),
        )

    def _extract_first_sentence(self, outputs: Any) -> str:
        """
        LLM 출력에서 첫 번째 문장을 추출하는 내부 함수

        다양한 형태의 LLM 출력을 처리하고 깨끗한 첫 번째 문장을 반환합니다.

        Args:
            outputs: LLM 출력 (리스트 또는 딕셔너리 형태)

        Returns:
            처리된 첫 번째 문장 (빈 문자열 가능)
        """
        # 출력 형태에 따라 텍스트 추출
        if isinstance(outputs, list) and outputs:
            candidate = outputs[0].get("generated_text", "")
        elif isinstance(outputs, dict):  # pragma: no cover - pipeline alt
            candidate = outputs.get("generated_text", "")
        else:
            candidate = ""

        candidate = candidate.strip()
        if not candidate:
            return ""

        # 첫 번째 줄 추출
        first_line = candidate.splitlines()[0].strip()
        if not first_line:
            return ""

        # 마침표 처리: 이미 있으면 그대로, 없으면 추가
        if first_line.endswith((".", "!", "?")):
            return first_line
        return first_line + "."

    def _fallback_message(self, ranked: RankedRecommendation) -> str:
        """
        LLM을 사용할 수 없을 때의 폴백 메시지를 생성하는 내부 함수

        이유 토큰을 기반으로 미리 정의된 템플릿을 사용하여
        사용자 친화적인 설명을 생성합니다.

        Args:
            ranked: 순위가 매겨진 추천 결과

        Returns:
            폴백 설명 메시지
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


class ZeroShotLLMScorer:
    """
    제로샷 LLM 스코러 클래스: 오픈소스 LLM을 통한 제로샷 POI 점수 계산

    사용자가 방문한 적 없는 POI에 대해 컨텍스트를 분석하여
    관련성 점수를 계산합니다. 기본적으로 Hugging Face 파이프라인을 사용합니다.
    """

    def __init__(
        self,
        generator: Any = None,
        *,
        model_id: str = "meta-llama/Llama-3-8b-instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.9,
        prompt_template: str | None = None,
        device_map: str | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        """
        제로샷 LLM 스코러를 초기화하는 함수

        Args:
            generator: 사전 설정된 텍스트 생성기 (선택사항)
            model_id: Hugging Face 모델 ID (기본: Llama-3-8b-instruct)
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 무작위성 (0.0-1.0, 낮을수록 일관성 있음)
            top_p: nucleus sampling 매개변수
            prompt_template: 사용자 정의 프롬프트 템플릿
            device_map: GPU/CPU 디바이스 매핑
            offload_folder: 모델 오프로드 폴더 (미사용)
            **pipeline_kwargs: 추가 파이프라인 매개변수
        """
        # 생성 매개변수 저장
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_template = prompt_template or DEFAULT_PROMPT

        if generator is not None:
            # 사전 설정된 생성기 사용
            self._generator = generator
        else:
            # transformers 라이브러리를 사용한 생성기 초기화
            try:
                from transformers import pipeline as hf_pipeline  # type: ignore import
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "transformers is required to instantiate ZeroShotLLMScorer. Install it with `pip install transformers accelerate`."
                ) from exc

            if device_map is not None:
                pipeline_kwargs.setdefault("device_map", device_map)
            pipeline_kwargs.setdefault("return_full_text", False)
            self._generator = hf_pipeline(
                "text-generation",
                model=model_id,
                **pipeline_kwargs,
            )

    def score(self, context: Dict[str, Any], poi_features: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        컨텍스트와 POI 특징을 바탕으로 제로샷 POI들의 점수를 계산하는 함수

        LLM에게 사용자 컨텍스트와 POI 정보를 제공하여
        각 POI에 대한 0.0-1.0 범위의 관련성 점수를 받습니다.

        Args:
            context: 사용자 컨텍스트 (시간, 위치, 의도 등)
            poi_features: 점수를 매길 POI 특징 목록

        Returns:
            POI ID에서 점수로의 매핑 딕셔너리
        """
        if not poi_features:
            return {}

        # LLM 입력용 프롬프트 구성
        prompt = self._build_prompt(context, poi_features)
        try:
            # LLM으로 JSON 형태의 점수 생성
            outputs = self._generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ZeroShotLLMScorer generation failed: %s", exc)
            # 실패 시 모든 POI에 0.0 점수 부여
            return {features["poi_id"]: 0.0 for features in poi_features}

        # LLM 출력에서 텍스트 추출
        raw_text = outputs[0].get("generated_text", "") if isinstance(outputs, list) and outputs else ""

        # JSON에서 점수 파싱
        parsed_scores = self._extract_scores(raw_text)

        # 각 POI에 대해 점수 반환 (0.0-1.0 범위로 제한)
        return {
            features["poi_id"]: self._clamp(parsed_scores.get(features["poi_id"], 0.0))
            for features in poi_features
        }

    def _build_prompt(self, context: Dict[str, Any], poi_features: List[Dict[str, Any]]) -> str:
        """
        컨텍스트와 POI 특징에서 LLM용 프롬프트를 구성하는 내부 함수

        사용자 컨텍스트와 POI 후보들을 LLM이 이해할 수 있는
        형태로 구조화하여 프롬프트를 생성합니다.

        Args:
            context: 사용자 컨텍스트 정보
            poi_features: POI 특징 목록

        Returns:
            LLM 입력용 프롬프트 문자열
        """
        # 컨텍스트 정보 라인들 구성
        context_lines = [
            f"- time_slot: {context.get('time_slot', 'unknown')}",
            f"- reference_time: {context.get('reference_time', 'unknown')}",
            f"- user_location: {context.get('user_location', {})}",
        ]

        # 사용자 의도 정보 추가 (있는 경우)
        intents = context.get("intent_scores", {})
        if intents:
            formatted = ", ".join(f"{intent}:{score:.3f}" for intent, score in intents.items())
            context_lines.append(f"- intents: {formatted}")

        # POI 후보 라인들 구성
        candidate_lines = []
        for features in poi_features:
            # 이유 토큰 포맷팅
            reasoning = features.get("reasoning_tokens", {})
            reasoning_str = ", ".join(
                f"{key}={value:.3f}" if isinstance(value, (int, float)) else f"{key}={value}"
                for key, value in reasoning.items()
            )
            # POI 정보 라인 구성
            candidate_lines.append(
                f"- {features['poi_id']} | name={features.get('name')} | category={features.get('category')}"
                + (f" | signals={reasoning_str}" if reasoning_str else "")
            )

        # 템플릿에 컨텍스트와 후보 정보 삽입
        return self.prompt_template.format(
            context="\n".join(context_lines),
            candidates="\n".join(candidate_lines),
        )

    def _extract_scores(self, generated_text: str) -> Dict[str, float]:
        """
        LLM이 생성한 텍스트에서 JSON 형태의 점수를 추출하는 내부 함수

        정규표현식으로 JSON 부분을 찾고 파싱하여
        POI ID에서 점수로의 매핑을 반환합니다.

        Args:
            generated_text: LLM이 생성한 원본 텍스트

        Returns:
            POI ID에서 점수로의 매핑 딕셔너리
        """
        # 정규표현식으로 JSON 부분 찾기
        match = re.search(r"\{.*?\}", generated_text, flags=re.DOTALL)
        if not match:
            logger.warning("ZeroShotLLMScorer could not find JSON in response: %s", generated_text)
            return {}

        # JSON 파싱 시도
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("ZeroShotLLMScorer JSON decoding failed: %s", exc)
            return {}

        # 숫자 값들만 추출하여 점수 딕셔너리 구성
        scores: Dict[str, float] = {}
        for key, value in data.items():
            try:
                scores[str(key)] = float(value)
            except (TypeError, ValueError):
                logger.debug("Discarding non-numeric score for %s: %s", key, value)
        return scores

    @staticmethod
    def _clamp(value: float) -> float:
        """
        점수를 0.0-1.0 범위로 제한하는 정적 메서드

        LLM이 범위를 벗어난 점수를 생성했을 때
        안전하게 유효한 범위로 조정합니다.

        Args:
            value: 제한할 점수 값

        Returns:
            0.0-1.0 범위로 제한된 점수
        """
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
