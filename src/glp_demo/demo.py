from __future__ import annotations

from datetime import datetime

from .app_signals import infer_intent_signals
from .ckg import CommonKnowledgeGraph
from .demo_data import create_sample_events, create_sample_pois
from .llm import ExplanationGenerator
from .pkg import PersonalKnowledgeGraph
from .process_mining import build_routine_model
from .recommendation import RecommendationEngine


def main() -> None:
    events = create_sample_events()
    routine = build_routine_model(events)

    pkg = PersonalKnowledgeGraph()
    pkg.ingest_events(events)
    pkg.link_routine(routine)

    ckg = CommonKnowledgeGraph(create_sample_pois())

    reference_time = datetime(2025, 9, 26, 18, 30)
    intents = infer_intent_signals(events, reference_time)

    engine = RecommendationEngine(routine, pkg, ckg)

    user_latitude = 36.0539
    user_longitude = 129.3745

    ranked = engine.recommend(
        latitude=user_latitude,
        longitude=user_longitude,
        reference_time=reference_time,
        intents=intents,
        radius_km=3.0,
        limit=3,
    )

    generator = ExplanationGenerator()

    print("--- Recommendations ---")
    for item in ranked:
        poi = item.candidate.poi
        message = generator.build_message(item)
        print(f"{poi.name}: score={item.candidate.score:.3f}")
        print(f"  reasoning={item.candidate.reasoning_tokens}")
        print(f"  message={message}\n")


if __name__ == "__main__":
    main()
