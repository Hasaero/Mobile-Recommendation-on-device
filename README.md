# Galaxy Life Planner Demo

This repository contains a lightweight Python demo that showcases how routine mining, a personal knowledge graph, a common knowledge graph, app category signals, and an explanation layer cooperate to produce on-device recommendations.

## Features

- **Process Mining**: builds a Markov-style routine model from timestamped event logs.
- **Personal Knowledge Graph**: compacts user-specific preferences, visit recency, and app usage counts.
- **Common Knowledge Graph**: stores open-vs-closed POI metadata and allows proximity queries.
- **App Category Signals**: infers short-term intent from recent app categories.
- **Zero-shot LLM Scoring**: queries an open-source text generation model to boost unseen POIs; falls back only if the generator errors.
- ****LLM Explanations**: queries an open-source text model to produce one-sentence English summaries that cite the top reasons (falls back to templates if transformers is unavailable).

## Project Layout

```
src/
  glp_demo/
    app_signals.py        # intent inference from app categories
    ckg.py                # minimal common knowledge graph
    data_models.py        # dataclasses shared across layers
    demo.py               # end-to-end demo entrypoint
    demo_data.py          # synthetic logs and POIs
    llm.py                # planner and explanation stub
    pkg.py                # personal knowledge graph logic
    process_mining.py     # routine extraction utilities
    recommendation.py     # multi-signal ranking engine
```

## Running the Demo

```
cd src
python -m glp_demo.demo
```

The script prints the planner tool order, scored recommendation candidates, and natural-language explanations.
> Note: Install `transformers` and `accelerate`, and download an open-source chat model (e.g. TinyLlama-1.1B-Chat) before running the demo to enable LLM-based scoring and explanations.

## Next Steps

- Replace the synthetic logs with device data ingestion APIs.
- Tune the explanation prompt or plug in your preferred instruction-tuned LLM.
- Expand the knowledge graphs with richer schemas and persistence.
- Integrate lightweight bandit logic to balance exploration versus exploitation.

### Zero-shot scoring hook

```
from glp_demo.llm import ZeroShotLLMScorer
from glp_demo.recommendation import RecommendationEngine

engine = RecommendationEngine(
    routine_model,
    personal_graph,
    common_graph,
    zero_shot_scorer=ZeroShotLLMScorer(),
)
```

The stub returns zeros by default so behaviour stays identical until you replace the scorer with a real LLM client.
ExplanationGenerator will also attempt to use transformers for narrative output, but will gracefully fall back to templated messages if the dependency is missing.


