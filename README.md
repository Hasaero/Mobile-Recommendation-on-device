# Mobile Rec system


This repository contains a lightweight Python demo that showcases how routine mining, a personal knowledge graph, a common knowledge graph, app category signals, and an explanation layer cooperate to produce on-device recommendations.

## Features

- **Process Mining**: builds a Markov-style routine model from timestamped event logs.
- **Personal Knowledge Graph**: compacts user-specific preferences, visit recency, and app usage counts.
- **Common Knowledge Graph**: stores open-vs-closed POI metadata and allows proximity queries.
- **App Category Signals**: infers short-term intent from recent app categories.
- **OpenAI-based Zero-shot Scoring**: uses OpenAI GPT API to score unseen POIs based on user context and POI features.
- **Template-based Explanations**: generates natural-language recommendation explanations using simple templates based on reasoning tokens.

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
> Note: Set `OPENAI_API_KEY` environment variable to enable OpenAI-based zero-shot scoring. Install the `openai` package with `pip install openai`.

## Next Steps

- Replace the synthetic logs with device data ingestion APIs.
- Tune the OpenAI scoring prompt or switch to a different model.
- Expand the knowledge graphs with richer schemas and persistence.
- Integrate lightweight bandit logic to balance exploration versus exploitation.

### OpenAI Zero-shot Scoring

```python
from glp_demo.llm import OpenAIZeroShotScorer
from glp_demo.recommendation import RecommendationEngine

# Initialize with OpenAI API key
scorer = OpenAIZeroShotScorer(
    api_key="your-api-key",  # or set OPENAI_API_KEY env var
    model="gpt-4",
    temperature=0.3,
)

engine = RecommendationEngine(
    routine_model,
    personal_graph,
    common_graph,
    zero_shot_scorer=scorer,
)
```

The scorer uses OpenAI GPT models to compute relevance scores for POI candidates based on user context. If the API call fails, it returns default scores (0.5) for all candidates.


