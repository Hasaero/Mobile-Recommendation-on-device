"""Test OpenAI API response format"""

import sys
import os
from datetime import datetime

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Add src directory to Python path
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from glp_demo.openai_scorer import OpenAIZeroShotScorer
from glp_demo.demo_data import create_sample_events, create_sample_pois
from glp_demo.pkg import PersonalKnowledgeGraph
from glp_demo.process_mining import build_routine_model
from glp_demo.app_signals import infer_intent_signals

print("=" * 80)
print("Testing OpenAI API Response")
print("=" * 80)

# Setup data
events = create_sample_events()
pois = create_sample_pois()
routine = build_routine_model(events)
pkg = PersonalKnowledgeGraph()
pkg.ingest_events(events)
pkg.link_routine(routine)

# Get user context
reference_time = datetime(2025, 10, 3, 18, 30)  # Evening workout time
intents = infer_intent_signals(events, reference_time)

print("\n1. User Context:")
print(f"   Time: {reference_time.strftime('%Y-%m-%d %H:%M')}")
print(f"   Time slot: evening")
print(f"   Top intents:")
for intent in intents[:3]:
    print(f"     - {intent.intent}: {intent.score:.3f}")

# Find zero-shot POIs
zero_shot_pois = []
for poi in pois:
    visit_count = pkg.place_visit_counts.get(poi.poi_id, 0)
    if visit_count == 0:
        zero_shot_pois.append(poi)

print(f"\n2. Zero-shot POIs (first 3):")
for poi in zero_shot_pois[:3]:
    print(f"   - {poi.name} ({poi.category})")

# Prepare features
poi_features = []
for poi in zero_shot_pois[:3]:
    poi_features.append({
        "poi_id": poi.poi_id,
        "name": poi.name,
        "category": poi.category,
        "reasoning_tokens": {
            "routine": 0.2,
            "personal": 0.0,
            "intent": 0.7,
            "distance": 0.8
        }
    })

context = {
    "time_slot": "evening",
    "intent_scores": {intent.intent: intent.score for intent in intents[:5]},
    "reference_time": reference_time.isoformat(),
    "user_location": {"latitude": 36.0545, "longitude": 129.3720}
}

print("\n3. Initializing OpenAI Scorer...")
try:
    scorer = OpenAIZeroShotScorer(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=500,
    )
    print("   ✓ OpenAI API initialized")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    print("\n   Make sure OPENAI_API_KEY is set in .env file!")
    sys.exit(1)

print("\n4. Sending request to OpenAI API...")
print("\n" + "=" * 80)
print("PROMPT SENT TO OPENAI:")
print("=" * 80)
prompt = scorer._build_prompt(context, poi_features)
print(prompt)
print("=" * 80)

print("\n5. Calling OpenAI API...")
try:
    scores = scorer.score(context, poi_features)

    print("\n" + "=" * 80)
    print("API RESPONSE (parsed scores):")
    print("=" * 80)

    for feature in poi_features:
        poi_id = feature['poi_id']
        score = scores.get(poi_id, 0.0)
        print(f"\n{feature['name']} ({feature['category']})")
        print(f"  POI ID: {poi_id}")
        print(f"  🤖 LLM Score: {score:.3f}")
        print(f"  Base scores: routine={feature['reasoning_tokens']['routine']:.2f}, "
              f"intent={feature['reasoning_tokens']['intent']:.2f}, "
              f"distance={feature['reasoning_tokens']['distance']:.2f}")

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: OpenAI API is working correctly!")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ TEST FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    print("=" * 80)