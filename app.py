import streamlit as st
import sys
import os
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

# Load environment variables from .env file
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

from glp_demo.demo_data import create_sample_events, create_sample_pois
from glp_demo.pkg import PersonalKnowledgeGraph
from glp_demo.ckg import CommonKnowledgeGraph
from glp_demo.process_mining import build_routine_model
from glp_demo.app_signals import infer_intent_signals
from glp_demo.recommendation import RecommendationEngine
from glp_demo.llm import ExplanationGenerator
from glp_demo.openai_scorer import OpenAIZeroShotScorer

st.set_page_config(page_title="Mobile POI Recommendation", page_icon="📱", layout="wide")

# Initialize session state
if 'engine' not in st.session_state:
    events = create_sample_events()
    routine = build_routine_model(events)
    pkg = PersonalKnowledgeGraph()
    pkg.ingest_events(events)
    pkg.link_routine(routine)
    ckg = CommonKnowledgeGraph(create_sample_pois())

    # Initialize models
    # Explanation: Template-based (no LLM)
    # Zero-shot scoring: OpenAI API

    # Template-based explanation (fast and reliable)
    explanation_gen = ExplanationGenerator(generator=None)

    # OpenAI API-based zero-shot scorer
    try:
        zero_shot_scorer = OpenAIZeroShotScorer(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
        )
        st.session_state.llm_available = True
    except Exception as e:
        st.warning(f"⚠️ OpenAI API initialization failed: {str(e)}. Set OPENAI_API_KEY environment variable.")
        zero_shot_scorer = None
        st.session_state.llm_available = False

    st.session_state.events = events
    st.session_state.routine = routine
    st.session_state.pkg = pkg
    st.session_state.ckg = ckg
    st.session_state.engine = RecommendationEngine(routine, pkg, ckg, zero_shot_scorer=zero_shot_scorer)
    st.session_state.explanation_gen = explanation_gen

# Header
st.title("📱 Mobile POI Recommendation System")
st.markdown("**On-device personalized location recommendation based on user behavior**")

# Show API status
if st.session_state.get('llm_available', False):
    st.success("🤖 **Zero-Shot Scoring**: OpenAI GPT-3.5-Turbo API active")
else:
    st.info("⚠️ **Zero-Shot Scoring**: Disabled (Set OPENAI_API_KEY environment variable)")

st.info("💬 **Explanation Mode**: Template-based (no LLM)")

# Sidebar - User Context Input
st.sidebar.header("🎯 User Context")

# Scenario presets
st.sidebar.subheader("📋 Scenario Presets")
scenario = st.sidebar.selectbox(
    "Select a scenario:",
    [
        "Custom",
        "Scenario 1: Regular routine (visited places)",
        "Scenario 2: New discovery (zero-shot POIs)",
        "Scenario 3: Unfamiliar city (Seoul)"
    ]
)

# Set default values based on scenario
if scenario == "Scenario 1: Regular routine (visited places)":
    default_lat = 36.0615
    default_lon = 129.3795
    default_hour = 7
    default_minute = 40
    default_radius = 1.0
    default_date = datetime(2025, 10, 1)  # Weekday within training period
    st.sidebar.info("📍 Morning coffee at familiar Bukbu Beach Cafe area")
elif scenario == "Scenario 2: New discovery (zero-shot POIs)":
    default_lat = 36.0545
    default_lon = 129.3720
    default_hour = 18
    default_minute = 30
    default_radius = 2.0
    default_date = datetime(2025, 10, 3)  # Weekday within training period
    st.sidebar.info("🆕 Evening workout - exploring new fitness options")
elif scenario == "Scenario 3: Unfamiliar city (Seoul)":
    default_lat = 37.4979
    default_lon = 127.0276
    default_hour = 12
    default_minute = 0
    default_radius = 3.0
    default_date = datetime(2025, 10, 15)  # Beyond training data (cold start)
    st.sidebar.info("🏙️ Visiting Gangnam area in Seoul - no local history")
else:
    default_lat = 36.0539
    default_lon = 129.3745
    default_hour = 18
    default_minute = 30
    default_radius = 3.0
    default_date = datetime(2025, 9, 26)

# Time selection
reference_date = st.sidebar.date_input(
    "Date",
    default_date
)
reference_hour = st.sidebar.slider("Hour", 0, 23, default_hour)
reference_minute = st.sidebar.slider("Minute", 0, 59, default_minute)
reference_time = datetime.combine(reference_date, datetime.min.time()) + timedelta(hours=reference_hour, minutes=reference_minute)

st.sidebar.markdown(f"**Selected Time:** {reference_time.strftime('%Y-%m-%d %H:%M')}")

# Location selection
st.sidebar.subheader("📍 Current Location")
latitude = st.sidebar.number_input("Latitude", value=default_lat, format="%.4f", step=0.0001)
longitude = st.sidebar.number_input("Longitude", value=default_lon, format="%.4f", step=0.0001)

# Search radius
radius_km = st.sidebar.slider("Search Radius (km)", 0.5, 5.0, default_radius, 0.5)

# Number of recommendations
limit = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Scenario explanation banner
if scenario != "Custom":
    st.markdown("---")
    if scenario == "Scenario 1: Regular routine (visited places)":
        st.info("""
        **📍 Scenario 1: Regular Routine**
        - **Context**: Morning coffee time (7:40 AM) near Bukbu Beach Cafe
        - **Expected**: Recommendations favor familiar places user has visited before
        - **Strength**: High personal preference scores from visit history
        """)
    elif scenario == "Scenario 2: New discovery (zero-shot POIs)":
        st.warning("""
        **🆕 Scenario 2: Zero-Shot Recommendation**
        - **Context**: Evening workout time (6:30 PM) in Pohang
        - **Expected**: System suggests new fitness options user hasn't tried
        - **Strength**: LLM analyzes context (time, intent) to recommend relevant new places
        """)
    elif scenario == "Scenario 3: Unfamiliar city (Seoul)":
        st.error("""
        **🏙️ Scenario 3: Unfamiliar City**
        - **Context**: Lunch time (12:00 PM) in Gangnam, Seoul
        - **Expected**: All recommendations are zero-shot (no local visit history)
        - **Strength**: System relies on intent signals and category matching despite no personal history
        """)
    st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔍 Recommendations")

    # Get recommendations
    intents = infer_intent_signals(st.session_state.events, reference_time)
    recommendations = st.session_state.engine.recommend(
        latitude=latitude,
        longitude=longitude,
        reference_time=reference_time,
        intents=intents,
        radius_km=radius_km,
        limit=limit,
    )

    # Display current intents
    with st.expander("📊 Current User Intents", expanded=False):
        for intent in intents[:5]:
            normalized_score = min(1.0, max(0.0, intent.score))
            st.progress(normalized_score, text=f"{intent.intent}: {intent.score:.3f}")

    # Display recommendations
    if recommendations:
        for i, item in enumerate(recommendations, 1):
            poi = item.candidate.poi
            visit_count = st.session_state.pkg.place_visit_counts.get(poi.poi_id, 0)
            is_zero_shot = visit_count == 0

            with st.container():
                st.markdown(f"### {i}. {poi.name}")

                # Badge for zero-shot or visited
                if is_zero_shot:
                    st.markdown("🆕 **New Place** (Never visited)")
                else:
                    st.markdown(f"📍 **Visited {visit_count} times**")

                # Score display
                col_score1, col_score2 = st.columns([1, 2])
                with col_score1:
                    st.metric("Final Score", f"{item.candidate.score:.3f}")
                with col_score2:
                    st.caption(f"Category: `{poi.category}`")
                    st.caption(f"Location: ({poi.latitude:.4f}, {poi.longitude:.4f})")

                # LLM Explanation
                explanation = st.session_state.explanation_gen.build_message(item)
                st.info(f"💬 {explanation}")

                # Display LLM score prominently for zero-shot POIs
                reasoning = item.candidate.reasoning_tokens
                if is_zero_shot and 'llm' in reasoning:
                    st.success(f"🤖 **LLM Zero-Shot Score**: {reasoning['llm']:.3f} (context-based relevance)")

                # Score breakdown
                with st.expander("📈 Score Breakdown"):
                    for signal, score in reasoning.items():
                        if signal == 'llm':
                            st.write(f"🤖 **LLM Score**: {score:.3f} (zero-shot bonus)")
                        elif signal == 'routine':
                            st.write(f"⏰ **Routine**: {score:.3f} (time-activity alignment)")
                        elif signal == 'personal':
                            st.write(f"👤 **Personal**: {score:.3f} (visit history)")
                        elif signal == 'intent':
                            st.write(f"🎯 **Intent**: {score:.3f} (app usage signals)")
                        elif signal == 'distance':
                            st.write(f"📍 **Distance**: {score:.3f} (proximity bonus)")
                        else:
                            st.write(f"**{signal.capitalize()}**: {score:.3f}")

                st.divider()
    else:
        st.info("No recommendations found in this area. Try adjusting the search radius.")

with col2:
    st.header("🗺️ Map View")

    # Create map
    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # Add user location marker
    folium.Marker(
        [latitude, longitude],
        popup="Your Location",
        tooltip="You are here",
        icon=folium.Icon(color="red", icon="user", prefix="fa")
    ).add_to(m)

    # Add search radius circle
    folium.Circle(
        [latitude, longitude],
        radius=radius_km * 1000,
        color="blue",
        fill=True,
        fillOpacity=0.1,
        popup=f"{radius_km}km radius"
    ).add_to(m)

    # Add POI markers
    if recommendations:
        for i, item in enumerate(recommendations, 1):
            poi = item.candidate.poi
            visit_count = st.session_state.pkg.place_visit_counts.get(poi.poi_id, 0)
            is_zero_shot = visit_count == 0

            # Choose icon color based on rank
            if i == 1:
                color = "green"
            elif i == 2:
                color = "orange"
            else:
                color = "blue"

            popup_html = f"""
            <div style="font-family: sans-serif;">
                <h4>{i}. {poi.name}</h4>
                <p><b>Score:</b> {item.candidate.score:.3f}</p>
                <p><b>Category:</b> {poi.category}</p>
                <p><b>Status:</b> {'🆕 New' if is_zero_shot else f'📍 Visited {visit_count}x'}</p>
            </div>
            """

            folium.Marker(
                [poi.latitude, poi.longitude],
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"#{i} {poi.name}",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)

    st_folium(m, width=700, height=600)

# Footer - User Activity Stats
st.header("📊 User Activity Statistics")

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    total_events = len(st.session_state.events)
    st.metric("Total Events", total_events)

with col_stat2:
    unique_locations = len(set(e.location for e in st.session_state.events))
    st.metric("Unique Locations", unique_locations)

with col_stat3:
    total_pois = len(st.session_state.ckg._pois)
    st.metric("Total POIs", total_pois)

with col_stat4:
    visited_pois = sum(1 for count in st.session_state.pkg.place_visit_counts.values() if count > 0)
    st.metric("Visited POIs", visited_pois)

# Activity timeline
with st.expander("📅 Recent Activity Timeline", expanded=False):
    recent_events = sorted(st.session_state.events, key=lambda e: e.timestamp, reverse=True)[:10]
    for event in recent_events:
        st.text(f"{event.timestamp.strftime('%Y-%m-%d %H:%M')} | {event.activity} @ {event.location} | App: {event.app_category}")