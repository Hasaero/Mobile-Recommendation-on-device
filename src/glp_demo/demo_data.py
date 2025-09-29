from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from .data_models import EventLogEntry, POI

BASE_TIME = datetime(2025, 9, 25, 7, 30)


def create_sample_events() -> List[EventLogEntry]:
    events: List[EventLogEntry] = []
    day_offsets = [0, 1, 2, 3]
    for day in day_offsets:
        morning = BASE_TIME + timedelta(days=day)
        events.extend(
            [
                EventLogEntry(
                    timestamp=morning,
                    activity="wake_up",
                    location="songdo_apartment",
                    app_category="alarm",
                ),
                EventLogEntry(
                    timestamp=morning + timedelta(minutes=10),
                    activity="coffee_break",
                    location="bukbu_beach_cafe",
                    app_category="news",
                ),
                EventLogEntry(
                    timestamp=morning + timedelta(minutes=40),
                    activity="commute",
                    location="pohang_bus_terminal",
                    app_category="maps",
                ),
                EventLogEntry(
                    timestamp=morning + timedelta(hours=1, minutes=20),
                    activity="work_start",
                    location="yeongil_science_park",
                    app_category="calendar",
                ),
                EventLogEntry(
                    timestamp=morning + timedelta(hours=5),
                    activity="meal",
                    location="jjukkumi_alley",
                    app_category="food_delivery",
                ),
                EventLogEntry(
                    timestamp=morning + timedelta(hours=19),
                    activity="exercise",
                    location="bukbu_fitness",
                    app_category="fitness",
                ),
            ]
        )
    weekend = BASE_TIME + timedelta(days=5, hours=9)
    events.append(
        EventLogEntry(
            timestamp=weekend,
            activity="coffee_break",
            location="homigot_bakery",
            app_category="messenger",
        )
    )
    return events


def create_sample_pois() -> List[POI]:
    return [
        POI(
            poi_id="bukbu_beach_cafe",
            name="Bukbu Beach Cafe",
            category="coffee_shop",
            latitude=36.0615,
            longitude=129.3795,
            is_open=True,
        ),
        POI(
            poi_id="jjukkumi_alley",
            name="Jjukkumi Alley",
            category="korean_restaurant",
            latitude=36.0328,
            longitude=129.3652,
            is_open=True,
        ),
        POI(
            poi_id="bukbu_fitness",
            name="Bukbu Fitness",
            category="gym",
            latitude=36.0591,
            longitude=129.3659,
            is_open=True,
        ),
        POI(
            poi_id="homigot_bakery",
            name="Homigot Bakery",
            category="bakery",
            latitude=36.0759,
            longitude=129.5682,
            is_open=True,
        ),
        POI(
            poi_id="cheonglim_pizza",
            name="Cheonglim Pizza",
            category="italian_restaurant",
            latitude=36.0443,
            longitude=129.3331,
            is_open=True,
        ),
        POI(
            poi_id="yeongildae_mart",
            name="Yeongildae Mart",
            category="convenience_store",
            latitude=36.0537,
            longitude=129.3741,
            is_open=True,
        ),
        # Zero-shot POI: 새로운 헬스장 (미방문)
        POI(
            poi_id="new_premium_gym",
            name="Premium Fitness Club",
            category="gym",
            latitude=36.0545,
            longitude=129.3720,
            is_open=True,
        ),
    ]
