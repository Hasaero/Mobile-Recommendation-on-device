from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from .data_models import EventLogEntry, POI

BASE_TIME = datetime(2025, 9, 25, 7, 30)


def create_sample_events() -> List[EventLogEntry]:
    events: List[EventLogEntry] = []
    # Extended to 2 weeks of data
    day_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    for day in day_offsets:
        morning = BASE_TIME + timedelta(days=day)
        is_weekend = day % 7 in [5, 6]  # Saturday, Sunday

        if is_weekend:
            # Weekend routine: wake up later, more leisure activities
            events.extend(
                [
                    EventLogEntry(
                        timestamp=morning + timedelta(hours=2),  # Wake up at 9:30
                        activity="wake_up",
                        location="songdo_apartment",
                        app_category="alarm",
                    ),
                    EventLogEntry(
                        timestamp=morning + timedelta(hours=2, minutes=30),
                        activity="coffee_break",
                        location="bukbu_beach_cafe",
                        app_category="messenger",
                    ),
                    EventLogEntry(
                        timestamp=morning + timedelta(hours=4),
                        activity="meal",
                        location="jjukkumi_alley",
                        app_category="food_delivery",
                    ),
                    EventLogEntry(
                        timestamp=morning + timedelta(hours=6),
                        activity="coffee_break",
                        location="homigot_bakery",
                        app_category="social_media",
                    ),
                    EventLogEntry(
                        timestamp=morning + timedelta(hours=9),
                        activity="meal",
                        location="jjukkumi_alley",
                        app_category="food_delivery",
                    ),
                ]
            )
        else:
            # Weekday routine
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
                        timestamp=morning + timedelta(hours=9, minutes=30),
                        activity="work_end",
                        location="yeongil_science_park",
                        app_category="calendar",
                    ),
                    EventLogEntry(
                        timestamp=morning + timedelta(hours=11),
                        activity="exercise",
                        location="bukbu_fitness",
                        app_category="fitness",
                    ),
                    EventLogEntry(
                        timestamp=morning + timedelta(hours=12, minutes=30),
                        activity="errand",
                        location="yeongildae_mart",
                        app_category="shopping",
                    ),
                ]
            )

            # Add evening coffee occasionally (every other day)
            if day % 2 == 0:
                events.append(
                    EventLogEntry(
                        timestamp=morning + timedelta(hours=13, minutes=30),
                        activity="coffee_break",
                        location="bukbu_beach_cafe",
                        app_category="music",
                    )
                )

    return events


def create_sample_pois() -> List[POI]:
    return [
        # Visited POIs (user has history with these places)
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
            poi_id="yeongildae_mart",
            name="Yeongildae Mart",
            category="convenience_store",
            latitude=36.0537,
            longitude=129.3741,
            is_open=True,
        ),

        # Zero-shot POIs (user has never visited these places)
        # These should be recommended based on LLM understanding of context
        POI(
            poi_id="new_premium_gym",
            name="Premium Fitness Club",
            category="gym",
            latitude=36.0545,
            longitude=129.3720,
            is_open=True,
        ),
        POI(
            poi_id="sunrise_yoga_studio",
            name="Sunrise Yoga Studio",
            category="pilates",
            latitude=36.0558,
            longitude=129.3698,
            is_open=True,
        ),
        POI(
            poi_id="ocean_view_cafe",
            name="Ocean View Cafe",
            category="coffee_shop",
            latitude=36.0622,
            longitude=129.3812,
            is_open=True,
        ),
        POI(
            poi_id="bibimbap_house",
            name="Bibimbap House",
            category="korean_restaurant",
            latitude=36.0535,
            longitude=129.3755,
            is_open=True,
        ),
        POI(
            poi_id="pasta_paradise",
            name="Pasta Paradise",
            category="italian_restaurant",
            latitude=36.0512,
            longitude=129.3688,
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
            poi_id="artisan_bakery",
            name="Artisan Bakery & Cafe",
            category="bakery",
            latitude=36.0548,
            longitude=129.3732,
            is_open=True,
        ),
        POI(
            poi_id="convenience_24",
            name="24/7 Convenience Store",
            category="convenience_store",
            latitude=36.0542,
            longitude=129.3750,
            is_open=True,
        ),
        POI(
            poi_id="healthy_smoothie_bar",
            name="Healthy Smoothie Bar",
            category="cafe",
            latitude=36.0595,
            longitude=129.3668,
            is_open=True,
        ),
        POI(
            poi_id="zen_pilates",
            name="Zen Pilates Studio",
            category="pilates",
            latitude=36.0525,
            longitude=129.3715,
            is_open=True,
        ),

        # Seoul POIs (for unfamiliar city scenario)
        # User visits Seoul and needs recommendations without local history
        POI(
            poi_id="gangnam_fitness",
            name="Gangnam Fitness Center",
            category="gym",
            latitude=37.4979,
            longitude=127.0276,
            is_open=True,
        ),
        POI(
            poi_id="hongdae_coffee",
            name="Hongdae Artisan Coffee",
            category="coffee_shop",
            latitude=37.5563,
            longitude=126.9236,
            is_open=True,
        ),
        POI(
            poi_id="itaewon_pasta",
            name="Itaewon Italian Kitchen",
            category="italian_restaurant",
            latitude=37.5346,
            longitude=126.9946,
            is_open=True,
        ),
        POI(
            poi_id="myeongdong_korean",
            name="Myeongdong Traditional House",
            category="korean_restaurant",
            latitude=37.5636,
            longitude=126.9850,
            is_open=True,
        ),
        POI(
            poi_id="gangnam_cafe",
            name="Gangnam Luxury Cafe",
            category="coffee_shop",
            latitude=37.4980,
            longitude=127.0270,
            is_open=True,
        ),
        POI(
            poi_id="seoul_station_mart",
            name="Seoul Station Convenience",
            category="convenience_store",
            latitude=37.5547,
            longitude=126.9707,
            is_open=True,
        ),
        POI(
            poi_id="gangnam_bakery",
            name="Gangnam French Bakery",
            category="bakery",
            latitude=37.4985,
            longitude=127.0280,
            is_open=True,
        ),
        POI(
            poi_id="gangnam_pilates",
            name="Gangnam Premium Pilates",
            category="pilates",
            latitude=37.4975,
            longitude=127.0265,
            is_open=True,
        ),
    ]
