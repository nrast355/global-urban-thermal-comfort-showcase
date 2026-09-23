from __future__ import annotations

import copy
import gzip
import json
import unicodedata
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from shapely.geometry import shape


st.set_page_config(
    page_title="全球城市历史舒适日数",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


DATA_DIR = Path(__file__).resolve().parent / "data"
MONTHLY_FILE = DATA_DIR / "city_year_month_dailyclass_counts.csv.gz"
MONTHLY_UTCI_FILE = DATA_DIR / "city_year_month_mean_utci.csv.gz"
BOUNDARY_FILE = DATA_DIR / "city_boundaries.geojson.gz"
MIGRATION_DIR = DATA_DIR / "migration"
MIGRATION_FILES = {
    "best_cities": MIGRATION_DIR / "SCI_walkability_36period_best_cities.csv",
    "city_climatology": MIGRATION_DIR / "SCI_walkability_36period_city_climatology.csv",
    "segments": MIGRATION_DIR / "SCI_walkability_migration_strategy_segments.csv",
    "summary": MIGRATION_DIR / "SCI_walkability_migration_strategy_summary.csv",
}

MONTHLY_REQUIRED_COLUMNS = {
    "city",
    "year",
    "month",
    "walkable_days",
    "valid_days",
}
MONTHLY_UTCI_REQUIRED_COLUMNS = {"city", "period", "mean_utci", "daily_records"}
BOUNDARY_CITY_PROPERTY = "city"

MIGRATION_REQUIRED_COLUMNS = {
    "best_cities": {
        "scope",
        "period",
        "period_label",
        "date_range",
        "best_city",
        "continent",
        "mean_walkable_days",
    },
    "city_climatology": {
        "city",
        "period_index",
        "mean_walkable_days",
        "sd_walkable_days",
        "mean_valid_days",
        "mean_utci",
        "period",
        "period_label",
        "start_month",
        "start_day",
        "end_month",
        "end_day",
        "date_range",
        "continent",
    },
    "segments": {
        "scope",
        "constraint",
        "allowed_migrations",
        "segment",
        "period_range",
        "start_period",
        "end_period",
        "city",
        "walkable_days",
    },
    "summary": {
        "scope",
        "constraint",
        "allowed_migrations",
        "used_migrations",
        "total_walkable_days",
        "route",
    },
}

# These are the fixed walkable-day classes used by the existing project
# distribution figures. Monthly observations are normalized to a 365-day
# equivalent before applying these annual-scale boundaries.
COMFORT_BOUNDARIES = [0, 100, 170, 240, 310, 365]
COMFORT_LEVELS = [
    ("0–100 days equivalent", "#b2182b"),
    ("100–170 days equivalent", "#ef8a62"),
    ("170–240 days equivalent", "#fdae61"),
    ("240–310 days equivalent", "#abd9e9"),
    ("310–365 days equivalent", "#74add1"),
]
COMFORT_LABELS = [label for label, _ in COMFORT_LEVELS]
COMFORT_COLORS = {label: color for label, color in COMFORT_LEVELS}
NO_DATA_LABEL = "No data"
NO_DATA_COLOR = "#c7c7c7"


def hex_to_rgba(color: str, alpha: int = 190) -> list[int]:
    """Convert a six-digit hex color to the RGBA format expected by deck.gl."""

    value = color.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex color, got {color!r}.")
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)] + [alpha]


COMFORT_FILL_COLORS = {label: hex_to_rgba(color) for label, color in COMFORT_LEVELS}
NO_DATA_FILL_COLOR = hex_to_rgba(NO_DATA_COLOR)


def city_key(value: object) -> str:
    """Normalize only Unicode form and surrounding whitespace for an exact join."""

    return unicodedata.normalize("NFC", str(value)).strip()


@st.cache_data(show_spinner=False)
def load_monthly_data() -> pd.DataFrame:
    if not MONTHLY_FILE.exists():
        raise FileNotFoundError(f"Missing monthly data file: {MONTHLY_FILE}")

    monthly = pd.read_csv(MONTHLY_FILE, compression="gzip")
    missing = sorted(MONTHLY_REQUIRED_COLUMNS - set(monthly.columns))
    if missing:
        raise ValueError(f"Monthly CSV is missing required columns: {missing}")

    monthly = monthly.copy()
    monthly["city_key"] = monthly["city"].map(city_key)
    for column in ["year", "month", "walkable_days", "valid_days"]:
        monthly[column] = pd.to_numeric(monthly[column], errors="raise")

    if (
        monthly[["city_key", "year", "month", "walkable_days", "valid_days"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Monthly data contain missing join, time, or day-count values."
        )
    if not monthly["month"].between(1, 12).all():
        raise ValueError("Monthly data contain month values outside 1–12.")
    if (monthly["valid_days"] <= 0).any():
        raise ValueError("Monthly data contain non-positive valid_days values.")
    if (monthly["walkable_days"] < 0).any():
        raise ValueError("Monthly data contain negative walkable_days values.")
    if (monthly["walkable_days"] > monthly["valid_days"]).any():
        raise ValueError("Monthly data contain walkable_days greater than valid_days.")

    duplicate_keys = monthly.duplicated(["city_key", "year", "month"])
    if duplicate_keys.any():
        raise ValueError(
            "Monthly data contain duplicate city-year-month keys: "
            f"{int(duplicate_keys.sum())} duplicates."
        )

    monthly["year"] = monthly["year"].astype(int)
    monthly["month"] = monthly["month"].astype(int)
    monthly["walkable_days"] = monthly["walkable_days"].astype(int)
    monthly["valid_days"] = monthly["valid_days"].astype(int)
    monthly["period_date"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1),
        errors="raise",
    )
    monthly["period"] = monthly["period_date"].dt.strftime("%Y-%m")

    if not MONTHLY_UTCI_FILE.exists():
        raise FileNotFoundError(f"Missing monthly UTCI file: {MONTHLY_UTCI_FILE}")
    monthly_utci = pd.read_csv(MONTHLY_UTCI_FILE, compression="gzip")
    missing_utci_columns = sorted(
        MONTHLY_UTCI_REQUIRED_COLUMNS - set(monthly_utci.columns)
    )
    if missing_utci_columns:
        raise ValueError(
            "Monthly UTCI CSV is missing required columns: "
            f"{missing_utci_columns}"
        )

    monthly_utci = monthly_utci.copy()
    monthly_utci["city_key"] = monthly_utci["city"].map(city_key)
    monthly_utci["period"] = monthly_utci["period"].astype(str).str.strip()
    period_dates = pd.to_datetime(
        monthly_utci["period"] + "-01", format="%Y-%m-%d", errors="coerce"
    )
    if period_dates.isna().any():
        raise ValueError("Monthly UTCI data contain invalid YYYY-MM periods.")
    raw_utci_values = monthly_utci["mean_utci"].copy()
    monthly_utci["mean_utci"] = pd.to_numeric(
        raw_utci_values, errors="coerce"
    )
    invalid_utci_values = monthly_utci["mean_utci"].isna() & raw_utci_values.notna()
    if invalid_utci_values.any():
        raise ValueError("Monthly UTCI data contain non-numeric mean_utci values.")
    monthly_utci["daily_records"] = pd.to_numeric(
        monthly_utci["daily_records"], errors="coerce"
    )
    if monthly_utci["daily_records"].isna().any() or (
        monthly_utci["daily_records"] <= 0
    ).any():
        raise ValueError("Monthly UTCI data contain invalid daily_records values.")
    if monthly_utci["city_key"].eq("").any():
        raise ValueError("Monthly UTCI data contain empty city names.")
    if monthly_utci.duplicated(["city_key", "period"]).any():
        raise ValueError("Monthly UTCI data contain duplicate city-period keys.")

    monthly = monthly.merge(
        monthly_utci[["city_key", "period", "mean_utci"]],
        on=["city_key", "period"],
        how="left",
        validate="one_to_one",
    )
    monthly["comfortable_days"] = monthly["walkable_days"]
    return monthly.sort_values(["period_date", "city_key"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_boundaries() -> dict:
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"Missing city-boundary file: {BOUNDARY_FILE}")

    with gzip.open(BOUNDARY_FILE, "rt", encoding="utf-8") as source:
        boundaries = json.load(source)
    features = boundaries.get("features", [])
    if boundaries.get("type") != "FeatureCollection":
        raise ValueError("City boundaries must be a GeoJSON FeatureCollection.")
    if len(features) != 120:
        raise ValueError(f"Expected 120 city boundaries, found {len(features)}.")

    city_values = []
    for feature in features:
        properties = feature.get("properties", {})
        if BOUNDARY_CITY_PROPERTY not in properties:
            raise ValueError(
                "City boundary properties do not contain the expected "
                f"'{BOUNDARY_CITY_PROPERTY}' field."
            )
        if not feature.get("geometry"):
            raise ValueError("A city boundary feature has no geometry.")
        city_values.append(city_key(properties[BOUNDARY_CITY_PROPERTY]))

    if any(not value for value in city_values):
        raise ValueError("City boundary properties contain an empty city value.")
    if len(set(city_values)) != len(city_values):
        raise ValueError("City boundary city values are not unique.")
    return boundaries


@st.cache_data(show_spinner=False)
def load_migration_data() -> dict[str, pd.DataFrame]:
    """Load and validate the four precomputed migration-result tables."""

    frames: dict[str, pd.DataFrame] = {}
    for name, path in MIGRATION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing migration data file: {path}")
        frame = pd.read_csv(path, encoding="utf-8")
        missing = sorted(MIGRATION_REQUIRED_COLUMNS[name] - set(frame.columns))
        if missing:
            raise ValueError(f"Migration CSV {path.name} is missing columns: {missing}")
        frames[name] = frame

    best_cities = frames["best_cities"].copy()
    best_cities["period"] = pd.to_numeric(best_cities["period"], errors="raise")
    best_cities["mean_walkable_days"] = pd.to_numeric(
        best_cities["mean_walkable_days"], errors="raise"
    )

    city_climatology = frames["city_climatology"].copy()
    for column in [
        "period_index",
        "period",
        "start_month",
        "start_day",
        "end_month",
        "end_day",
    ]:
        city_climatology[column] = pd.to_numeric(
            city_climatology[column], errors="raise"
        )
    for column in [
        "mean_walkable_days",
        "sd_walkable_days",
        "mean_valid_days",
        "mean_utci",
    ]:
        city_climatology[column] = pd.to_numeric(
            city_climatology[column], errors="raise"
        )

    segments = frames["segments"].copy()
    for column in ["allowed_migrations", "segment"]:
        segments[column] = pd.to_numeric(segments[column], errors="raise").astype(int)
    segments["walkable_days"] = pd.to_numeric(segments["walkable_days"], errors="raise")

    summary = frames["summary"].copy()
    for column in ["allowed_migrations", "used_migrations"]:
        summary[column] = pd.to_numeric(summary[column], errors="raise").astype(int)
    summary["total_walkable_days"] = pd.to_numeric(
        summary["total_walkable_days"], errors="raise"
    )

    frames["best_cities"] = best_cities
    frames["city_climatology"] = city_climatology
    frames["segments"] = segments
    frames["summary"] = summary
    return frames


def validate_city_join(monthly: pd.DataFrame, boundaries: dict) -> dict[str, list[str]]:
    data_cities = set(monthly["city_key"])
    boundary_cities = {
        city_key(feature["properties"][BOUNDARY_CITY_PROPERTY])
        for feature in boundaries["features"]
    }
    return {
        "data_without_boundary": sorted(data_cities - boundary_cities),
        "boundary_without_data": sorted(boundary_cities - data_cities),
    }


def validate_period_coverage(
    monthly: pd.DataFrame, expected_cities: set[str]
) -> dict[str, object]:
    """Check that the available periods are continuous and city-complete."""

    actual_periods = set(monthly["period"])
    expected_periods = (
        pd.date_range(
            monthly["period_date"].min(), monthly["period_date"].max(), freq="MS"
        )
        .strftime("%Y-%m")
        .tolist()
    )
    missing_periods = [
        period for period in expected_periods if period not in actual_periods
    ]

    incomplete_periods = {}
    for period, values in monthly.groupby("period", sort=True)["city_key"]:
        missing_cities = sorted(expected_cities - set(values))
        if missing_cities:
            incomplete_periods[period] = missing_cities

    return {
        "missing_periods": missing_periods,
        "incomplete_periods": incomplete_periods,
    }


def build_city_anchor_map(boundaries: dict) -> dict[str, list[float]]:
    """Derive stable in-polygon WGS84 anchors from the UC boundary geometry."""

    anchors: dict[str, list[float]] = {}
    for feature in boundaries["features"]:
        properties = feature["properties"]
        city = city_key(properties[BOUNDARY_CITY_PROPERTY])
        geometry = shape(feature["geometry"])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Invalid or empty boundary geometry for {city}.")
        point = geometry.representative_point()
        longitude = float(point.x)
        latitude = float(point.y)
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError(f"Boundary anchor for {city} is outside WGS84 bounds.")
        anchors[city] = [longitude, latitude]
    return anchors


def migration_city_names(migration_data: dict[str, pd.DataFrame]) -> set[str]:
    names: set[str] = set()
    for frame_name, column in [
        ("best_cities", "best_city"),
        ("city_climatology", "city"),
        ("segments", "city"),
    ]:
        names.update(city_key(value) for value in migration_data[frame_name][column])
    return names


def format_days(value: object) -> str:
    number = float(value)
    return f"{number:.2f}".rstrip("0").rstrip(".")


def format_utci(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f} °C"


def build_migration_routes(
    migration_data: dict[str, pd.DataFrame],
    city_anchors: dict[str, list[float]],
    scope: str,
    constraint: str,
    allowed_migrations: int,
    selected_migration_period: int | None,
) -> tuple[list[dict], list[dict]]:
    """Build ArcLayer and endpoint records from adjacent strategy segments.

    The migration period attached to a route is the actual 36-period label at
    which the destination segment starts. It is deliberately independent of
    the historical YYYY-MM slider.
    """

    segments = migration_data["segments"]
    selected_segments = segments.loc[
        segments["scope"].eq(scope)
        & segments["constraint"].eq(constraint)
        & segments["allowed_migrations"].eq(allowed_migrations)
    ].sort_values("segment")
    if selected_segments.empty:
        return [], []

    climatology = migration_data["city_climatology"]
    period_lookup = (
        climatology[["period", "period_label"]]
        .drop_duplicates()
        .assign(period=lambda frame: frame["period"].astype(int))
        .set_index("period_label")["period"]
        .to_dict()
    )

    summary = migration_data["summary"]
    summary_rows = summary.loc[
        summary["scope"].eq(scope)
        & summary["constraint"].eq(constraint)
        & summary["allowed_migrations"].eq(allowed_migrations)
    ]
    strategy_total = (
        float(summary_rows.iloc[0]["total_walkable_days"])
        if not summary_rows.empty
        else None
    )
    strategy_text = f"{scope} · {constraint} · allowed migrations: {allowed_migrations}"

    routes: list[dict] = []
    endpoint_records: dict[str, dict] = {}
    segment_records = list(selected_segments.to_dict(orient="records"))
    for previous, current in zip(segment_records, segment_records[1:]):
        origin = city_key(previous["city"])
        destination = city_key(current["city"])
        if origin == destination:
            continue
        transition_period = period_lookup.get(str(current["start_period"]))
        if (
            selected_migration_period is not None
            and transition_period != selected_migration_period
        ):
            continue
        if origin not in city_anchors or destination not in city_anchors:
            continue

        period_range = str(current["period_range"])
        tooltip_line3 = f"Strategy: {strategy_text}"
        if strategy_total is not None:
            tooltip_line3 += f" · total: {format_days(strategy_total)} days"
        route = {
            "source_position": city_anchors[origin],
            "target_position": city_anchors[destination],
            "width": 3.5,
            "origin": origin,
            "destination": destination,
            "transition_period": transition_period,
            "tooltip_title": f"{origin} → {destination}",
            "tooltip_line1": f"Migration period: {period_range}",
            "tooltip_line2": (
                "Walkable days: "
                f"{format_days(previous['walkable_days'])} → "
                f"{format_days(current['walkable_days'])}"
            ),
            "tooltip_line3": tooltip_line3,
            "tooltip_line4": "",
        }
        routes.append(route)

        for city, role in [(origin, "Origin"), (destination, "Destination")]:
            endpoint_records[city] = {
                "position": city_anchors[city],
                "city": city,
                "tooltip_title": city,
                "tooltip_line1": f"Migration {role.lower()}",
                "tooltip_line2": f"Route period: {period_range}",
                "tooltip_line3": strategy_text,
                "tooltip_line4": "",
            }

    return routes, list(endpoint_records.values())


def assign_comfort_level(annualized_days: float) -> tuple[str, str]:
    if pd.isna(annualized_days) or not 0 <= annualized_days <= COMFORT_BOUNDARIES[-1]:
        return NO_DATA_LABEL, NO_DATA_COLOR
    for index, label in enumerate(COMFORT_LABELS):
        low = COMFORT_BOUNDARIES[index]
        high = COMFORT_BOUNDARIES[index + 1]
        if low <= annualized_days < high or (
            index == len(COMFORT_LABELS) - 1 and annualized_days <= high
        ):
            return label, COMFORT_COLORS[label]
    return NO_DATA_LABEL, NO_DATA_COLOR


def build_period_geojson(
    boundaries: dict, period_data: pd.DataFrame, period: str
) -> tuple[dict, list[str]]:
    period_lookup = period_data.set_index("city_key").to_dict(orient="index")
    missing_period_cities = []
    output = copy.deepcopy(boundaries)

    for feature in output["features"]:
        properties = feature["properties"]
        key = city_key(properties[BOUNDARY_CITY_PROPERTY])
        row = period_lookup.get(key)
        if row is None:
            city = str(properties[BOUNDARY_CITY_PROPERTY])
            tooltip_values = {
                "city": city,
                "period": period,
                "comfortable_days": None,
                "mean_utci": None,
                "comfort_level": NO_DATA_LABEL,
                "tooltip_title": f"City: {city}",
                "tooltip_line1": f"Year-Month: {period}",
                "tooltip_line2": "Comfortable days: No data",
                "tooltip_line3": "Mean UTCI: N/A",
                "tooltip_line4": f"Comfort level: {NO_DATA_LABEL}",
            }
            properties.update(
                {
                    **tooltip_values,
                    "fill_color": NO_DATA_FILL_COLOR.copy(),
                }
            )
            feature.update(tooltip_values)
            missing_period_cities.append(key)
            continue

        annualized_days = round(
            float(row["walkable_days"] / row["valid_days"] * 365), 1
        )
        comfort_level, _ = assign_comfort_level(annualized_days)
        city = str(properties[BOUNDARY_CITY_PROPERTY])
        mean_utci = row["mean_utci"]
        mean_utci_value = None if pd.isna(mean_utci) else float(mean_utci)
        tooltip_values = {
            "city": city,
            "period": period,
            "comfortable_days": int(row["comfortable_days"]),
            "mean_utci": mean_utci_value,
            "comfort_level": comfort_level,
            "tooltip_title": f"City: {city}",
            "tooltip_line1": f"Year-Month: {period}",
            "tooltip_line2": f"Comfortable days: {int(row['comfortable_days'])}",
            "tooltip_line3": f"Mean UTCI: {format_utci(mean_utci_value)}",
            "tooltip_line4": f"Comfort level: {comfort_level}",
        }
        properties.update(
            {
                **tooltip_values,
                "fill_color": COMFORT_FILL_COLORS.get(
                    comfort_level, NO_DATA_FILL_COLOR
                ).copy(),
            }
        )
        feature.update(tooltip_values)
    return output, sorted(missing_period_cities)


def selected_city_from_map_event(
    map_event: object, valid_city_keys: set[str]
) -> str | None:
    """Return the city key selected from the GeoJSON layer, if any.

    Streamlit exposes PyDeck selections through an event object's ``selection``
    mapping.  GeoJsonLayer selections contain a GeoJSON Feature with the city
    name under ``properties``; migration routes and endpoints do not, so they
    are intentionally ignored here.
    """

    if map_event is None:
        return None
    if isinstance(map_event, dict):
        selection = map_event.get("selection")
    else:
        selection = getattr(map_event, "selection", None)
    if not isinstance(selection, dict):
        return None

    objects = selection.get("objects", {})
    if not isinstance(objects, dict):
        return None

    for layer_id, layer_objects in objects.items():
        if "city" not in str(layer_id).lower() and "geojson" not in str(
            layer_id
        ).lower():
            continue
        if isinstance(layer_objects, dict):
            layer_objects = [layer_objects]
        if not isinstance(layer_objects, list):
            continue
        for selected_object in layer_objects:
            if not isinstance(selected_object, dict):
                continue
            properties = selected_object.get("properties")
            selected_city = (
                properties.get(BOUNDARY_CITY_PROPERTY)
                if isinstance(properties, dict)
                else None
            )
            if selected_city is None:
                selected_city = selected_object.get(BOUNDARY_CITY_PROPERTY)
            if selected_city is None:
                continue
            selected_key = city_key(selected_city)
            if selected_key in valid_city_keys:
                return selected_key
    return None


MAP_TOOLTIP = {
    "html": (
        "<b>{tooltip_title}</b><br/>"
        "{tooltip_line1}<br/>"
        "{tooltip_line2}<br/>"
        "{tooltip_line3}<br/>"
        "{tooltip_line4}"
    ),
    "style": {
        "backgroundColor": "rgba(25, 25, 25, 0.9)",
        "color": "white",
        "fontSize": "12px",
    },
}


def make_map(
    period_geojson: dict,
    migration_routes: list[dict] | None = None,
    migration_endpoints: list[dict] | None = None,
) -> pdk.Deck:
    city_layer = pdk.Layer(
        "GeoJsonLayer",
        id="city-boundaries",
        data=period_geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[75, 75, 75, 190],
        get_line_width=1,
        line_width_min_pixels=0.45,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 220],
    )
    layers = [city_layer]
    if migration_routes:
        layers.append(
            pdk.Layer(
                "ArcLayer",
                id="migration-routes",
                data=migration_routes,
                pickable=True,
                auto_highlight=True,
                get_source_position="source_position",
                get_target_position="target_position",
                get_source_color=[28, 92, 140, 190],
                get_target_color=[215, 70, 64, 210],
                get_width="width",
                great_circle=True,
            )
        )
    if migration_endpoints:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="migration-endpoints",
                data=migration_endpoints,
                pickable=True,
                auto_highlight=True,
                get_position="position",
                get_radius=45000,
                radius_min_pixels=4,
                radius_max_pixels=12,
                get_fill_color=[255, 255, 255, 235],
                get_line_color=[28, 92, 140, 255],
                line_width_min_pixels=1.5,
                stroked=True,
            )
        )
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=15,
            longitude=0,
            zoom=0.35,
            min_zoom=0.2,
            max_zoom=8,
            pitch=0,
            bearing=0,
        ),
        map_style="light",
        tooltip=MAP_TOOLTIP,
    )


def render_legend() -> None:
    items = []
    for label, color in COMFORT_LEVELS:
        items.append(
            "<div style='display:flex;align-items:center;gap:8px;margin:3px 12px 3px 0;'>"
            f"<span style='display:inline-block;width:18px;height:18px;background:{color};"
            "border:1px solid #666;border-radius:2px;'></span>"
            f"<span>{label}</span></div>"
        )
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;align-items:center;margin:4px 0 8px;'>"
        + "".join(items)
        + "</div>",
        unsafe_allow_html=True,
    )


try:
    monthly = load_monthly_data()
    boundaries = load_boundaries()
    migration_data = load_migration_data()
    city_anchors = build_city_anchor_map(boundaries)
except (FileNotFoundError, ValueError):
    st.info(
        "This public repository contains the application source code only. "
        "Research datasets are not distributed through this repository. "
        "Please use the live demo to explore the application."
    )
    st.stop()

join_report = validate_city_join(monthly, boundaries)
if join_report["data_without_boundary"] or join_report["boundary_without_data"]:
    st.error("城市边界与月度数据无法完成匹配，地图暂不可用。")
    st.stop()

if migration_city_names(migration_data) - set(city_anchors):
    st.error("迁移路线城市无法与城市边界完成匹配，地图暂不可用。")
    st.stop()

city_display_names = {
    city_key(row.city): str(row.city)
    for row in monthly[["city_key", "city"]]
    .drop_duplicates("city_key")
    .itertuples(index=False)
}
valid_city_keys = set(city_display_names)
default_city_key = (
    city_key("Shanghai") if "Shanghai" in valid_city_keys else sorted(valid_city_keys)[0]
)
selected_city_key = city_key(
    st.session_state.get("selected_city", default_city_key)
)
if selected_city_key not in valid_city_keys:
    selected_city_key = default_city_key
st.session_state["selected_city"] = city_display_names[selected_city_key]

periods = monthly["period"].drop_duplicates().tolist()
periods = sorted(periods)
if not periods:
    st.error("当前数据没有可用的历史时期。")
    st.stop()

coverage_report = validate_period_coverage(
    monthly,
    {
        city_key(feature["properties"][BOUNDARY_CITY_PROPERTY])
        for feature in boundaries["features"]
    },
)
if coverage_report["missing_periods"] or coverage_report["incomplete_periods"]:
    st.error("当前历史数据的时间覆盖不完整，地图暂不可用。")
    st.stop()

if (
    "period_slider" not in st.session_state
    or st.session_state["period_slider"] not in periods
):
    st.session_state["period_slider"] = periods[-1]
selected_period = st.session_state["period_slider"]
period_data = monthly.loc[monthly["period"].eq(selected_period)].copy()
period_geojson, missing_period_cities = build_period_geojson(
    boundaries, period_data, selected_period
)

st.title("全球 120 城市历史舒适日数与 UTCI 时空变化")
st.caption(
    f"以城市 Urban Centre 边界为掩膜，拖动时间条查看 {periods[0]}–{periods[-1]} 的全球变化，点击城市查看长期趋势。"
)

migration_routes: list[dict] = []
migration_endpoints: list[dict] = []
show_migration_routes = st.checkbox("Show migration routes", value=False)
if show_migration_routes:
    segments = migration_data["segments"]
    scope_order = [
        "Global",
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "Oceania",
        "South America",
    ]
    scope_options = [scope for scope in scope_order if scope in set(segments["scope"])]
    scope_options.extend(sorted(set(segments["scope"]) - set(scope_options)))
    control_columns = st.columns(4)
    with control_columns[0]:
        selected_scope = st.selectbox("Migration scope", scope_options)
    with control_columns[1]:
        constraint_options = sorted(
            segments.loc[segments["scope"].eq(selected_scope), "constraint"].unique()
        )
        selected_constraint = st.selectbox(
            "Constraint", constraint_options, format_func=lambda value: value
        )
    with control_columns[2]:
        allowed_options = sorted(
            segments.loc[
                segments["scope"].eq(selected_scope)
                & segments["constraint"].eq(selected_constraint),
                "allowed_migrations",
            ]
            .astype(int)
            .unique()
        )
        selected_allowed_migrations = st.selectbox(
            "Allowed migrations",
            allowed_options,
            index=allowed_options.index(1) if 1 in allowed_options else 0,
        )
    with control_columns[3]:
        period_options_frame = (
            migration_data["city_climatology"][["period", "period_label"]]
            .drop_duplicates()
            .sort_values("period")
        )
        migration_period_options = ["All route transitions"] + [
            f"{int(row.period):02d} · {row.period_label}"
            for row in period_options_frame.itertuples(index=False)
        ]
        selected_migration_period_label = st.selectbox(
            "Migration period", migration_period_options
        )
        migration_period_lookup = {
            option: None for option in [migration_period_options[0]]
        }
        migration_period_lookup.update(
            {
                f"{int(row.period):02d} · {row.period_label}": int(row.period)
                for row in period_options_frame.itertuples(index=False)
            }
        )
        selected_migration_period = migration_period_lookup[
            selected_migration_period_label
        ]

    migration_routes, migration_endpoints = build_migration_routes(
        migration_data,
        city_anchors,
        selected_scope,
        selected_constraint,
        int(selected_allowed_migrations),
        selected_migration_period,
    )

if missing_period_cities:
    st.error("当前时期缺少城市数据，地图暂不可用。")
else:
    st.subheader(f"当前时期：{selected_period}")
    map_event = st.pydeck_chart(
        make_map(period_geojson, migration_routes, migration_endpoints),
        width="stretch",
        selection_mode="single-object",
        on_select="rerun",
        key="city_map",
    )
    clicked_city_key = selected_city_from_map_event(map_event, valid_city_keys)
    if clicked_city_key is not None:
        st.session_state["selected_city"] = city_display_names[clicked_city_key]
        selected_city_key = clicked_city_key

st.markdown("**舒适日数等级**")
render_legend()
st.caption(
    "等级沿用项目既有 0–365 天舒适日数五级色带；月度 walkable_days 按当月 valid_days 换算为 365 天等效值后分级。"
)

st.select_slider(
    "历史时期（YYYY-MM）",
    options=periods,
    key="period_slider",
)

selected_city = city_display_names[selected_city_key]
st.subheader(f"City details — {selected_city}")
st.caption("点击上方地图中的城市边界，自动切换下方 30 年趋势。")
city_trend = monthly.loc[monthly["city_key"].eq(selected_city_key)].copy()
city_trend = city_trend.sort_values("period_date")

if city_trend.empty:
    st.info("No time-series data are available for this city.")
else:
    selected_city_period = city_trend.loc[
        city_trend["period"].eq(selected_period)
    ]
    if not selected_city_period.empty:
        selected_city_row = selected_city_period.iloc[0]
        summary_columns = st.columns(3)
        with summary_columns[0]:
            st.metric("Selected month", selected_period)
        with summary_columns[1]:
            st.metric("Mean UTCI", format_utci(selected_city_row["mean_utci"]))
        with summary_columns[2]:
            comfortable_days = selected_city_row["comfortable_days"]
            days_text = (
                "N/A" if pd.isna(comfortable_days) else f"{int(comfortable_days)}"
            )
            st.metric("Comfortable days", days_text)

    chart_frame = city_trend.set_index("period_date")
    utci_chart = chart_frame[["mean_utci"]].rename(
        columns={"mean_utci": "Monthly mean UTCI"}
    )
    utci_chart["12-month rolling mean"] = utci_chart[
        "Monthly mean UTCI"
    ].rolling(window=12, min_periods=12).mean()
    st.subheader(f"Monthly Mean UTCI — {selected_city}")
    if utci_chart["Monthly mean UTCI"].notna().any():
        st.line_chart(
            utci_chart,
            x_label="Time",
            y_label="UTCI (°C)",
            height=320,
        )
    else:
        st.info("No UTCI observations are available for this city.")

    comfortable_days_chart = chart_frame[["comfortable_days"]].rename(
        columns={"comfortable_days": "Monthly comfortable days"}
    )
    comfortable_days_chart["12-month rolling mean"] = comfortable_days_chart[
        "Monthly comfortable days"
    ].rolling(window=12, min_periods=12).mean()
    st.subheader(f"Monthly Comfortable Days — {selected_city}")
    st.line_chart(
        comfortable_days_chart,
        x_label="Time",
        y_label="Comfortable days",
        height=320,
    )
