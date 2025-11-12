import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
import inspect
import logging
import math
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.util import dt as dt_util, slugify

from .const import DOMAIN, ENERGA_BASE_URL, ENERGA_PAGE_URL, UPDATE_INTERVAL


_LOGGER = logging.getLogger(__name__)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA

DEFAULT_ZONE_RADIUS_M = 100.0
EARTH_RADIUS_M = 6_371_008.8


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info=None,
):
    """Legacy setup using YAML configuration."""
    coordinator = PowerOutageCoordinator(hass)
    try:
        await coordinator.async_refresh()
    except UpdateFailed as err:
        raise PlatformNotReady(err) from err

    manager = ZoneEntityManager(hass, coordinator, async_add_entities)
    await manager.async_initialize(update_before_add=True)

    hass.data.setdefault(DOMAIN, {}).setdefault("legacy_managers", []).append(manager)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
):
    coordinator = PowerOutageCoordinator(hass)
    await coordinator.async_config_entry_first_refresh()

    manager = ZoneEntityManager(hass, coordinator, async_add_entities)
    await manager.async_initialize(update_before_add=True)

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "coordinator": coordinator,
        "manager": manager,
    }

    entry.async_on_unload(lambda: hass.async_create_task(manager.async_shutdown()))


class PowerOutageCoordinator(DataUpdateCoordinator):
    def __init__(self, hass: HomeAssistant) -> None:
        self._zones: dict[str, dict[str, Any]] = {}
        super().__init__(
            hass,
            _LOGGER,
            name="Power Outage",
            update_interval=timedelta(seconds=UPDATE_INTERVAL),
        )

    async def _async_update_data(self):
        """Fetch and parse shutdown data."""
        self.update_zones()

        zones_snapshot = {zone_id: zone.copy() for zone_id, zone in self._zones.items()}
        if not zones_snapshot:
            return {}

        return await asyncio.to_thread(fetch_shutdowns, zones_snapshot)

    @property
    def zones(self) -> dict[str, dict[str, Any]]:
        """Return the currently resolved zones."""
        return self._zones

    def update_zones(self) -> bool:
        """Refresh stored zones from Home Assistant state."""
        zones = _resolve_zones(self.hass)
        if zones == self._zones:
            return False
        self._zones = zones
        return True


class ZoneEntityManager:
    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: PowerOutageCoordinator,
        async_add_entities: AddEntitiesCallback,
    ) -> None:
        self._hass = hass
        self._coordinator = coordinator
        self._async_add_entities = async_add_entities
        self._entities: dict[str, PowerOutageSensor] = {}
        self._unsub_zone_listener: Callable[[], None] | None = None
        self._remove_coordinator_listener: Callable[[], None] | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def async_initialize(self, update_before_add: bool = False) -> None:
        if self._initialized:
            return

        await self._sync_entities(update_before_add)

        self._remove_coordinator_listener = self._coordinator.async_add_listener(
            self._handle_coordinator_update
        )
        self._unsub_zone_listener = self._hass.bus.async_listen(
            EVENT_STATE_CHANGED, self._handle_zone_event
        )

        self._initialized = True

    async def async_shutdown(self) -> None:
        if self._remove_coordinator_listener is not None:
            self._remove_coordinator_listener()
            self._remove_coordinator_listener = None
        if self._unsub_zone_listener is not None:
            self._unsub_zone_listener()
            self._unsub_zone_listener = None

        async with self._lock:
            self._entities.clear()

        self._initialized = False

    async def _sync_entities(self, update_before_add: bool = False) -> None:
        async with self._lock:
            current_zone_ids = set(self._coordinator.zones)

            new_entities: list[PowerOutageSensor] = []
            for zone_id in current_zone_ids:
                if zone_id not in self._entities:
                    sensor = PowerOutageSensor(self._coordinator, zone_id)
                    self._entities[zone_id] = sensor
                    new_entities.append(sensor)

            if new_entities:
                result = self._async_add_entities(
                    new_entities,
                    update_before_add=update_before_add,
                )
                if inspect.isawaitable(result):
                    await result

            removed_zone_ids = [zone_id for zone_id in self._entities if zone_id not in current_zone_ids]
            for zone_id in removed_zone_ids:
                entity = self._entities.pop(zone_id, None)
                if entity is not None:
                    await entity.async_remove()

            for entity in self._entities.values():
                if entity.entity_id:
                    entity.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        self._hass.async_create_task(self._sync_entities())

    @callback
    def _handle_zone_event(self, event) -> None:
        entity_id = event.data.get("entity_id")
        if not entity_id or not entity_id.startswith("zone."):
            return

        self._hass.async_create_task(self._process_zone_change())

    async def _process_zone_change(self) -> None:
        if not self._coordinator.update_zones():
            return

        await self._sync_entities()
        await self._coordinator.async_request_refresh()


def fetch_shutdowns(zones: dict[str, dict[str, Any]]):
    """Fetch data from Energa and return relevant outages grouped by zone."""
    results: dict[str, list[dict[str, str | None]]] = {
        zone_id: [] for zone_id in zones
    }

    if not zones:
        return results

    try:
        response = requests.get(ENERGA_PAGE_URL, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        div = soup.find("div", attrs={"data-shutdowns": True})
        if not div:
            raise UpdateFailed("data-shutdowns not found")
        if not isinstance(div, Tag):
            raise UpdateFailed("data-shutdowns container malformed")

        shutdowns_path = div.attrs.get("data-shutdowns")
        if isinstance(shutdowns_path, list):
            shutdowns_path = shutdowns_path[0] if shutdowns_path else None
        if not shutdowns_path:
            raise UpdateFailed("data-shutdowns attribute missing")

        json_url = urljoin(ENERGA_BASE_URL, str(shutdowns_path))
        json_response = requests.get(json_url, timeout=15)
        json_response.raise_for_status()
        data = json_response.json()

        payload = data.get("document", {}).get("payload", {})
        shutdowns = payload.get("shutdowns")
        if isinstance(shutdowns, list):
            for shutdown in shutdowns:
                for zone_id, zone in zones.items():
                    if _shutdown_overlaps_zone(shutdown, zone):
                        results[zone_id].append(
                            {
                                "message": shutdown.get("message", ""),
                                "startDate": shutdown.get("startDate"),
                                "endDate": shutdown.get("endDate"),
                                "zone": zone.get("name"),
                            }
                        )

        return results
    except UpdateFailed:
        raise
    except requests.exceptions.RequestException as err:
        raise UpdateFailed(f"Network error fetching Energa data: {err}") from err
    except ValueError as err:
        raise UpdateFailed(f"Invalid JSON from Energa: {err}") from err
    except Exception as err:
        raise UpdateFailed(f"Unexpected error fetching Energa data: {err}") from err


def _shutdown_overlaps_zone(
    shutdown: dict[str, Any], zone: dict[str, Any]
) -> bool:
    if not isinstance(shutdown, dict) or not zone:
        return False

    # Each shutdown supplies polygon geometry. We convert that into plain coordinate
    # pairs and treat the zone as a circle. The zone overlaps the shutdown if
    # either (a) the polygon intersects the circle or (b) the polygon's centroid
    # lands within the zone's radius.
    polygon = shutdown.get("polygon")
    if isinstance(polygon, dict):
        points = _extract_polygon_points(polygon)
        if points and _zone_overlaps_polygon(zone, points):
            return True
        centroid = _extract_centroid(polygon)
        if centroid and _circle_contains_point(zone, centroid):
            return True

    return False


def _extract_polygon_points(polygon: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def _collect(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and all(isinstance(value, (int, float)) for value in node[:2]):
                lon, lat = node[0], node[1]
                points.append((float(lat), float(lon)))
            else:
                for item in node:
                    _collect(item)

    coords = None
    poly = polygon.get("poly")
    if isinstance(poly, dict):
        coords = poly.get("coordinates")
    elif isinstance(polygon.get("coordinates"), (list, tuple)):
        coords = polygon.get("coordinates")

    _collect(coords)
    return points


def _extract_centroid(polygon: dict[str, Any]) -> tuple[float, float] | None:
    centroid_data = polygon.get("centroid") if isinstance(polygon, dict) else None
    centroid = centroid_data.get("coordinates") if isinstance(centroid_data, dict) else None

    if (
        isinstance(centroid, (list, tuple))
        and len(centroid) >= 2
        and all(isinstance(value, (int, float)) for value in centroid[:2])
    ):
        lon, lat = centroid[:2]
        return float(lat), float(lon)
    return None


def _zone_overlaps_polygon(
    zone: dict[str, Any], polygon_points: list[tuple[float, float]]
) -> bool:
    if not polygon_points:
        return False

    latitude = zone.get("latitude")
    longitude = zone.get("longitude")
    radius = float(zone.get("radius", DEFAULT_ZONE_RADIUS_M) or DEFAULT_ZONE_RADIUS_M)

    if latitude is None or longitude is None:
        return False

    if radius <= 0:
        radius = DEFAULT_ZONE_RADIUS_M

    # First check whether the zone centre falls inside the polygon (ray casting).
    if _point_in_polygon(latitude, longitude, polygon_points):
        return True

    # Otherwise measure the shortest distance from the centre to any polygon edge;
    # if that distance is within the zone's radius the shapes touch or overlap.
    return _min_distance_to_polygon(latitude, longitude, polygon_points) <= radius


def _point_in_polygon(lat: float, lon: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False

    inside = False
    x = lon
    y = lat

    for i in range(len(polygon)):
        y1, x1 = polygon[i]
        y2, x2 = polygon[(i + 1) % len(polygon)]

        if ((x1 > x) != (x2 > x)):
            if x2 == x1:
                continue
            intersect_y = (y2 - y1) * (x - x1) / (x2 - x1) + y1
            if y < intersect_y:
                inside = not inside

    return inside


def _min_distance_to_polygon(
    lat: float, lon: float, polygon: list[tuple[float, float]]
) -> float:
    if len(polygon) == 1:
        single_lat, single_lon = polygon[0]
        return _distance_m(lat, lon, single_lat, single_lon)

    # Walk each edge, measuring the distance from the zone centre to that segment,
    # and take the minimum. If any segment is closer than the zone radius, the
    # outage geometry intersects the circle.
    min_distance = math.inf
    for index in range(len(polygon)):
        lat1, lon1 = polygon[index]
        lat2, lon2 = polygon[(index + 1) % len(polygon)]
        distance = _distance_point_to_segment_m(lat, lon, lat1, lon1, lat2, lon2)
        if distance < min_distance:
            min_distance = distance

    return min_distance


def _distance_point_to_segment_m(
    lat: float, lon: float, lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    origin_x, origin_y = 0.0, 0.0
    x1, y1 = _project_to_meters(lat1, lon1, lat, lon)
    x2, y2 = _project_to_meters(lat2, lon2, lat, lon)
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.hypot(x1 - origin_x, y1 - origin_y)

    # Project the zone centre onto the line segment to find the closest point,
    # clamping the projection to stay within the segment endpoints.
    t = ((origin_x - x1) * dx + (origin_y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(closest_x - origin_x, closest_y - origin_y)


def _project_to_meters(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    ref_lat_rad = math.radians(ref_lat)
    lon_rad = math.radians(lon)
    ref_lon_rad = math.radians(ref_lon)

    x = (lon_rad - ref_lon_rad) * math.cos((lat_rad + ref_lat_rad) / 2) * EARTH_RADIUS_M
    y = (lat_rad - ref_lat_rad) * EARTH_RADIUS_M
    return x, y


def _circle_contains_point(
    zone: dict[str, Any], point: tuple[float, float]
) -> bool:
    latitude = zone.get("latitude")
    longitude = zone.get("longitude")
    radius = float(zone.get("radius", DEFAULT_ZONE_RADIUS_M) or DEFAULT_ZONE_RADIUS_M)

    if latitude is None or longitude is None:
        return False

    if radius <= 0:
        radius = DEFAULT_ZONE_RADIUS_M

    # Treat the centroid as a simple point and measure its spherical distance to
    # the zone's centre. If the point lies inside the circle, we consider the
    # outage relevant to the zone.
    point_lat, point_lon = point
    return _distance_m(latitude, longitude, point_lat, point_lon) <= radius


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    a = min(1.0, max(0.0, a))
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _resolve_zones(hass: HomeAssistant) -> dict[str, dict[str, Any]]:
    zones: dict[str, dict[str, Any]] = {}
    if not hasattr(hass, "states"):
        return zones

    zone_states: list[Any]
    if hasattr(hass.states, "async_all"):
        zone_states = hass.states.async_all("zone")  # type: ignore[attr-defined]
    else:
        zone_states = [
            state
            for state in getattr(hass.states, "all", lambda: [])()
            if getattr(state, "entity_id", "").startswith("zone.")
        ]

    for state in zone_states:
        attributes = getattr(state, "attributes", {}) or {}
        latitude = attributes.get("latitude")
        longitude = attributes.get("longitude")
        radius = attributes.get("radius", DEFAULT_ZONE_RADIUS_M)

        if latitude is None or longitude is None:
            continue

        try:
            latitude_f = float(latitude)
            longitude_f = float(longitude)
        except (TypeError, ValueError):
            continue

        try:
            radius_f = float(radius) if radius is not None else DEFAULT_ZONE_RADIUS_M
        except (TypeError, ValueError):
            radius_f = DEFAULT_ZONE_RADIUS_M

        if radius_f <= 0:
            radius_f = DEFAULT_ZONE_RADIUS_M

        zone_name = attributes.get("friendly_name") or getattr(state, "name", state.entity_id)
        zones[state.entity_id] = {
            "name": zone_name,
            "latitude": latitude_f,
            "longitude": longitude_f,
            "radius": radius_f,
        }

    if not zones:
        latitude = getattr(hass.config, "latitude", None)
        longitude = getattr(hass.config, "longitude", None)
        if latitude is None or longitude is None:
            return zones

        try:
            latitude_f = float(latitude)
            longitude_f = float(longitude)
        except (TypeError, ValueError):
            return zones

        zones["zone.home"] = {
            "name": "Home",
            "latitude": latitude_f,
            "longitude": longitude_f,
            "radius": DEFAULT_ZONE_RADIUS_M,
        }

    return zones


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = dt_util.parse_datetime(value)
    else:
        return None

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_util.DEFAULT_TIME_ZONE)
    return parsed


def _format_datetime(value: Any) -> str | None:
    parsed = _parse_datetime(value)
    if parsed is None:
        return None
    localized = dt_util.as_local(parsed)
    return localized.strftime("%Y-%m-%d %H:%M")


def _shutdown_sort_key(shutdown: dict[str, Any]) -> tuple[datetime, str]:
    parsed = _parse_datetime(shutdown.get("startDate"))
    if parsed is None:
        parsed = datetime.max.replace(tzinfo=dt_util.UTC)
    message = shutdown.get("message")
    return parsed, message or ""


def _format_shutdowns(shutdowns: list[dict[str, Any]]) -> str:
    sentences: list[str] = []
    for shutdown in shutdowns:
        if not isinstance(shutdown, dict):
            continue
        start_text = _format_datetime(shutdown.get("startDate")) or "Unknown start"
        end_text = _format_datetime(shutdown.get("endDate")) or "Unknown end"
        message = (shutdown.get("message") or "").strip()
        location = message if message else "unspecified location"
        sentences.append(f"{start_text} - {end_text}: {location}.")
    return " ".join(sentences)


def _select_earliest_shutdown(shutdowns: list[dict[str, Any]]) -> dict[str, Any] | None:
    for shutdown in shutdowns:
        if _parse_datetime(shutdown.get("startDate")) is not None:
            return shutdown
    return shutdowns[0] if shutdowns else None


class PowerOutageSensor(CoordinatorEntity, SensorEntity):
    def __init__(self, coordinator: PowerOutageCoordinator, zone_id: str) -> None:
        super().__init__(coordinator)
        self._zone_id = zone_id
        zone = coordinator.zones.get(zone_id, {})
        zone_name = zone.get("name") or zone_id
        self._attr_unique_id = f"power_outages_{slugify(zone_id)}"
        self._initial_zone_name = str(zone_name)

    @property
    def name(self) -> str:
        zone = self.coordinator.zones.get(self._zone_id, {})
        zone_name = zone.get("name") or self._initial_zone_name
        return f"Power Outages ({zone_name})"

    @property
    def native_value(self):
        """Return number of active/planned outages."""
        data = self.coordinator.data
        if not isinstance(data, dict):
            return 0
        shutdowns = data.get(self._zone_id, [])
        return len(shutdowns)

    @property
    def extra_state_attributes(self):
        """Provide details of each outage."""
        zone = self.coordinator.zones.get(self._zone_id, {})
        data = self.coordinator.data if isinstance(self.coordinator.data, dict) else {}
        shutdowns_raw = data.get(self._zone_id, []) if isinstance(data, dict) else []
        if not isinstance(shutdowns_raw, list):
            shutdowns_raw = []

        shutdown_dicts = [shutdown for shutdown in shutdowns_raw if isinstance(shutdown, dict)]
        shutdowns_sorted = sorted(shutdown_dicts, key=_shutdown_sort_key)
        formatted_shutdowns = _format_shutdowns(shutdowns_sorted)
        earliest_shutdown = _select_earliest_shutdown(shutdowns_sorted)

        start_date = earliest_shutdown.get("startDate") if earliest_shutdown else None
        end_date = earliest_shutdown.get("endDate") if earliest_shutdown else None

        return {
            "zone": zone.get("name") or self._initial_zone_name,
            "zone_entity_id": self._zone_id,
            "latitude": zone.get("latitude"),
            "longitude": zone.get("longitude"),
            "radius": zone.get("radius"),
            "shutdowns": formatted_shutdowns,
            "startDate": start_date,
            "endDate": end_date,
        }
