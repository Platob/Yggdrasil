from __future__ import annotations

import logging
import time
from threading import Lock

from yggdrasil.dataclasses.expiring import ExpiringDict

from ..config import Settings
from ..remote import list_registered
from ..schemas.discovery import HelloRequest, HelloResponse, NodeInfo, PeerListResponse

LOGGER = logging.getLogger(__name__)

_PEER_TTL = 300  # seconds


class _Peer:
    __slots__ = ("node_id", "host", "port", "version", "lat", "lon", "last_seen")

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        version: str,
        lat: float | None = None,
        lon: float | None = None,
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.version = version
        self.lat = lat
        self.lon = lon
        self.last_seen: float = time.monotonic()

    def touch(self) -> None:
        self.last_seen = time.monotonic()

    def to_node_info(self) -> NodeInfo:
        return NodeInfo(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            version=self.version,
            lat=self.lat,
            lon=self.lon,
        )


class DiscoveryService:
    def __init__(self, settings: Settings, messenger_service=None) -> None:
        self.settings = settings
        self._start_time: float = time.monotonic()
        self._peers: dict[str, _Peer] = {}
        self._lock = Lock()
        self._messenger_service = messenger_service
        # Cache for geolocation (5s TTL in nanoseconds)
        self._geo_cache: ExpiringDict[str, tuple[float | None, float | None]] = ExpiringDict(default_ttl=5_000_000_000)

    async def hello(self, req: HelloRequest) -> HelloResponse:
        with self._lock:
            self._purge_stale_peers()
            existing = req.node_id in self._peers
            peer = self._peers.get(req.node_id)
            if peer is None:
                peer = _Peer(
                    node_id=req.node_id,
                    host=req.host,
                    port=req.port,
                    version=req.version,
                    lat=req.lat,
                    lon=req.lon,
                )
                self._peers[req.node_id] = peer
            else:
                peer.host = req.host
                peer.port = req.port
                peer.version = req.version
                peer.lat = req.lat
                peer.lon = req.lon
                peer.touch()
            peers = [p.to_node_info() for p in self._peers.values()]

        if not existing:
            LOGGER.info("Registered new peer %r (%s:%d)", req.node_id, req.host, req.port)

        lat, lon = self._get_cached_location()

        return HelloResponse(
            node_id=self.settings.node_id,
            host=self.settings.host,
            port=self.settings.port,
            version=self.settings.app_version,
            lat=lat,
            lon=lon,
            peers=peers,
        )

    async def get_peers(self) -> PeerListResponse:
        with self._lock:
            self._purge_stale_peers()
            peers = [p.to_node_info() for p in self._peers.values()]
        return PeerListResponse(node_id=self.settings.node_id, peers=peers)

    async def get_self_info(self) -> NodeInfo:
        uptime = time.monotonic() - self._start_time
        channels = self._get_channels()
        functions = sorted(list_registered().keys())
        lat, lon = self._get_cached_location()
        return NodeInfo(
            node_id=self.settings.node_id,
            host=self.settings.host,
            port=self.settings.port,
            version=self.settings.app_version,
            uptime=uptime,
            channels=channels,
            functions=functions,
            lat=lat,
            lon=lon,
        )

    def _purge_stale_peers(self) -> None:
        """Remove peers not seen in the last ``_PEER_TTL`` seconds.

        Must be called while holding ``self._lock``.
        """
        now = time.monotonic()
        stale: list[str] = []
        for node_id, peer in self._peers.items():
            if now - peer.last_seen > _PEER_TTL:
                stale.append(node_id)
        for node_id in stale:
            del self._peers[node_id]
            LOGGER.debug("Purged stale peer %r", node_id)

    def _get_cached_location(self) -> tuple[float | None, float | None]:
        """Return cached geolocation (lat, lon) with 5s TTL."""
        cached = self._geo_cache.get("_geo")
        if cached is not None:
            return cached
        from ..geo import get_location
        loc = get_location()
        self._geo_cache.set("_geo", loc, ttl=5_000_000_000)
        return loc

    def _get_channels(self) -> list[str]:
        """Return channel names from the messenger service if accessible."""
        if self._messenger_service is None:
            return []
        try:
            with self._messenger_service._lock:
                return sorted(self._messenger_service._channels.keys())
        except Exception:
            return []
