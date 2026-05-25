from __future__ import annotations

import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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

        from ..geo import get_location
        lat, lon = get_location()

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
        from ..geo import get_location

        uptime = time.monotonic() - self._start_time
        channels = self._get_channels()
        functions = sorted(list_registered().keys())
        lat, lon = get_location()
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

    async def discover_friends(self, targets: list[str]) -> PeerListResponse:
        if not targets:
            return PeerListResponse(node_id=self.settings.node_id, peers=[])

        from ..geo import get_location

        discovered: list[NodeInfo] = []
        lat, lon = get_location()
        payload = json.dumps({
            "node_id": self.settings.node_id,
            "host": self.settings.host,
            "port": self.settings.port,
            "version": self.settings.app_version,
            "lat": lat,
            "lon": lon,
        }).encode()

        def _contact(url: str) -> NodeInfo | None:
            hello_url = url.rstrip("/") + "/api/hello"
            req = urllib.request.Request(
                hello_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                return NodeInfo(
                    node_id=data["node_id"],
                    host=data["host"],
                    port=data["port"],
                    version=data["version"],
                    lat=data.get("lat"),
                    lon=data.get("lon"),
                )
            except Exception as exc:
                LOGGER.debug("Failed to contact %r: %s", url, exc)
                return None

        with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as pool:
            futures = {pool.submit(_contact, t): t for t in targets}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    discovered.append(result)
                    # Register the discovered node as a peer.
                    with self._lock:
                        peer = self._peers.get(result.node_id)
                        if peer is None:
                            peer = _Peer(
                                node_id=result.node_id,
                                host=result.host,
                                port=result.port,
                                version=result.version,
                                lat=result.lat,
                                lon=result.lon,
                            )
                            self._peers[result.node_id] = peer
                            LOGGER.info(
                                "Discovered peer %r (%s:%d)",
                                result.node_id, result.host, result.port,
                            )
                        else:
                            peer.lat = result.lat
                            peer.lon = result.lon
                            peer.touch()

        return PeerListResponse(node_id=self.settings.node_id, peers=discovered)

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

    def _get_channels(self) -> list[str]:
        """Return channel names from the messenger service if accessible."""
        if self._messenger_service is None:
            return []
        try:
            with self._messenger_service._lock:
                return sorted(self._messenger_service._channels.keys())
        except Exception:
            return []
