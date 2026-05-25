"""Tests for MonitorService and monitor endpoints."""
from __future__ import annotations

import pytest

from yggdrasil.node.config import Settings
from yggdrasil.node.services.monitor import MonitorService


class TestMonitorService:
    def setup_method(self):
        self.settings = Settings()
        self.service = MonitorService(self.settings, history_size=10)

    def test_snapshot_returns_resource_data(self):
        snap = self.service.snapshot()
        assert snap.timestamp != ""
        assert snap.cpu_percent >= 0
        assert snap.memory_total_mb > 0

    def test_snapshot_includes_network(self):
        snap = self.service.snapshot()
        assert snap.network.bytes_sent >= 0
        assert snap.network.bytes_recv >= 0

    def test_history_accumulates(self):
        for _ in range(5):
            self.service.snapshot()
            self.service._last_collect = 0  # force re-collect
        hist = self.service.history(10)
        assert len(hist) >= 1

    def test_history_limit(self):
        for _ in range(15):
            self.service.snapshot()
            self.service._last_collect = 0
        hist = self.service.history(5)
        assert len(hist) <= 10  # capped by deque maxlen


class TestMonitorEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        from httpx import ASGITransport, AsyncClient
        from yggdrasil.node.app import create_app
        self.app = create_app()
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://test",
        )

    @pytest.mark.asyncio
    async def test_get_monitor(self):
        resp = await self.client.get("/api/monitor")
        assert resp.status_code == 200
        data = resp.json()
        assert "snapshot" in data
        assert "history" in data
        assert data["snapshot"]["cpu_percent"] >= 0
