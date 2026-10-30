"""Warmup and HTTP requests must use the same production singleton."""
from concurrent.futures import ThreadPoolExecutor
import importlib
from types import SimpleNamespace


def test_warmup_uses_request_pipeline_before_optional_furniture(monkeypatch):
    from backend.app import main
    from src.furniture.image_to_3d import ImageTo3DGenerator

    production = importlib.import_module("app.floorplan_engine")
    calls = []

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(main.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(production.FloorplanPipeline, "get_instance", lambda: calls.append("floorplan"))
    monkeypatch.setattr(ImageTo3DGenerator, "get_instance", lambda: (calls.append("furniture"), SimpleNamespace(device_name="test"))[1])
    main.preload_models()
    assert calls == ["floorplan", "furniture"]


def test_simultaneous_warmup_and_request_construct_one_pipeline(monkeypatch):
    from backend.app.floorplan_engine import FloorplanPipeline

    constructions = []
    monkeypatch.setattr(FloorplanPipeline, "_instance", None)
    monkeypatch.setattr(FloorplanPipeline, "__init__", lambda self: constructions.append(self))
    with ThreadPoolExecutor(max_workers=8) as executor:
        instances = list(executor.map(lambda _: FloorplanPipeline.get_instance(), range(16)))
    assert len(constructions) == 1
    assert all(instance is constructions[0] for instance in instances)
