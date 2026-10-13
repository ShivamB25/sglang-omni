# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from queue import Queue
from threading import Event, Thread

import pytest
import torch
from torch.profiler import ProfilerActivity

from sglang_omni.profiler import torch_profiler


def test_existing_worker_operations_include_shapes_and_source_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work: Queue[bool] = Queue()
    results: Queue[float] = Queue()
    ready = Event()

    def worker() -> None:
        matrix = torch.ones(16, 16)
        ready.set()
        while work.get():
            results.put(torch.mm(matrix, matrix)[0, 0].item())

    thread = Thread(target=worker)
    thread.start()
    try:
        assert ready.wait(timeout=10)
        monkeypatch.setenv("SGLANG_TORCH_PROFILER_PROFILE_MEMORY", "0")
        monkeypatch.setenv("SGLANG_TORCH_PROFILER_RECORD_SHAPES", "1")
        monkeypatch.setenv("SGLANG_TORCH_PROFILER_WITH_STACK", "1")
        monkeypatch.setattr(
            torch_profiler, "profiler_activities", lambda: [ProfilerActivity.CPU]
        )
        controller = torch_profiler.TorchProfiler
        assert not controller.is_active()
        controller.start(str(tmp_path / "worker"), run_id="worker-thread")
        capture = controller.profiler
        try:
            work.put(True)
            assert results.get(timeout=10) == 16
        finally:
            controller.stop(run_id="worker-thread")
        work.put(True)
        assert results.get(timeout=10) == 16
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU], record_shapes=True
        ) as post_capture:
            work.put(True)
            assert results.get(timeout=10) == 16
            control_matrix = torch.ones(3, 3)
            assert torch.mm(control_matrix, control_matrix)[0, 0].item() == 3
        post_capture_shapes = [
            event.input_shapes
            for event in post_capture.events()
            if event.name == "aten::mm"
        ]
        assert post_capture_shapes == [[[3, 3], [3, 3]]]
        assert capture is not None
        matrix_operations = [
            event for event in capture.events() if event.name == "aten::mm"
        ]
        assert len(matrix_operations) == 1
        assert matrix_operations[0].input_shapes == [[16, 16], [16, 16]]
        assert any(Path(__file__).name in frame for frame in matrix_operations[0].stack)
    finally:
        work.put(False)
        thread.join(timeout=10)
        assert not thread.is_alive()


def test_repeated_start_preserves_recorded_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_RECORD_SHAPES", "1")
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_PROFILE_MEMORY", "0")
    monkeypatch.setattr(
        torch_profiler, "profiler_activities", lambda: [ProfilerActivity.CPU]
    )
    controller = torch_profiler.TorchProfiler
    assert not controller.is_active()
    first_path = controller.start(str(tmp_path / "shared"), run_id="shared")
    capture = controller.profiler
    try:
        first_matrix = torch.ones(2, 2)
        torch.mm(first_matrix, first_matrix)
        repeated_path = controller.start(str(tmp_path / "unused"), run_id="shared")
        second_matrix = torch.ones(3, 3)
        torch.mm(second_matrix, second_matrix)
        assert repeated_path == first_path
    finally:
        controller.stop(run_id="shared")
    assert capture is not None
    shapes = [
        event.input_shapes for event in capture.events() if event.name == "aten::mm"
    ]
    assert shapes == [[[2, 2], [2, 2]], [[3, 3], [3, 3]]]


def test_replacement_capture_ignores_stale_stop_and_separates_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_PROFILE_MEMORY", "0")
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_RECORD_SHAPES", "1")
    monkeypatch.setattr(
        torch_profiler, "profiler_activities", lambda: [ProfilerActivity.CPU]
    )
    controller = torch_profiler.TorchProfiler
    assert not controller.is_active()
    controller.start(str(tmp_path / "first"), run_id="first")
    first_capture = controller.profiler
    try:
        first_matrix = torch.ones(2, 2)
        assert torch.mm(first_matrix, first_matrix)[0, 0].item() == 2
        controller.start(str(tmp_path / "second"), run_id="second")
        second_capture = controller.profiler
        assert controller.stop(run_id="first") is None
        second_matrix = torch.ones(3, 3)
        assert torch.mm(second_matrix, second_matrix)[0, 0].item() == 3
    finally:
        controller.stop()
    assert first_capture is not None
    assert second_capture is not None
    first_shapes = [
        event.input_shapes
        for event in first_capture.events()
        if event.name == "aten::mm"
    ]
    second_shapes = [
        event.input_shapes
        for event in second_capture.events()
        if event.name == "aten::mm"
    ]
    assert first_shapes == [[[2, 2], [2, 2]]]
    assert second_shapes == [[[3, 3], [3, 3]]]


def test_memory_capture_preserves_cpu_allocation_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_PROFILE_MEMORY", "1")
    monkeypatch.setattr(
        torch_profiler, "profiler_activities", lambda: [ProfilerActivity.CPU]
    )
    controller = torch_profiler.TorchProfiler
    assert not controller.is_active()
    controller.start(str(tmp_path / "memory"), run_id="memory")
    capture = controller.profiler
    try:
        allocation = torch.empty(1_000_003, dtype=torch.float32)
        assert allocation.numel() == 1_000_003
    finally:
        controller.stop(run_id="memory")
    assert capture is not None
    allocation_bytes = [
        event.cpu_memory_usage
        for event in capture.events()
        if event.name == "aten::empty"
    ]
    assert allocation_bytes == [4_000_012]
