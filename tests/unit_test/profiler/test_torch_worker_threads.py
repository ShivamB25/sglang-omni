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
