import pytest

from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.report import summarize_output_throughput


def scenario_result(
    scenario_id: str,
    *,
    actual_start_s: float | None,
    completed_s: float | None,
    audio_duration_s: float = 0.0,
    completion_tokens: int | None = None,
    success: bool = True,
    status: str = "ok",
    category: str = "traffic",
    cancelled: bool = False,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id,
        endpoint="speech",
        category=category,
        stage_id="stage-1",
        success=success,
        status=status,
        actual_start_s=actual_start_s,
        completed_s=completed_s,
        audio_duration_s=audio_duration_s,
        completion_tokens=completion_tokens,
        was_cancelled=cancelled,
    )


def test_output_throughput_uses_all_traffic_observation_window() -> None:
    results = [
        scenario_result(
            "audio-1",
            actual_start_s=0.0,
            completed_s=3.0,
            audio_duration_s=6.0,
            completion_tokens=30,
        ),
        scenario_result(
            "audio-2",
            actual_start_s=1.0,
            completed_s=5.0,
            audio_duration_s=4.0,
            completion_tokens=20,
        ),
        scenario_result(
            "late-failure", actual_start_s=2.0, completed_s=10.0, success=False
        ),
        scenario_result(
            "capability-probe",
            actual_start_s=-100.0,
            completed_s=100.0,
            audio_duration_s=100.0,
            completion_tokens=100,
            category="capability_probe",
        ),
    ]

    summary = summarize_output_throughput(results)

    assert summary == {
        "observation_wall_time_s": 10.0,
        "audio_throughput_s_per_s": 1.0,
        "completion_tokens_per_s": 5.0,
        "audio_duration_sample_count": 2,
        "completion_token_sample_count": 2,
    }


def test_output_throughput_is_null_without_valid_window_or_token_samples() -> None:
    missing_window = summarize_output_throughput(
        [
            scenario_result(
                "missing",
                actual_start_s=0.0,
                completed_s=None,
                audio_duration_s=4.0,
                completion_tokens=8,
            )
        ]
    )
    assert missing_window == {
        "observation_wall_time_s": None,
        "audio_throughput_s_per_s": None,
        "completion_tokens_per_s": None,
        "audio_duration_sample_count": 1,
        "completion_token_sample_count": 1,
    }

    nonpositive_window = summarize_output_throughput(
        [
            scenario_result(
                "instant",
                actual_start_s=2.0,
                completed_s=2.0,
                audio_duration_s=4.0,
                completion_tokens=8,
            )
        ]
    )
    assert nonpositive_window["observation_wall_time_s"] is None
    assert nonpositive_window["audio_throughput_s_per_s"] is None

    missing_tokens = summarize_output_throughput(
        [
            scenario_result(
                "audio", actual_start_s=0.0, completed_s=2.0, audio_duration_s=4.0
            )
        ]
    )
    assert missing_tokens["observation_wall_time_s"] == 2.0
    assert missing_tokens["audio_throughput_s_per_s"] == 2.0
    assert missing_tokens["completion_tokens_per_s"] is None
    assert missing_tokens["completion_token_sample_count"] == 0


def test_output_throughput_counts_only_successful_non_cancelled_audio() -> None:
    results = [
        scenario_result(
            "expected-error",
            actual_start_s=0.0,
            completed_s=4.0,
            audio_duration_s=20.0,
            success=False,
            status="expected_error",
        ),
        scenario_result(
            "cancelled",
            actual_start_s=1.0,
            completed_s=5.0,
            audio_duration_s=20.0,
            cancelled=True,
        ),
        scenario_result(
            "success",
            actual_start_s=2.0,
            completed_s=6.0,
            audio_duration_s=6.0,
            completion_tokens=12,
        ),
    ]

    summary = summarize_output_throughput(results)

    assert summary["observation_wall_time_s"] == 6.0
    assert summary["audio_throughput_s_per_s"] == 1.0
    assert summary["completion_tokens_per_s"] == 2.0
    assert summary["audio_duration_sample_count"] == 1
    assert summary["completion_token_sample_count"] == 1


def test_output_throughput_handles_zero_timestamps_no_audio_and_all_failures() -> None:
    zero_timestamp = summarize_output_throughput(
        [
            scenario_result(
                "zero",
                actual_start_s=0.0,
                completed_s=4.0,
                audio_duration_s=2.0,
                completion_tokens=3,
            )
        ]
    )
    assert zero_timestamp["observation_wall_time_s"] == 4.0
    assert zero_timestamp["audio_throughput_s_per_s"] == 0.5

    no_audio = summarize_output_throughput(
        [
            scenario_result(
                "empty-audio", actual_start_s=0.0, completed_s=4.0, completion_tokens=3
            )
        ]
    )
    assert no_audio["audio_throughput_s_per_s"] == 0.0
    assert no_audio["audio_duration_sample_count"] == 0

    all_failed = summarize_output_throughput(
        [
            scenario_result(
                "failure-1", actual_start_s=0.0, completed_s=2.0, success=False
            ),
            scenario_result(
                "failure-2", actual_start_s=1.0, completed_s=5.0, success=False
            ),
        ]
    )
    assert all_failed["observation_wall_time_s"] == 5.0
    assert all_failed["audio_throughput_s_per_s"] == 0.0
    assert all_failed["completion_tokens_per_s"] is None


@pytest.mark.parametrize(
    ("unobserved_start_s", "unobserved_completion_s"),
    [(None, 1.5), (1.0, None)],
)
def test_partial_traffic_timestamps_do_not_invent_throughput_denominator(
    unobserved_start_s: float | None, unobserved_completion_s: float | None
) -> None:
    summary = summarize_output_throughput(
        [
            scenario_result(
                "observed",
                actual_start_s=0.0,
                completed_s=3.0,
                audio_duration_s=4.0,
                completion_tokens=8,
            ),
            scenario_result(
                "unobserved",
                actual_start_s=unobserved_start_s,
                completed_s=unobserved_completion_s,
                audio_duration_s=10.0,
                completion_tokens=20,
            ),
        ]
    )

    assert summary["observation_wall_time_s"] is None
    assert summary["audio_throughput_s_per_s"] is None
    assert summary["completion_tokens_per_s"] is None
    assert summary["audio_duration_sample_count"] == 2
    assert summary["completion_token_sample_count"] == 2


def test_partial_token_usage_does_not_estimate_unreported_output() -> None:
    summary = summarize_output_throughput(
        [
            scenario_result(
                "reported",
                actual_start_s=0.0,
                completed_s=2.0,
                audio_duration_s=2.0,
                completion_tokens=20,
            ),
            scenario_result(
                "unreported",
                actual_start_s=1.0,
                completed_s=4.0,
                audio_duration_s=6.0,
            ),
        ]
    )

    assert summary["observation_wall_time_s"] == 4.0
    assert summary["audio_throughput_s_per_s"] == 2.0
    assert summary["completion_tokens_per_s"] == 5.0
    assert summary["audio_duration_sample_count"] == 2
    assert summary["completion_token_sample_count"] == 1
