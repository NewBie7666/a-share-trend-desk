from src.runtime_profiler import RuntimeProfiler, add_runtime_diagnostics


def test_total_stage_is_not_added_to_child_stages():
    profiler = RuntimeProfiler()
    profiler._stages = {
        "provider_fetch": {"stage_name": "provider_fetch", "elapsed_seconds": 98, "start_time": "", "end_time": ""},
        "total": {"stage_name": "total", "elapsed_seconds": 100, "start_time": "", "end_time": ""},
    }
    report = add_runtime_diagnostics(profiler.get_runtime_report(), {"provider_fetch": 120})
    assert report["total_seconds"] == 100
    assert report["provider_fetch_ratio"] == 0.98
    assert report["provider_bottleneck"] is True
