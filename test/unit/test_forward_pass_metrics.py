import types
import unittest
from unittest.mock import patch

from sglang.srt.observability.scheduler_metrics_mixin import (
    PrefillStats,
    SchedulerMetricsMixin,
)


class _FakeReq:
    def __init__(self, prompt_len: int, output_len: int = 0):
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = list(range(output_len))
        self.seqlen = prompt_len + output_len


class _FakeForwardMode:
    def __init__(self, *, is_mixed: bool = False, is_extend: bool = False):
        self._is_mixed = is_mixed
        self._is_extend = is_extend

    def is_mixed(self):
        return self._is_mixed

    def is_extend(self, include_draft_extend_v2: bool = False):
        return self._is_extend

    def is_decode(self):
        return not self._is_mixed and not self._is_extend


class _CollectingPublisher:
    def __init__(self):
        self.metrics = []

    def publish(self, metrics):
        self.metrics.append(metrics)


class _DummyPublisherThread:
    def __init__(self, endpoint: str, worker_id: str, dp_rank: int, **_: object):
        self.endpoint = endpoint
        self.worker_id = worker_id
        self.dp_rank = dp_rank

    def shutdown(self):
        pass


class _DummyScheduler(SchedulerMetricsMixin):
    pass


class TestForwardPassMetrics(unittest.TestCase):
    def setUp(self):
        self.scheduler = _DummyScheduler()
        self.scheduler.enable_fpm = True
        self.scheduler._fpm_worker_id = "worker-7"
        self.scheduler._fpm_dp_rank = 0
        self.scheduler._fpm_publisher = _CollectingPublisher()
        self.scheduler.waiting_queue = []

    def _make_batch(self, **overrides):
        defaults = dict(
            forward_mode=_FakeForwardMode(),
            reqs=[],
            decoding_reqs=[],
            prefill_stats=None,
            seq_lens_cpu=[],
            fpm_start_time=100.0,
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_emit_mixed_batch_separates_prefill_and_decode(self):
        self.scheduler._fpm_dp_rank = 3
        self.scheduler.waiting_queue = [_FakeReq(6), _FakeReq(4, output_len=2)]

        prefill_a = _FakeReq(10)
        prefill_b = _FakeReq(14)
        decode_req = _FakeReq(8, output_len=3)
        batch = self._make_batch(
            forward_mode=_FakeForwardMode(is_mixed=True, is_extend=True),
            reqs=[prefill_a, prefill_b, decode_req],
            decoding_reqs=[decode_req],
            prefill_stats=PrefillStats(
                log_input_tokens=12,
                log_hit_tokens=5,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=2,
            ),
            seq_lens_cpu=[decode_req.seqlen],
        )

        with patch(
            "sglang.srt.observability.scheduler_metrics_mixin.time.monotonic",
            return_value=104.5,
        ):
            self.scheduler._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.worker_id, "worker-7")
        self.assertEqual(metrics.dp_rank, 3)
        self.assertEqual(metrics.wall_time, 4.5)
        self.assertEqual(metrics.scheduled_requests.num_prefill_requests, 2)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_tokens, 12)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_kv_tokens, 5)
        self.assertEqual(metrics.scheduled_requests.num_decode_requests, 1)
        self.assertEqual(
            metrics.scheduled_requests.sum_decode_kv_tokens, decode_req.seqlen
        )
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 1)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 1)

    def test_emit_uses_cuda_events_when_available(self):
        batch = self._make_batch()
        mock_start = types.SimpleNamespace(elapsed_time=lambda end: 42.0)
        mock_end = types.SimpleNamespace(query=lambda: True)
        result = types.SimpleNamespace(
            fpm_start_event=mock_start,
            fpm_end_event=mock_end,
        )

        self.scheduler._emit_forward_pass_metrics(batch, result)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertAlmostEqual(
            self.scheduler._fpm_publisher.metrics[0].wall_time, 0.042, places=4
        )

    def test_emit_skips_when_cuda_events_not_ready(self):
        batch = self._make_batch()
        mock_start = types.SimpleNamespace(elapsed_time=lambda end: 42.0)
        mock_end = types.SimpleNamespace(query=lambda: False)
        result = types.SimpleNamespace(
            fpm_start_event=mock_start,
            fpm_end_event=mock_end,
        )

        self.scheduler._emit_forward_pass_metrics(batch, result)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 0)

    def test_emit_falls_back_to_monotonic_without_events(self):
        batch = self._make_batch()

        with patch(
            "sglang.srt.observability.scheduler_metrics_mixin.time.monotonic",
            return_value=100.035,
        ):
            self.scheduler._emit_forward_pass_metrics(batch, result=None)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertAlmostEqual(
            self.scheduler._fpm_publisher.metrics[0].wall_time, 0.035, places=4
        )

    def test_init_metrics_uses_server_worker_id(self):
        scheduler = _DummyScheduler()
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            kv_events_config=None,
        )
        scheduler.attn_tp_rank = 0
        scheduler.dp_rank = 2
        scheduler.pp_rank = 0
        scheduler.pp_size = 1
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            scheduler.init_metrics(tp_rank=0, pp_rank=0, dp_rank=2)

        self.assertTrue(scheduler.enable_fpm)
        self.assertEqual(scheduler._fpm_worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_dp_rank, 2)
        self.assertEqual(scheduler._fpm_publisher.worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_publisher.dp_rank, 2)
        self.assertTrue(scheduler._fpm_publisher.endpoint.startswith("ipc://"))
        self.assertIsNotNone(scheduler.server_args.forward_pass_metrics_ipc_name)

    def test_init_fpm_disabled_on_non_last_pp_rank(self):
        scheduler = _DummyScheduler()
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            kv_events_config=None,
        )
        scheduler.attn_tp_rank = 0
        scheduler.dp_rank = 0
        scheduler.pp_rank = 0
        scheduler.pp_size = 2
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            scheduler.init_metrics(tp_rank=0, pp_rank=0, dp_rank=0)

        self.assertFalse(scheduler.enable_fpm)


if __name__ == "__main__":
    unittest.main()
