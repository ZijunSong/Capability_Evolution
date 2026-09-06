from __future__ import annotations

import asyncio

import torch

from trim.training.hf_rl_opd_client import HFDebugTrainingClient


class _TinyBackend:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.ones(1))
        self.optimizer = torch.optim.SGD([self.param], lr=0.0)
        self.model = torch.nn.Linear(1, 1)
        self._device = torch.device("cpu")
        self.n_single = 0
        self.n_batch = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def _teacher_forced_logprobs(self, prompt_ids, response_ids, *, require_grad: bool):
        self.n_single += 1
        n = max(1, len(response_ids))
        vals = self.param.expand(n)
        return vals if require_grad else vals.detach()

    def _teacher_forced_logprobs_batch(self, pairs, *, require_grad: bool):
        self.n_batch += 1
        return [
            self._teacher_forced_logprobs(p, r, require_grad=require_grad) for p, r in pairs
        ]


def _rows(n: int) -> list[dict]:
    return [
        {
            "prompt_ids": [1, 2, 3, 4],
            "action_ids": [5, 6],
            "logprob_old": 0.0,
            "advantage": 1.0 if i % 2 == 0 else -0.5,
        }
        for i in range(n)
    ]


def test_cispo_microbatch_uses_batched_forward():
    backend = _TinyBackend()
    client = HFDebugTrainingClient(backend, micro_batch_size=4, heartbeat_every=1)
    out = client._cispo_backward(_rows(8))
    assert out["n_datums"] == 8
    assert out["n_microbatches"] >= 1
    assert backend.n_batch >= 1
    assert backend.n_single == 8  # batch helper still calls per-row fake logprobs
    assert backend.param.grad is not None


def test_cispo_serial_microbatch_still_trains():
    backend = _TinyBackend()
    client = HFDebugTrainingClient(backend, micro_batch_size=1, heartbeat_every=100)
    out = client._cispo_backward(_rows(3))
    assert out["n_datums"] == 3
    assert out["n_microbatches"] == 3
    assert backend.n_batch == 0
    assert backend.n_single == 3


def test_forward_backward_async_logs_and_counts():
    backend = _TinyBackend()
    client = HFDebugTrainingClient(backend, micro_batch_size=2, heartbeat_every=1)
    payload = asyncio.run(client.forward_backward_async(_rows(4), "cispo"))
    assert payload["n_datums"] == 4
    assert client.calls == [("fb", "cispo", 4)]


def test_cli_exposes_new_hf_train_settings(tmp_path):
    from trim.cli.launch import parse_train_args

    args, _spec = parse_train_args(
        [
            "--train_method",
            "trim",
            "--component",
            "zero",
            "--out",
            str(tmp_path / "out"),
            "--validate-only",
        ]
    )
    assert args.train_groups_per_step == 32
    assert args.train_micro_batch_size == 4
    args0, _ = parse_train_args(
        [
            "--train_method",
            "trim",
            "--component",
            "zero",
            "--out",
            str(tmp_path / "out0"),
            "--validate-only",
            "--train-groups-per-step",
            "0",
            "--train-micro-batch-size",
            "1",
        ]
    )
    assert args0.train_groups_per_step == 0
    assert args0.train_micro_batch_size == 1
