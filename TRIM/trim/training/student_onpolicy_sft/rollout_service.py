"""Persistent Zero rollout workers for Student on-policy SFT.

Training ranks talk to this service. They do not import the Full Harness.
A worker ACKs the adapter it actually loaded; a path change alone is not an ACK.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

from trim.training.sft_ablation.io_util import atomic_write_json, read_json, sha256_tree


def load_endpoints(path: Path | str) -> dict[str, Any]:
    payload = read_json(Path(path))
    workers = payload.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError("rollout endpoints require a non-empty workers list")
    seen: set[str] = set()
    for worker in workers:
        for key in ("worker_id", "host", "port", "cuda_visible_devices"):
            if key not in worker:
                raise ValueError(f"rollout endpoint missing {key}")
        if worker["worker_id"] in seen:
            raise ValueError(f"duplicate rollout worker_id {worker['worker_id']}")
        seen.add(str(worker["worker_id"]))
    return payload


def assert_endpoints_disjoint(
    path: Path | str,
    *,
    train_visible: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    payload = load_endpoints(path)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") if train_visible is None else train_visible
    host = hostname or socket.gethostname()
    mine = {part.strip() for part in str(visible).split(",") if part.strip()}
    if not mine:
        return payload
    for worker in payload["workers"]:
        if not _same_machine(worker, host):
            continue
        theirs = {part.strip() for part in str(worker["cuda_visible_devices"]).split(",") if part.strip()}
        overlap = mine & theirs
        if overlap:
            raise SystemExit(f"CE GPUs overlap rollout GPUs {sorted(overlap)} on {host}")
    return payload


def _same_machine(worker: dict[str, Any], host: str) -> bool:
    declared = str(worker.get("hostname") or "")
    if not declared:
        return True
    return declared in {host, "localhost", "127.0.0.1"}


class RolloutPool:
    """Blocking client. COLLECT is allowed only after every worker ACKs the load."""

    def __init__(self, path: Path | str, *, timeout_s: float = 3600) -> None:
        self.endpoints = load_endpoints(path)["workers"]
        self.timeout_s = float(timeout_s)

    def load_all(self, version: str, adapter_path: str, adapter_digest: str) -> list[dict[str, Any]]:
        acks = []
        for worker in self.endpoints:
            ack = self._call(
                worker,
                {
                    "op": "LOAD_POLICY",
                    "version": version,
                    "adapter_path": adapter_path,
                    "adapter_digest": adapter_digest,
                },
            )
            if ack.get("loaded_version") != version or ack.get("loaded_digest") != adapter_digest:
                raise RuntimeError(f"worker {worker['worker_id']} ACK does not match the published policy: {ack}")
            acks.append(ack)
        return acks

    def collect_all(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        shards = []
        queries = list(request.get("queries") or [])
        width = max(1, len(self.endpoints))
        for index, worker in enumerate(self.endpoints):
            owned = queries[index::width]
            response = self._call(
                worker,
                {
                    **request,
                    "op": "COLLECT",
                    "queries": owned,
                    "worker_id": worker["worker_id"],
                },
            )
            if response.get("op") != "SHARD_READY":
                raise RuntimeError(f"worker {worker['worker_id']} did not return SHARD_READY")
            shards.append(response)
        return shards

    def quiesce_all(self, round_id: int) -> None:
        for worker in self.endpoints:
            ack = self._call(worker, {"op": "QUIESCE", "round_id": int(round_id)})
            if int(ack.get("inflight") or 0) != 0:
                raise RuntimeError(f"worker {worker['worker_id']} still has inflight rollout work")

    def shutdown_all(self) -> None:
        for worker in self.endpoints:
            self._call(worker, {"op": "SHUTDOWN"})

    def _call(self, worker: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        with socket.create_connection((str(worker["host"]), int(worker["port"])), timeout=self.timeout_s) as sock:
            sock.settimeout(self.timeout_s)
            _send(sock, payload)
            response = _recv(sock)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error") or f"rollout call failed: {response}")
        return response


class VLLMRolloutSession:
    """One base model, one live adapter. A new policy rebuilds the client."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.client: Any = None
        self.loaded: dict[str, str] | None = None

    def load_policy(self, version: str, adapter_path: str, adapter_digest: str) -> dict[str, Any]:
        path = Path(adapter_path)
        if not path.exists():
            raise RuntimeError(f"adapter path does not exist: {path}")
        actual = sha256_tree(path)
        if actual != adapter_digest:
            raise RuntimeError(f"adapter digest mismatch: {actual} != {adapter_digest}")
        self.close()
        from trim.training.vllm_hybrid import VLLMGenerateClient

        client = VLLMGenerateClient(
            model_path=str(self.config["model_name"]),
            session_dir=Path(self.config.get("session_dir") or "rollout_session"),
            tensor_parallel_size=int(self.config.get("tensor_parallel_size") or 1),
            max_model_len=int(self.config["max_model_len"]),
            lora_path=str(path),
        )
        client.start()
        self.client = client
        self.loaded = {"version": version, "digest": actual, "path": str(path)}
        return {
            "loaded_version": version,
            "loaded_digest": actual,
            "probe_result": "client_started",
        }

    def collect(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.client is None or self.loaded is None:
            raise RuntimeError("COLLECT before a successful LOAD_POLICY ACK")
        if self.loaded["version"] != request.get("expected_version") or self.loaded["digest"] != request.get("expected_digest"):
            raise RuntimeError("refusing COLLECT for a policy this worker did not load")
        from trim.training.student_onpolicy_sft.runtime import collect_student_points

        job = {
            **self.config,
            **request,
            "adapter": self.loaded["path"],
            "adapter_digest": self.loaded["digest"],
            "policy_version": str(request.get("policy_version") or request["expected_version"]),
            "seed": int(self.config.get("seed") or 0) + int(request.get("round_id") or 0),
        }
        points = collect_student_points(self.client, job, self.loaded["digest"])
        shard_dir = str(request.get("shard_dir") or "")
        shard_path = ""
        if shard_dir:
            directory = Path(shard_dir)
            directory.mkdir(parents=True, exist_ok=True)
            shard_path = str(directory / f"{request.get('worker_id') or 'worker'}.json")
            atomic_write_json(Path(shard_path), {"points": points, "loaded_digest": self.loaded["digest"]})
            atomic_write_json(Path(shard_path + ".READY"), {"ok": True, "n_points": len(points)})
        return {
            "n_points": len(points),
            "points": points,
            "loaded_digest": self.loaded["digest"],
            "shard_path": shard_path,
        }

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.loaded = None


def serve(host: str, port: int, session: Any, worker_id: str) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(8)
    try:
        while True:
            connection, _addr = server.accept()
            with connection:
                try:
                    request = _recv(connection)
                except (ConnectionError, json.JSONDecodeError):
                    continue
                try:
                    response = _dispatch(session, worker_id, request)
                except Exception as exc:
                    response = {"op": "ACK", "ok": False, "error": f"{type(exc).__name__}: {exc}", "worker_id": worker_id}
                _send(connection, response)
                if request.get("op") == "SHUTDOWN":
                    break
    finally:
        server.close()
        session.close()


def _dispatch(session: Any, worker_id: str, request: dict[str, Any]) -> dict[str, Any]:
    op = str(request.get("op") or "")
    if op == "LOAD_POLICY":
        ack = session.load_policy(str(request["version"]), str(request["adapter_path"]), str(request["adapter_digest"]))
        return {"op": "ACK", "ok": True, "worker_id": worker_id, **ack}
    if op == "COLLECT":
        shard = session.collect(request)
        return {"op": "SHARD_READY", "ok": True, "worker_id": worker_id, **shard}
    if op == "QUIESCE":
        inflight = int(getattr(session, "inflight", 0) or 0)
        return {"op": "ACK", "ok": True, "worker_id": worker_id, "inflight": inflight, "round_id": request.get("round_id")}
    if op == "SHUTDOWN":
        session.close()
        return {"op": "ACK", "ok": True, "worker_id": worker_id, "shutdown": True}
    raise ValueError(f"unknown rollout op {op}")


def _send(sock: socket.socket, payload: dict[str, Any]) -> None:
    blob = (json.dumps(payload) + "\n").encode("utf-8")
    sock.sendall(blob)


def _recv(sock: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        if chunks and b"\n" in chunks[-1]:
            break
        piece = sock.recv(1024 * 1024)
        if not piece:
            break
        chunks.append(piece)
        if b"\n" in piece:
            break
    blob = b"".join(chunks)
    if b"\n" not in blob:
        raise ConnectionError("rollout worker closed the connection")
    line = blob.split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="student_rollout_service", allow_abbrev=False)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--train-env", default="upstream")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--session-dir", default="")
    args = parser.parse_args(argv)
    session_dir = args.session_dir or str(Path("rollout_sessions") / args.worker_id)
    session = VLLMRolloutSession(
        {
            "model_name": args.model_name,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "train_env": args.train_env,
            "group_size": args.group_size,
            "max_turns": args.max_turns,
            "seed": args.seed,
            "session_dir": session_dir,
        }
    )
    serve(args.host, int(args.port), session, args.worker_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
