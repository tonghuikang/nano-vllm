"""OpenAI-compatible HTTP server for nano-vllm.

Single-process. The engine lives in a background driver thread that owns
LLMEngine.step(); HTTP handler threads enqueue requests and await
completion via concurrent.futures.Future. New requests flow through a
queue so HTTP threads never block on a step in progress, which lets the
running batch grow naturally as more requests arrive.

Implements just `/v1/completions` (and a stub `/v1/models`) — enough for
the bench harness in `bench/`. `/v1/chat/completions` is intentionally
out of scope.

Run with::

    uv run python -m nanovllm.server --model ~/huggingface/Qwen3-0.6B/

Accepted vLLM-extension fields on the request body: `min_tokens`
(ignored; nano-vllm has no min-length, but `ignore_eos=True` +
`max_tokens` already pin output length) and `ignore_eos`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class EngineService:
    def __init__(self, model: str, **kwargs) -> None:
        self.engine = LLMEngine(model, **kwargs)
        self.tokenizer = self.engine.tokenizer
        self.model_name = model
        self.incoming: Queue = Queue()
        self.futures: dict[int, Future] = {}
        self._stop = False
        self._driver = threading.Thread(target=self._loop, name="engine-driver", daemon=True)
        self._driver.start()

    def _loop(self) -> None:
        engine = self.engine
        while not self._stop:
            # Drain any newly submitted requests into the scheduler before
            # stepping. Pulling between steps (rather than during) avoids
            # touching scheduler state while step() is mid-flight.
            while True:
                try:
                    seq, fut = self.incoming.get_nowait()
                except Empty:
                    break
                engine.scheduler.add(seq)
                self.futures[seq.seq_id] = fut

            if engine.is_finished():
                # Park briefly so we don't hot-spin while idle. Wait on the
                # queue so a new request unblocks us immediately.
                try:
                    seq, fut = self.incoming.get(timeout=0.05)
                except Empty:
                    continue
                engine.scheduler.add(seq)
                self.futures[seq.seq_id] = fut
                continue

            output, _ = engine.step()
            for seq_id, token_ids in output:
                fut = self.futures.pop(seq_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(token_ids)

    def submit(self, prompt_ids: list[int], sp: SamplingParams) -> Future:
        seq = Sequence(prompt_ids, sp)
        fut: Future = Future()
        self.incoming.put((seq, fut))
        return fut

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def shutdown(self) -> None:
        self._stop = True
        self._driver.join(timeout=5)


def _make_handler(service: EngineService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence per-request access logs
            return

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{"id": service.model_name, "object": "model"}],
                })
                return
            if self.path in ("/health", "/healthz"):
                self._json(200, {"status": "ok"})
                return
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

        def do_POST(self) -> None:
            if self.path != "/v1/completions":
                self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                req = json.loads(self.rfile.read(length))
            except Exception as e:
                self._json(400, {"error": {"message": f"bad json: {e}"}})
                return

            prompt = req.get("prompt")
            if isinstance(prompt, str):
                prompt_ids = service.encode(prompt)
            elif isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
                prompt_ids = list(prompt)
            else:
                self._json(400, {"error": {"message": "prompt must be a non-empty str or list[int]"}})
                return

            try:
                max_tokens = int(req.get("max_tokens", 16))
            except (TypeError, ValueError):
                self._json(400, {"error": {"message": "max_tokens must be int"}})
                return
            ignore_eos = bool(req.get("ignore_eos", False))
            try:
                temperature = float(req.get("temperature", 1.0))
            except (TypeError, ValueError):
                self._json(400, {"error": {"message": "temperature must be float"}})
                return
            # nano-vllm forbids temperature ~ 0 (no greedy path); epsilon-clamp
            # so OpenAI clients that send temperature=0 still go through.
            if temperature < 1e-5:
                temperature = 1e-5

            sp = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
            )
            fut = service.submit(prompt_ids, sp)
            try:
                token_ids = fut.result(timeout=600)
            except Exception as e:
                self._json(500, {"error": {"message": str(e)}})
                return

            text = service.decode(token_ids)
            usage = {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            }
            self._json(200, {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": service.model_name,
                "choices": [{
                    "index": 0,
                    "text": text,
                    "finish_reason": "length" if len(token_ids) >= max_tokens else "stop",
                    "logprobs": None,
                }],
                "usage": usage,
            })

    return Handler


class _ThreadedServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Default socketserver backlog is 5, which causes "Connection reset by
    # peer" when 256+ clients connect at once. Bump to match kernel
    # somaxconn (typically 4096 on Linux).
    request_queue_size = 4096


def main() -> int:
    # Default thread stack is 8 MB; with up to 1024 in-flight requests that
    # would reserve 8 GB of VM. 512 KB is plenty for a thread that just
    # parses JSON and waits on a Future.
    threading.stack_size(512 * 1024)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=34816)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    print(
        f"# loading {args.model} (max_model_len={args.max_model_len}, "
        f"max_num_seqs={args.max_num_seqs}, mbt={args.max_num_batched_tokens}, "
        f"util={args.gpu_memory_utilization}, eager={args.enforce_eager}) ...",
        file=sys.stderr,
    )
    service = EngineService(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    handler = _make_handler(service)
    srv = _ThreadedServer((args.host, args.port), handler)
    print(f"# serving on http://{args.host}:{args.port}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
