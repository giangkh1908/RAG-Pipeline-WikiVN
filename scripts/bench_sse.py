"""Micro-benchmark the SSE token-batching layer in chat.py (v2, deterministic).

A producer puts tokens onto an asyncio.Queue at scheduled wall-clock arrival
times (matching the generator's inter-token cadence). Each variant consumes
the queue with its flush logic and records time-to-first-flush + total.
No network / OpenRouter / Qdrant — isolates the batching overhead only.
"""
from __future__ import annotations

import asyncio
import time

_FLUSH_CHARS = 32
_FLUSH_INTERVAL = 0.04  # 40 ms — current chat.py values


async def producer(queue: asyncio.Queue, schedule: list[tuple[float, str]]) -> None:
    """Emit each (arrival_s, token) at its scheduled time, then a None sentinel."""
    start = time.perf_counter()
    for arrival, tok in schedule:
        delay = arrival - (time.perf_counter() - start)
        if delay > 0:
            await asyncio.sleep(delay)
        await queue.put(tok)
    await queue.put(None)


def make_schedule(tokens: list[str], ttft_s: float, gap_s: float) -> list[tuple[float, str]]:
    t = ttft_s
    out = []
    for tok in tokens:
        out.append((t, tok))
        t += gap_s
    return out


def make_bursty_schedule(tokens: list[str]) -> list[tuple[float, str]]:
    t = 0.0
    out = []
    i = 0
    while i < len(tokens):
        for _ in range(5):
            if i >= len(tokens):
                break
            out.append((t, tokens[i]))
            i += 1
            t += 0.005
        t += 0.060  # 60 ms think gap
    return out


async def variant_batching(queue: asyncio.Queue) -> tuple[float, float, int]:
    buffer = ""
    last_flush = time.monotonic()
    start = time.monotonic()
    first: float | None = None
    flushes = 0

    def flush() -> None:
        nonlocal buffer, last_flush, flushes, first
        buffer = ""
        last_flush = time.monotonic()
        flushes += 1
        if first is None:
            first = time.monotonic() - start

    while True:
        if buffer and (time.monotonic() - last_flush) >= _FLUSH_INTERVAL:
            flush()
            continue
        if buffer:
            remaining = _FLUSH_INTERVAL - (time.monotonic() - last_flush)
            if remaining <= 0:
                flush()
                continue
            try:
                event = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                flush()
                continue
        else:
            event = await queue.get()
        if event is None:
            if buffer:
                flush()
            break
        if isinstance(event, str):
            buffer += event
            if len(buffer) >= _FLUSH_CHARS or (time.monotonic() - last_flush) >= _FLUSH_INTERVAL:
                flush()
    return (first or 0.0) * 1000, (time.monotonic() - start) * 1000, flushes


async def variant_direct(queue: asyncio.Queue) -> tuple[float, float, int]:
    start = time.monotonic()
    first: float | None = None
    flushes = 0
    while True:
        event = await queue.get()
        if event is None:
            break
        if first is None:
            first = time.monotonic() - start
        flushes += 1
    return (first or 0.0) * 1000, (time.monotonic() - start) * 1000, flushes


async def variant_char_only(queue: asyncio.Queue) -> tuple[float, float, int]:
    buffer = ""
    start = time.monotonic()
    first: float | None = None
    flushes = 0

    def flush() -> None:
        nonlocal buffer, flushes, first
        buffer = ""
        flushes += 1
        if first is None:
            first = time.monotonic() - start

    while True:
        event = await queue.get()
        if event is None:
            if buffer:
                flush()
            break
        if isinstance(event, str):
            buffer += event
            if len(buffer) >= _FLUSH_CHARS:
                flush()
    return (first or 0.0) * 1000, (time.monotonic() - start) * 1000, flushes


async def measure(schedule, label: str) -> dict:
    tokens_n = len(schedule)
    print(f"\n--- {label}  (tokens={tokens_n}) ---")
    print(f"{'variant':<12}{'TTFB(ms)':>11}{'total(ms)':>12}{'flushes':>9}{'add_TTFB(ms)':>15}{'add_total(ms)':>15}")
    direct_ttfb = direct_total = 0.0
    rows = {}
    for name, fn in (("direct", variant_direct), ("batching", variant_batching), ("char-only", variant_char_only)):
        q: asyncio.Queue = asyncio.Queue()
        await asyncio.gather(producer(q, schedule), fn(q))
        ttfb, total, flushes = await _rerun(fn, schedule)
        if name == "direct":
            direct_ttfb, direct_total = ttfb, total
        add_ttfb = ttfb - direct_ttfb
        add_total = total - direct_total
        rows[name] = dict(ttfb=ttfb, total=total, flushes=flushes, add_ttfb=add_ttfb, add_total=add_total)
        print(f"{name:<12}{ttfb:>11.1f}{total:>12.1f}{flushes:>9}{add_ttfb:>15.1f}{add_total:>15.1f}")
    return rows


async def _rerun(fn, schedule):
    q: asyncio.Queue = asyncio.Queue()
    # run fn and producer concurrently; fn returns the timings
    prod = asyncio.create_task(producer(q, schedule))
    res = await fn(q)
    await prod
    return res


async def main() -> None:
    text = "Hà Nội là thủ đô của Việt Nam, " + "nhiều du lịch đặc sản " * 30
    tokens = [t for t in text.split(" ") if t]
    await measure(make_schedule(tokens, 0.0, 0.010), "A: steady fast 10 ms/token")
    await measure(make_schedule(tokens, 0.0, 0.030), "B: steady slow 30 ms/token")
    await measure(make_bursty_schedule(tokens), "C: bursty 5x5ms + 60ms gap")


if __name__ == "__main__":
    asyncio.run(main())