"""
Codeforces interaction data collection pipeline.

Collects user submissions + rating histories, derives per-(user, problem)
interaction features, and writes partitioned Parquet files for downstream ML.

Outputs (all under DATA_DIR):
  users.csv                         – handle, user_idx, registration_time_seconds
  interactions/part_NNN.parquet     – edge list with features
  tags.json                         – {tag_name: tag_index}
  problems.parquet                  – problem metadata
  .progress.json                    – handles already processed (resume state)

Stop at any time with Ctrl-C: in-flight edges are flushed, state is saved,
and the next run resumes from where it left off.
"""

import os
import json
import time
import signal
import random
import logging
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import polars as pl
from dotenv import load_dotenv

from cf_api import CodeForcesAPI

# ── configuration ─────────────────────────────────────────────────────────────
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROXY_FILE = Path("proxylist.txt")
RATE_LIMIT = 2  # seconds between CF API calls per thread
RETRY = 5
TIMEOUT = 15
FLUSH_EVERY = 5_000_000  # edges per parquet partition
NUM_THREADS = 20
PROGRESS_FILE = DATA_DIR / ".progress.json"
PROXY_FAIL_LIMIT = 5  # consecutive failures before killing the thread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── shutdown flag ─────────────────────────────────────────────────────────────

_stop = threading.Event()  # set → workers finish current user then exit


def _handle_sigint(sig, frame):
    if _stop.is_set():
        logger.warning("Second Ctrl-C — forcing exit")
        raise SystemExit(1)
    logger.warning(
        "Ctrl-C received — finishing in-flight users, then stopping …  (Ctrl-C again to force)"
    )
    _stop.set()


signal.signal(signal.SIGINT, _handle_sigint)

# ── shared mutable state ──────────────────────────────────────────────────────

problem_lock = threading.Lock()
tag_lock = threading.Lock()
flush_lock = threading.Lock()
part_idx_lock = threading.Lock()
progress_lock = threading.Lock()
done_lock = threading.Lock()

problem_map: dict[str, dict] = {}  # problem_id → {idx, rating, tags(indices)}
tag_map: dict[str, int] = {}  # tag_name → tag_index
done_handles: set[str] = set()  # handles already saved to disk
part_idx = 1

users_done = 0
users_total = 0
pipeline_start_time = 0.0

# ── helpers ───────────────────────────────────────────────────────────────────


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def progress_str(done: int, total: int, elapsed: float) -> str:
    pct = 100.0 * done / total if total else 0.0
    if done > 0 and elapsed > 0:
        rate = done / elapsed
        remaining = (total - done) / rate
        eta = f"ETA {fmt_duration(remaining)} @ {rate:.1f} users/s"
    else:
        eta = "ETA unknown"
    return f"{done:,}/{total:,} ({pct:.1f}%) — {eta}"


def load_proxies() -> list[Optional[dict]]:
    if not PROXY_FILE.exists():
        logger.warning("proxylist.txt not found — running without proxies")
        return [None]
    lines = [l.strip() for l in PROXY_FILE.read_text().splitlines() if l.strip()]
    if not lines:
        return [None]
    proxies = [{"http": l, "https": l} for l in lines]
    logger.info(f"Loaded {len(proxies)} proxies")
    return proxies


def load_progress() -> set[str]:
    """Load set of already-completed handles from disk (if any)."""
    if not PROGRESS_FILE.exists():
        return set()
    try:
        data = json.loads(PROGRESS_FILE.read_text())
        handles = set(data.get("done_handles", []))
        logger.info(
            f"Resuming — {len(handles):,} users already processed, skipping them"
        )
        return handles
    except Exception as e:
        logger.warning(f"Could not read progress file: {e} — starting fresh")
        return set()


def save_progress(handles: set[str]) -> None:
    """Persist completed handles to disk atomically."""
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"done_handles": list(handles)}, ensure_ascii=False))
    tmp.replace(PROGRESS_FILE)


def get_or_add_tag(tag: str) -> int:
    with tag_lock:
        if tag not in tag_map:
            tag_map[tag] = len(tag_map)
        return tag_map[tag]


def get_or_add_problem(p: dict) -> str:
    contest_id = p.get("contestId", "")
    index = p.get("index", "")
    problem_id = f"{contest_id}.{index}"
    with problem_lock:
        if problem_id not in problem_map:
            tag_indices = [get_or_add_tag(t) for t in p.get("tags", [])]
            problem_map[problem_id] = {
                "idx": len(problem_map),
                "rating": p.get("rating"),
                "tags": tag_indices,
            }
    return problem_id


def nearest_rating_at(rating_history: list[dict], ts: int) -> Optional[int]:
    best_rating, best_ts = None, -1
    for entry in rating_history:
        t = entry.get("ratingUpdateTimeSeconds", 0)
        if t <= ts and t > best_ts:
            best_ts = t
            best_rating = entry.get("newRating")
    return best_rating


def flush_edges(edges: list[dict], out_dir: Path) -> None:
    global part_idx
    if not edges:
        return
    t0 = time.perf_counter()
    with part_idx_lock:
        idx = part_idx
        part_idx += 1
    fname = out_dir / f"part_{idx:03d}.parquet"
    pl.DataFrame(edges).write_parquet(fname)
    logger.info(
        f"[flush] {len(edges):,} edges → {fname.name} ({fmt_duration(time.perf_counter() - t0)})"
    )
    edges.clear()


# ── worker ────────────────────────────────────────────────────────────────────


def worker(
    queue: Queue,
    user_map: dict[str, dict],
    api: CodeForcesAPI,
    out_dir: Path,
    proxy_label: str,  # human-readable proxy id for log messages
) -> None:
    global users_done

    handled_verdicts = {
        "FAILED",
        "OK",
        "COMPILATION_ERROR",
        "RUNTIME_ERROR",
        "WRONG_ANSWER",
        "TIME_LIMIT_EXCEEDED",
        "MEMORY_LIMIT_EXCEEDED",
        "IDLENESS_LIMIT_EXCEEDED",
        "SECURITY_VIOLATED",
        "REJECTED",
    }

    name = threading.current_thread().name
    local_edges: list[dict] = []
    local_done = 0
    log_every = 50
    consecutive_failures = 0  # resets to 0 on any successful user

    while not _stop.is_set():
        try:
            handle, user_idx = queue.get(timeout=1)
        except Empty:
            if queue.empty():
                break
            continue

        reg_ts = user_map[handle]["registration_time_seconds"]
        t_user = time.perf_counter()
        user_ok = False

        try:
            submissions: list[dict] = api.request(
                "user.status",
                {"handle": handle},
                timeout=TIMEOUT,
                retry=RETRY,
            )
            try:
                rating_history: list[dict] = api.request(
                    "user.rating",
                    {"handle": handle},
                    timeout=TIMEOUT,
                    retry=RETRY,
                )
            except Exception:
                rating_history = []

            problem_subs: dict[str, list[dict]] = {}
            for sub in submissions:
                if sub.get("verdict") not in handled_verdicts:
                    continue
                if sub["problem"]["type"] != "PROGRAMMING":
                    continue
                pid = get_or_add_problem(sub["problem"])
                problem_subs.setdefault(pid, []).append(sub)

            for pid, subs in problem_subs.items():
                subs_chrono = list(reversed(subs))
                verdicts = [s.get("verdict") for s in subs_chrono]
                solved = "OK" in verdicts
                first_ok_idx = verdicts.index("OK") if solved else None
                tries = (first_ok_idx + 1) if solved else len(subs_chrono)
                decisive_sub = subs_chrono[first_ok_idx] if solved else subs_chrono[-1]
                decisive_ts = decisive_sub.get("creationTimeSeconds", reg_ts)
                final_rating = nearest_rating_at(rating_history, decisive_ts)
                experience_yrs = (decisive_ts - reg_ts) / (365.25 * 24 * 3600)

                local_edges.append(
                    {
                        "user_index": user_idx,
                        "problem_index": problem_map[pid]["idx"],
                        "solved": solved,
                        "tries": tries,
                        "final_rating": final_rating,
                        "experience_years": experience_yrs,
                    }
                )

            with done_lock:
                done_handles.add(handle)

            user_ok = True

        except Exception as e:
            logger.error(f"[{name}] Failed for user {handle!r}: {e}")
        finally:
            queue.task_done()

        # ── proxy health tracking ─────────────────────────────────────────────
        if user_ok:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= PROXY_FAIL_LIMIT:
                logger.error(
                    f"[{name}] Proxy {proxy_label!r} failed {consecutive_failures} users "
                    f"in a row — killing thread"
                )
                break

        user_elapsed = time.perf_counter() - t_user
        local_done += 1

        with progress_lock:
            users_done += 1
            global_done = users_done
            wall = time.perf_counter() - pipeline_start_time

        if local_done % log_every == 0:
            logger.info(
                f"[{name}] {progress_str(global_done, users_total, wall)} "
                f"| last user {fmt_duration(user_elapsed)} "
                f"| {len(local_edges):,} edges buffered"
            )

        if len(local_edges) >= FLUSH_EVERY:
            with flush_lock:
                flush_edges(local_edges, out_dir)

    # ── thread is exiting (queue empty, _stop set, or proxy dead) ─────────────
    if local_edges:
        with flush_lock:
            flush_edges(local_edges, out_dir)

    with progress_lock:
        wall = time.perf_counter() - pipeline_start_time
    logger.info(f"[{name}] done — {local_done:,} users in {fmt_duration(wall)} total")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    global users_total, pipeline_start_time, done_handles

    api_key = os.getenv("CF_API_KEY")
    api_secret = os.getenv("CF_SECRET")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    interactions_dir = DATA_DIR / "interactions"
    interactions_dir.mkdir(exist_ok=True)

    proxies = load_proxies()

    # ── 1. fetch all rated users ──────────────────────────────────────────────
    logger.info("Fetching rated user list …")
    t0 = time.perf_counter()
    bootstrap_api = CodeForcesAPI(api_key, api_secret, rate_limit=RATE_LIMIT)
    users_raw: list[dict] = bootstrap_api.request(
        "user.ratedList",
        {"includeRetired": "false", "activeOnly": "false"},
        timeout=30,
        retry=RETRY,
    )
    logger.info(
        f"Got {len(users_raw):,} rated users in {fmt_duration(time.perf_counter() - t0)}"
    )

    random.shuffle(users_raw)

    # ── 2. build user_map and save users.csv ──────────────────────────────────
    user_map: dict[str, dict] = {}
    rows = []
    for idx, u in enumerate(users_raw):
        handle = u["handle"]
        reg_ts = u["registrationTimeSeconds"]
        user_map[handle] = {"user_idx": idx, "registration_time_seconds": reg_ts}
        rows.append(
            {"handle": handle, "user_idx": idx, "registration_time_seconds": reg_ts}
        )

    t0 = time.perf_counter()
    pl.DataFrame(rows).write_csv(DATA_DIR / "users.csv")
    logger.info(
        f"Saved {len(rows):,} users → users.csv ({fmt_duration(time.perf_counter() - t0)})"
    )

    # ── 3. load resume state, fill work queue ────────────────────────────────
    done_handles = load_progress()
    users_to_process = [u for u in users_raw if u["handle"] not in done_handles]
    users_total = len(users_to_process)

    if not users_to_process:
        logger.info("All users already processed — nothing to do")
        return

    logger.info(f"Queueing {users_total:,} users ({len(done_handles):,} skipped)")
    queue: Queue = Queue()
    for u in users_to_process:
        h = u["handle"]
        queue.put((h, user_map[h]["user_idx"]))

    # ── 4. start worker threads ───────────────────────────────────────────────
    n_threads = min(
        NUM_THREADS, len(proxies) if proxies[0] is not None else NUM_THREADS
    )
    threads: list[threading.Thread] = []

    pipeline_start_time = time.perf_counter()
    for i in range(n_threads):
        proxy = proxies[i % len(proxies)]
        proxy_label = proxy["http"] if proxy else "none"
        thread_api = CodeForcesAPI(
            api_key, api_secret, rate_limit=RATE_LIMIT, proxy=proxy
        )
        t = threading.Thread(
            target=worker,
            args=(queue, user_map, thread_api, interactions_dir, proxy_label),
            daemon=True,
            name=f"worker-{i:02d}",
        )
        t.start()
        threads.append(t)
        logger.info(f"Started {t.name} (proxy={proxy_label!r})")

    logger.info(f"All {n_threads} workers running — {users_total:,} users queued")

    for t in threads:
        t.join()

    wall = time.perf_counter() - pipeline_start_time
    stopped_early = _stop.is_set()
    logger.info(
        f"{'Stopped' if stopped_early else 'All workers finished'} — "
        f"{users_done:,} users in {fmt_duration(wall)} ({users_done / max(wall, 1):.1f} users/s)"
    )

    # ── 5. persist resume state ───────────────────────────────────────────────
    with done_lock:
        save_progress(done_handles)
    logger.info(f"Progress saved ({len(done_handles):,} total done handles)")

    if stopped_early:
        logger.info("Run was interrupted — restart to continue from this checkpoint")
        return

    # ── 6. save tags.json ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    tags_path = DATA_DIR / "tags.json"
    tags_path.write_text(json.dumps(tag_map, ensure_ascii=False, indent=2))
    logger.info(
        f"Saved {len(tag_map)} tags → tags.json ({fmt_duration(time.perf_counter() - t0)})"
    )

    # ── 7. save problems.parquet ──────────────────────────────────────────────
    if problem_map:
        t0 = time.perf_counter()
        items = list(problem_map.items())
        pl.DataFrame(
            {
                "problem_id": [pid for pid, _ in items],
                "problem_index": [m["idx"] for _, m in items],
                "problem_rating": [m["rating"] for _, m in items],
                "problem_tags": [m["tags"] for _, m in items],
            }
        ).write_parquet(DATA_DIR / "problems.parquet")
        logger.info(
            f"Saved {len(items):,} problems → problems.parquet "
            f"({fmt_duration(time.perf_counter() - t0)})"
        )

    total_wall = time.perf_counter() - pipeline_start_time
    logger.info(f"Pipeline complete ✓ — total runtime {fmt_duration(total_wall)}")


if __name__ == "__main__":
    main()
