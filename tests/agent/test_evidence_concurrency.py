"""Concurrent tools must keep source IDs, JSON snapshots and raw records intact."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
from threading import Barrier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from agent.evidence import EvidenceStore, Source


def source(url):
    return Source("", "tavily", url, url, "Evidence " * 100, [])


def test_parallel_adds_deduplicate_assign_unique_ids_and_preserve_raw_results(tmp_path):
    store = EvidenceStore(tmp_path)
    workers, batches = 8, 12
    start = Barrier(workers)

    def collect(worker):
        start.wait(timeout=5)
        added = []
        for batch in range(batches):
            query = f"{worker}-{batch}"
            added.extend(store.add([
                source("https://example.org/shared"),
                source(f"https://example.org/{query}"),
            ]))
            store.save_raw("tavily", query, [{"content": "Unicode evidence: αβγ " * 200}])
        return added

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(collect, worker) for worker in range(workers)]
        added = [item for future in futures for item in future.result(timeout=15)]

    records = json.loads((tmp_path / "sources.json").read_text())
    expected = workers * batches + 1
    assert len(added) == len(store.sources) == len(records) == expected
    assert len({record["url"] for record in records}) == expected
    assert [record["source_id"] for record in records] == [f"S{i}" for i in range(1, expected + 1)]
    assert records == [asdict(item) for item in store.sources]
    raw = [json.loads(line) for line in (tmp_path / "raw_results.jsonl").read_text().splitlines()]
    assert {record["query"] for record in raw} == {
        f"{worker}-{batch}" for worker in range(workers) for batch in range(batches)
    }
    assert len(raw) == workers * batches
    assert all(record["results"] == [{"content": "Unicode evidence: αβγ " * 200}] for record in raw)
    assert not (tmp_path / "sources.tmp").exists()


def test_parallel_save_calls_do_not_collide_on_temporary_file(tmp_path):
    store = EvidenceStore(tmp_path)
    store.add([source(f"https://example.org/{i}") for i in range(20)])
    start = Barrier(8)

    def save():
        start.wait(timeout=5)
        for _ in range(10):
            store.save()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(save) for _ in range(8)]
        for future in futures:
            future.result(timeout=15)
    assert json.loads((tmp_path / "sources.json").read_text()) == [asdict(item) for item in store.sources]
    assert not (tmp_path / "sources.tmp").exists()
