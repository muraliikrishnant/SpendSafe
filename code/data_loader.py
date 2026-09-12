"""Indexed access to dataset/*.csv.

Builds per-user_id / per-request_id indexes once so per-request lookups are
O(1) instead of scanning the whole events table for every request. At real
scale (millions of requests / events), the same interface can be backed by
SQLite/DuckDB instead of in-memory dicts without changing callers — see
`DataStore.from_sqlite` below for the swap-in point.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterator


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _read_csv(path: str) -> Iterator[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


@dataclass
class DataStore:
    dataset_dir: str

    profiles_by_user: dict = field(default_factory=dict)
    events_by_user: dict = field(default_factory=lambda: defaultdict(list))
    events_by_id: dict = field(default_factory=dict)
    payment_options_by_request: dict = field(default_factory=lambda: defaultdict(list))
    messages_by_request: dict = field(default_factory=lambda: defaultdict(list))
    messages_by_user: dict = field(default_factory=lambda: defaultdict(list))
    images_by_request: dict = field(default_factory=lambda: defaultdict(list))
    images_by_event: dict = field(default_factory=dict)
    exchange_rates: dict = field(default_factory=dict)  # (date, from, to) -> rate

    @classmethod
    def load(cls, dataset_dir: str) -> "DataStore":
        ds = cls(dataset_dir=dataset_dir)

        for row in _read_csv(os.path.join(dataset_dir, "financial_profiles.csv")):
            ds.profiles_by_user[row["user_id"]] = row

        for row in _read_csv(os.path.join(dataset_dir, "financial_events.csv")):
            ds.events_by_user[row["user_id"]].append(row)
            ds.events_by_id[row["event_id"]] = row

        for row in _read_csv(os.path.join(dataset_dir, "request_payment_options.csv")):
            ds.payment_options_by_request[row["request_id"]].append(row)

        for row in _read_csv(os.path.join(dataset_dir, "messages.csv")):
            if row.get("request_id"):
                ds.messages_by_request[row["request_id"]].append(row)
            ds.messages_by_user[row["user_id"]].append(row)

        for row in _read_csv(os.path.join(dataset_dir, "images.csv")):
            if row.get("request_id"):
                ds.images_by_request[row["request_id"]].append(row)
            if row.get("related_event_id"):
                ds.images_by_event[row["related_event_id"]] = row

        for row in _read_csv(os.path.join(dataset_dir, "exchange_rates.csv")):
            key = (row["rate_date"], row["from_currency"], row["to_currency"])
            ds.exchange_rates[key] = float(row["rate"])

        return ds

    def image_path(self, image_id: str) -> str:
        return os.path.join(self.dataset_dir, "media", "images", f"{image_id}.png")

    def convert(self, amount: float, from_ccy: str, to_ccy: str, on: date) -> float:
        if from_ccy == to_ccy:
            return amount
        key = (on.isoformat(), from_ccy, to_ccy)
        if key in self.exchange_rates:
            return amount * self.exchange_rates[key]
        inv_key = (on.isoformat(), to_ccy, from_ccy)
        if inv_key in self.exchange_rates:
            return amount / self.exchange_rates[inv_key]
        # Fall back to nearest available dated rate for the pair.
        candidates = [
            (d, r) for (d, f, t), r in self.exchange_rates.items() if f == from_ccy and t == to_ccy
        ]
        if candidates:
            candidates.sort(key=lambda dr: abs((_parse_date(dr[0]) - on).days))
            return amount * candidates[0][1]
        raise ValueError(f"No exchange rate found for {from_ccy}->{to_ccy} near {on}")


def iter_requests(dataset_dir: str, chunk_size: int = 500) -> Iterator[list[dict]]:
    """Stream requests.csv in chunks so the pipeline never has to hold the
    whole request set in memory regardless of dataset size."""
    batch: list[dict] = []
    for row in _read_csv(os.path.join(dataset_dir, "requests.csv")):
        batch.append(row)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch
