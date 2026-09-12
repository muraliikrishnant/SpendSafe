"""Incremental, resumable output.csv writer with bounds validation."""
from __future__ import annotations

import csv
import os
import threading

from schemas import OUTPUT_COLUMNS, OutputRow


class OutputWriter:
    def __init__(self, path: str, resume: bool = True):
        self.path = path
        self._lock = threading.Lock()
        self.done_request_ids: set[str] = set()

        if resume and os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.done_request_ids.add(row["request_id"])

        write_header = not (resume and os.path.exists(path))
        mode = "a" if resume and os.path.exists(path) else "w"
        self._file = open(path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def is_done(self, request_id: str) -> bool:
        return request_id in self.done_request_ids

    def write(self, row: OutputRow) -> None:
        _validate(row)
        with self._lock:
            self._writer.writerow(row.to_csv_row())
            self._file.flush()
            self.done_request_ids.add(row.request_id)

    def close(self) -> None:
        self._file.close()


def _validate(row: OutputRow) -> None:
    if not (0 <= row.amount_safe_to_pay):
        raise ValueError(f"{row.request_id}: amount_safe_to_pay below 0")
    for pay in row.payment_plan:
        if pay.amount < 0:
            raise ValueError(f"{row.request_id}: negative payment amount")
    stopped = {c.event_id for c in row.spending_changes_needed if c.action == "stop"}
    reduced = {c.event_id for c in row.spending_changes_needed if c.action == "reduce_to"}
    if stopped & reduced:
        raise ValueError(f"{row.request_id}: same event both stopped and reduced")
    if len(row.spending_changes_needed) > 3:
        raise ValueError(f"{row.request_id}: more than 3 spending changes")
