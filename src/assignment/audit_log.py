"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.

Thiết kế: side observer (không phải ADK plugin). Pipeline gọi
``record_input`` trước khi request vào plugin chain và ``record_output``
sau khi có câu trả lời cuối cùng — nhờ vậy log được cả request bị
rate-limit / input guardrail chặn (những request không bao giờ tới LLM).
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    # Giới hạn độ dài text lưu trong log (tránh log phình to khi bị spam input dài)
    MAX_TEXT = 2000

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # request_id -> thông tin input đang chờ output
        self._open: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or f"user:{user_id}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu input + thời điểm bắt đầu. Trả về request_id dùng cho record_output."""
        request_id = request_id or uuid.uuid4().hex[:12]
        self._open[self._key(user_id, request_id)] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": (text or "")[: self.MAX_TEXT],
            "input_length": len(text or ""),
            "started_at": utc_now_iso(),
            "_t0": time.perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu output, lớp quyết định, latency; append vào self.logs."""
        entry = self._open.pop(self._key(user_id, request_id), None)
        if entry is None:  # output không có input tương ứng — vẫn log để không mất dấu
            entry = {
                "request_id": request_id or uuid.uuid4().hex[:12],
                "user_id": user_id,
                "input": None,
                "input_length": None,
                "started_at": None,
                "_t0": None,
            }
        t0 = entry.pop("_t0")
        entry.update(
            {
                "output": (text or "")[: self.MAX_TEXT],
                "blocked": bool(blocked),
                "layer": layer,
                "finished_at": utc_now_iso(),
                "latency_ms": (
                    round((time.perf_counter() - t0) * 1000, 1) if t0 is not None else None
                ),
            }
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
