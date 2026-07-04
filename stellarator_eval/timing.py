from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


_ACTIVE = ContextVar("stellarator_eval_timing_active", default=None)
_PHASE = ContextVar("stellarator_eval_timing_phase", default="unscoped")


@dataclass
class TimingSession:
    phases: dict[str, dict] = field(default_factory=dict)

    def phase_data(self, name: str) -> dict:
        if name not in self.phases:
            self.phases[name] = {
                "B_time_s": 0.0,
                "B_calls": 0,
                "B_points": 0,
            }
        return self.phases[name]

    def add_b_call(self, phase: str, elapsed_s: float, points: int) -> None:
        d = self.phase_data(phase)
        d["B_time_s"] += float(elapsed_s)
        d["B_calls"] += 1
        d["B_points"] += int(points)

    def as_dict(self) -> dict:
        return {k: dict(v) for k, v in self.phases.items()}


@contextmanager
def timing_session():
    session = TimingSession()
    token = _ACTIVE.set(session)
    try:
        yield session
    finally:
        _ACTIVE.reset(token)


@contextmanager
def timing_phase(name: str):
    token = _PHASE.set(name)
    try:
        yield
    finally:
        _PHASE.reset(token)


def record_b_call(elapsed_s: float, points: int) -> None:
    session = _ACTIVE.get()
    if session is not None:
        session.add_b_call(_PHASE.get(), elapsed_s, points)
