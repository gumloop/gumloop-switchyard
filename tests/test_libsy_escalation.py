# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the escalation-mode Python binding.

Imports go through ``switchyard_rust.libsy`` — the bindings-only surface that must
stay importable without the ``lib`` extra's provider SDKs.
"""

import json
from typing import Any

import pytest

from switchyard_rust.libsy import LlmTarget, escalation


def request_body(text: str = "keep going") -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ],
    }


class EchoClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    async def call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "model": self.model,
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.model}],
                    "stop_reason": "end_turn",
                }
            ],
        }


class VerdictJudge(EchoClient):
    def __init__(self, verdicts: list[bool]) -> None:
        super().__init__("judge")
        self.verdicts = verdicts

    async def call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        escalate = self.verdicts.pop(0)
        text = json.dumps({"escalate": escalate, "reason": "test verdict"})
        return {
            "model": self.model,
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                }
            ],
        }


def build(judge, efficient, capable, confirmations=1):
    return escalation(
        LlmTarget("judge", judge),
        LlmTarget("efficient", efficient),
        LlmTarget("capable", capable),
        confirmations=confirmations,
    )


async def test_decline_returns_the_efficient_answer_without_a_second_call() -> None:
    efficient, capable = EchoClient("efficient"), EchoClient("capable")
    algorithm = build(VerdictJudge([False]), efficient, capable)

    _, response = await algorithm.run(request_body())

    assert response["model"] == "efficient"
    assert len(efficient.calls) == 1
    assert capable.calls == []


async def test_confirmed_escalation_moves_to_the_capable_target() -> None:
    efficient, capable = EchoClient("efficient"), EchoClient("capable")
    algorithm = build(VerdictJudge([True]), efficient, capable, confirmations=1)

    decisions, response = await algorithm.run(request_body())

    assert response["model"] == "capable"
    assert len(efficient.calls) == 1  # judged answer was produced first
    assert len(capable.calls) == 1
    assert decisions[-1]["selected_model"] == "capable"


async def test_streak_below_confirmations_stays_efficient_then_latches() -> None:
    """With confirmations=2 the first escalate verdict stays efficient; the second,
    in the same session, latches capable."""
    efficient, capable = EchoClient("efficient"), EchoClient("capable")
    judge = VerdictJudge([True, True])
    algorithm = build(judge, efficient, capable, confirmations=2)
    headers = {"session-id": "run-42"}

    _, first = await algorithm.run(request_body("step one"), headers=headers)
    _, second = await algorithm.run(request_body("step two"), headers=headers)

    assert first["model"] == "efficient"
    assert second["model"] == "capable"
    # A latched session stops consulting the judge entirely.
    _, third = await algorithm.run(request_body("step three"), headers=headers)
    assert third["model"] == "capable"
    assert len(judge.calls) == 2


async def test_streak_does_not_cross_sessions() -> None:
    efficient, capable = EchoClient("efficient"), EchoClient("capable")
    judge = VerdictJudge([True, True])
    algorithm = build(judge, efficient, capable, confirmations=2)

    _, first = await algorithm.run(request_body(), headers={"session-id": "run-a"})
    _, second = await algorithm.run(request_body(), headers={"session-id": "run-b"})

    assert first["model"] == "efficient"
    assert second["model"] == "efficient"


async def test_judge_failure_stays_on_the_efficient_answer() -> None:
    class FailingJudge:
        async def call(self, request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("judge unavailable")

    efficient, capable = EchoClient("efficient"), EchoClient("capable")
    algorithm = build(FailingJudge(), efficient, capable)

    _, response = await algorithm.run(request_body())

    assert response["model"] == "efficient"
    assert capable.calls == []


def test_zero_confirmations_is_rejected() -> None:
    with pytest.raises(ValueError, match="confirmations"):
        build(EchoClient("judge"), EchoClient("efficient"), EchoClient("capable"), confirmations=0)
