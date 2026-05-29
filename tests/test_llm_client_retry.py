"""Robustness tests for the hemorrhage LLM client (timeout config + retries)."""

from unittest.mock import patch

import pytest
import requests

from src.tasks.hemorrhage.inference import llm_client


def test_timeout_default_is_240(monkeypatch):
    monkeypatch.delenv("HEMORRHAGE_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)
    assert llm_client.get_timeout_seconds() == 240


def test_timeout_env_override(monkeypatch):
    monkeypatch.setenv("HEMORRHAGE_LLM_TIMEOUT_SECONDS", "300")
    assert llm_client.get_timeout_seconds() == 300


def test_max_retries_default_is_1(monkeypatch):
    monkeypatch.delenv("HEMORRHAGE_LLM_MAX_RETRIES", raising=False)
    assert llm_client.get_max_retries() == 1


def test_max_retries_env_override(monkeypatch):
    monkeypatch.setenv("HEMORRHAGE_LLM_MAX_RETRIES", "2")
    assert llm_client.get_max_retries() == 2


def test_non_numeric_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HEMORRHAGE_LLM_TIMEOUT_SECONDS", "abc")
    assert llm_client.get_timeout_seconds() == 240


def test_retry_succeeds_on_second_attempt(monkeypatch):
    monkeypatch.setenv("HEMORRHAGE_LLM_MAX_RETRIES", "1")
    calls = {"n": 0}

    def flaky(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("read timed out")
        return '{"ok": true}'

    with patch.object(llm_client, "_call_provider_once", side_effect=flaky), patch.object(
        llm_client.time, "sleep"
    ) as sleep_mock:
        out = llm_client.call_llm([{"role": "user", "content": "x"}])

    assert out == '{"ok": true}'
    assert calls["n"] == 2
    sleep_mock.assert_called_once_with(llm_client.RETRY_WAIT_SECONDS)


def test_retry_exhausted_raises_llm_call_error(monkeypatch):
    monkeypatch.setenv("HEMORRHAGE_LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("HEMORRHAGE_LLM_TIMEOUT_SECONDS", "240")

    def always_timeout(_messages):
        raise requests.exceptions.ReadTimeout("read timed out")

    with patch.object(llm_client, "_call_provider_once", side_effect=always_timeout), patch.object(
        llm_client.time, "sleep"
    ) as sleep_mock:
        with pytest.raises(llm_client.LLMCallError) as exc_info:
            llm_client.call_llm([{"role": "user", "content": "x"}])

    msg = str(exc_info.value)
    assert "ReadTimeout" in msg
    assert "240 seconds" in msg
    # one retry => exactly one sleep
    assert sleep_mock.call_count == 1


def test_connection_error_is_retried(monkeypatch):
    monkeypatch.setenv("HEMORRHAGE_LLM_MAX_RETRIES", "1")

    def conn_err(_messages):
        raise requests.exceptions.ConnectionError("refused")

    with patch.object(llm_client, "_call_provider_once", side_effect=conn_err), patch.object(
        llm_client.time, "sleep"
    ):
        with pytest.raises(llm_client.LLMCallError):
            llm_client.call_llm([{"role": "user", "content": "x"}])


def test_non_retryable_error_propagates_immediately(monkeypatch):
    monkeypatch.setenv("HEMORRHAGE_LLM_MAX_RETRIES", "1")
    calls = {"n": 0}

    def value_err(_messages):
        calls["n"] += 1
        raise ValueError("bad config")

    with patch.object(llm_client, "_call_provider_once", side_effect=value_err), patch.object(
        llm_client.time, "sleep"
    ) as sleep_mock:
        with pytest.raises(ValueError):
            llm_client.call_llm([{"role": "user", "content": "x"}])

    assert calls["n"] == 1
    sleep_mock.assert_not_called()
