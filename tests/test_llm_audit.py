"""End-to-end evidence, offline replay and honest cost accounting."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from trialgrid.controllers.llm_audit import (
    LLMRun,
    replay_llm_run,
    run_llm_episode,
    summarize_llm_run,
)
from trialgrid.controllers.llm_models import FakeModel, OpenAIModel
from trialgrid.controllers.llm_types import LLMConfig, ModelCallError, ModelReply, TokenPricing
from trialgrid.domain import Action
from trialgrid.scenario import Scenario


def scenario(hours=1):
    source = Scenario.from_json(Path('examples/scenarios/hold_24h.json').read_text())
    return replace(
        source,
        prices_eur_per_mwh=source.prices_eur_per_mwh[:hours],
        reserve_requirements_kwh=source.reserve_requirements_kwh[:hours],
        forecast_prices_eur_per_mwh=tuple(
            row[:hours - i] for i, row in enumerate(source.forecast_prices_eur_per_mwh[:hours])
        ),
    )


def reply(power=0, *, latency=0.25, usage=True, malformed=False, writes=0):
    data = {
        'model': 'recorded-model', 'status': 'completed',
        'output': [{'type': 'function_call', 'name': 'dispatch', 'call_id': 'test-call',
                    'arguments': json.dumps({'p_kw': power}) if not malformed else '{'}],
    }
    if usage:
        data['usage'] = {
            'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
            'input_tokens_details': {'cached_tokens': 40, 'cache_write_tokens': writes},
        }
    return ModelReply(json.dumps(data), latency)


class ScriptedModel:
    def __init__(self, *outcomes):
        self.outcomes = iter(outcomes)

    def complete(self, request_json):
        result = next(self.outcomes)
        if isinstance(result, Exception):
            raise result
        return result


def test_full_fake_episode_is_deterministic_and_replays_without_any_model(monkeypatch):
    source = scenario(24)
    config = LLMConfig('fake-hold')
    first = run_llm_episode(source, FakeModel(), config)
    assert first == run_llm_episode(source, FakeModel(), config)
    assert len(first.decisions) == len(first.episode.trace) == 24

    def no_model(*args, **kwargs):
        raise AssertionError('Replay attempted to construct or invoke a live/fake model')

    monkeypatch.setattr(OpenAIModel, '__init__', no_model)
    monkeypatch.setattr(FakeModel, 'complete', no_model)
    assert LLMRun.from_json(first.to_json()) == replay_llm_run(first) == first
    summary = summarize_llm_run(first)
    assert summary['ending_energy_kwh'] == 50
    assert summary['realized_profit_eur'] == 0
    assert summary['reserve_violations'] == 4
    assert summary['model_decisions'] == 24
    assert summary['fallback_holds'] == summary['rejected_actions'] == 0


def test_error_malformed_valid_retry_sequence_replays_with_exact_attempts():
    run = run_llm_episode(scenario(), ScriptedModel(
        ModelCallError('api_timeout', True, 0.5), reply(malformed=True), reply(power=5),
    ), LLMConfig('model'))
    assert replay_llm_run(run) == run
    record = run.decisions[0]
    assert record.model_requested_action == Action(5)
    assert [a.attempt_number for a in record.attempts] == [1, 2, 3]
    assert record.attempts[0].transport_error.code == 'api_timeout'
    assert record.attempts[1].output_error == 'invalid_dispatch_arguments'
    summary = summarize_llm_run(run)
    assert summary['call_attempts'] == 3
    assert summary['retry_attempts'] == 2
    assert summary['transport_failures'] == summary['output_failures'] == 1
    assert summary['total_call_latency_s'] == 1
    assert summary['api_cost_estimate_usd'] is None


def test_fallback_hold_is_distinguished_from_model_hold():
    run = run_llm_episode(scenario(2), ScriptedModel(
        ModelCallError('api_status_401', False), reply(),
    ), LLMConfig('model', failure_policy='hold'))
    assert replay_llm_run(run) == run
    fallback, decision = run.decisions
    assert fallback.outcome == 'fallback_hold'
    assert fallback.model_requested_action is None
    assert fallback.returned_action == decision.model_requested_action == Action()
    summary = summarize_llm_run(run)
    assert summary['model_decisions'] == summary['fallback_holds'] == 1
    assert summary['failed_decisions'] == summary['retry_attempts'] == 0


def test_abort_preserves_failure_evidence_and_offline_replay():
    run = run_llm_episode(scenario(3), ScriptedModel(
        reply(), ModelCallError('api_status_401', False, 2),
    ), LLMConfig('model'))
    assert run.episode is None
    assert run.failure == 'model_call_failure'
    assert len(run.decisions) == 2
    assert run.decisions[-1].returned_action is None
    assert replay_llm_run(run) == run
    summary = summarize_llm_run(run)
    assert summary['status'] == 'model_call_failure'
    assert summary['realized_profit_eur'] is summary['ending_energy_kwh'] is None
    assert summary['model_decisions'] == summary['failed_decisions'] == 1


@pytest.mark.parametrize('mutation', [
    'request', 'observation', 'action', 'model', 'prompt', 'attempt_number',
    'extra_response', 'missing_response', 'reordered_responses', 'episode', 'schema',
])
def test_tampered_evidence_is_rejected(mutation):
    run = run_llm_episode(scenario(2), ScriptedModel(reply(2), reply(3)), LLMConfig('model'))
    data = run.to_dict()
    first = data['decisions'][0]
    if mutation == 'request':
        first['attempts'][0]['request_json'] = '{}'
    elif mutation == 'observation':
        first['observation_json'] = '{}'
    elif mutation == 'action':
        first['model_requested_action']['p_kw'] = 19
    elif mutation == 'model':
        data['config']['model_id'] = 'other-model'
    elif mutation == 'prompt':
        data['config']['prompt_version'] = 'unknown-version'
    elif mutation == 'attempt_number':
        first['attempts'][0]['attempt_number'] = 2
    elif mutation == 'extra_response':
        first['attempts'].append(first['attempts'][0].copy())
    elif mutation == 'missing_response':
        first['attempts'].clear()
    elif mutation == 'reordered_responses':
        a, b = (record['attempts'][0] for record in data['decisions'])
        a['response_json'], b['response_json'] = b['response_json'], a['response_json']
    elif mutation == 'episode':
        data['episode']['metrics']['profit_eur'] = 1000
    elif mutation == 'schema':
        data['schema_version'] = True
    with pytest.raises(ValueError):
        LLMRun.from_dict(data)


def test_cost_counts_cached_tokens_and_malformed_but_billed_response():
    rates = TokenPricing(2, 1, 5, 'test tariffs')
    run = run_llm_episode(scenario(), ScriptedModel(reply(malformed=True), reply()),
                          LLMConfig('model', pricing=rates))
    assert replay_llm_run(run) == run
    summary = summarize_llm_run(run)
    assert summary['api_cost_estimate_usd'] == pytest.approx(2 * (60 * 2 + 40 + 20 * 5) / 1e6)
    assert summary['token_usage'] == {
        'input_tokens': 200, 'output_tokens': 40, 'cached_input_tokens': 80,
        'cache_write_tokens': 0,
    }
    assert summary['total_call_latency_s'] == 0.5


@pytest.mark.parametrize('unknown', ['usage', 'error', 'cache_writes', 'rates'])
def test_unknown_cost_is_not_reported_as_zero(unknown):
    outcomes = [reply(usage=unknown != 'usage', writes=10 if unknown == 'cache_writes' else 0)]
    if unknown == 'error':
        outcomes.insert(0, ModelCallError('api_timeout'))
    rates = None if unknown == 'rates' else TokenPricing(2, 1, 5, 'test tariffs')
    run = run_llm_episode(scenario(), ScriptedModel(*outcomes), LLMConfig('model', pricing=rates))
    assert replay_llm_run(run) == run
    assert summarize_llm_run(run)['api_cost_estimate_usd'] is None
