"""Exercise the example's actual update against an independent PPO calculation."""

from __future__ import annotations

import inspect
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtor.agent import combine, partition
from jaxtor.sampler import Roll

pytest.importorskip("equinox")
optax = pytest.importorskip("optax")
rlax = pytest.importorskip("rlax")
pytest.importorskip("gymnax")
pytest.importorskip("tyro")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ppo.py"


def assert_same(actual, expected, *, exact=False):
    """Compare numerical leaves, including typed PRNG keys."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        if exact or not jnp.issubdtype(a.dtype, jnp.inexact):
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


def test_import_does_not_construct_an_experiment():
    """Importing definitions must not initialize an environment or any arrays."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import runpy
import sys
from unittest.mock import patch

with patch('jaxtor.env.gymnax.make', side_effect=AssertionError('environment init')), \
     patch('jax.random.key', side_effect=AssertionError('key init')):
    module = runpy.run_path(sys.argv[1])
assert 'train' in module
assert 'cfg' not in module
""",
            str(EXAMPLE),
        ],
        check=True,
        timeout=60,
    )


@pytest.fixture
def recipe(monkeypatch):
    """Capture the local compiled update without adding a public testing API."""
    module = runpy.run_path(str(EXAMPLE))
    cfg = module["Config"](
        n_iters=0,
        n_envs=2,
        n_eval_envs=2,
        seq_len=4,
        n_epochs=3,
        n_batches=2,
        hidden_size=8,
        max_eps_len=8,
        lr=0.05,
        clip_eps=0.05,
        max_grad_norm=0.1,
    )
    compiled = {}
    jit = jax.jit

    def capture(fn, *args, **kwargs):
        result = jit(fn, *args, **kwargs)
        if fn.__name__ == "update":
            compiled["update"] = result
            compiled["components"] = inspect.getclosurevars(fn).nonlocals
        return result

    monkeypatch.setattr(jax, "jit", capture)
    state = module["train"](cfg)
    return module, cfg, state, compiled["update"], compiled["components"]


def test_update_matches_independent_ppo(recipe, monkeypatch):
    """Check GAE boundaries, clipping, frozen statistics, and all committed state."""
    module, cfg, state, update, c = recipe
    seq, sampled = c["roll"].sample(state.sample)
    seq = replace(
        seq,
        rew=jnp.array([[2.0, -3.0, 7.0, -1.0], [-2.0, 4.0, -5.0, 3.0]]),
        term=jnp.array([[False, True, False, False], [False, False, False, True]]),
        trun=jnp.array([[False, False, True, False], [False, True, False, False]]),
        nobs=seq.nobs + 0.7,
    )
    monkeypatch.setattr(Roll, "sample", lambda self, unused: (seq, sampled))
    parts = partition(sampled.agent)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr, eps=1e-5),
    )

    @jax.jit
    def reference():
        rew, rew_state = c["rew_norm"].update(
            seq.rew, seq.term | seq.trun, state.rew_norm
        )
        behavior, _ = c["agent"].vpi(seq.obs, sampled.agent)
        nv, _ = c["agent"].v(seq.nobs, sampled.agent)
        tail = nv[:, -1]
        returns = []
        for t in reversed(range(cfg.seq_len)):
            discount = cfg.gamma * (~seq.term[:, t])
            trace = cfg.gae_lambda * (~(seq.term[:, t] | seq.trun[:, t]))
            tail = rew[:, t] + discount * ((1 - trace) * nv[:, t] + trace * tail)
            returns.append(tail)
        adv = jnp.stack(returns[::-1], axis=1) - behavior.v
        batch = module["Batch"](
            obs=seq.obs,
            act=seq.act,
            log_mu=behavior.pi.evaluate(seq.act).logp,
            adv=(adv - adv.mean()) / (adv.std() + 1e-8),
            ret=adv + behavior.v,
        )

        def loss(params, b):
            vp, _ = c["agent"].vpi(b.obs, combine(params, parts.frozen))
            evaluated = vp.pi.evaluate(b.act)
            ratio = jnp.exp(evaluated.logp - b.log_mu)
            pg = rlax.clipped_surrogate_pg_loss(ratio, b.adv, cfg.clip_eps)
            vf = 0.5 * jnp.mean((vp.v - b.ret) ** 2)
            entropy = jnp.mean(evaluated.entropy)
            total = pg + cfg.vf_coef * vf - cfg.ent_coef * entropy
            favorable = ((b.adv > 0) & (ratio > 1 + cfg.clip_eps)) | (
                (b.adv < 0) & (ratio < 1 - cfg.clip_eps)
            )
            return total, (
                dict(loss=total, pg=pg, vf=vf, entropy=entropy),
                favorable.sum(),
            )

        def minibatch(carry, b):
            params, opt = carry
            (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params, b)
            updates, opt = optimizer.update(grads, opt, params)
            return (optax.apply_updates(params, updates), opt), metrics

        def epoch(carry, key):
            return jax.lax.scan(minibatch, carry, c["batches"].shuffle(key, batch))

        key, shuffle_key = jax.random.split(state.key)
        (params, opt), (metrics, clipped) = jax.lax.scan(
            epoch,
            (parts.params, state.opt),
            jax.random.split(shuffle_key, cfg.n_epochs),
        )
        learned = combine(params, parts.frozen)
        learned = replace(learned, body=c["body"].update(seq.obs, learned.body))
        expected = module["State"](
            sample=replace(sampled, agent=learned),
            opt=opt,
            episodes=c["episode_stats"].update(seq, state.episodes),
            rew_norm=rew_state,
            key=key,
        )
        return jax.tree.map(jnp.mean, metrics), expected, clipped.sum()

    expected_metrics, expected_state, clipped = reference()
    actual_metrics, actual_state = update(state)
    assert int(clipped) > 0
    assert_same(actual_metrics, expected_metrics)
    assert_same(actual_state, expected_state)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(parts.params),
            jax.tree.leaves(partition(actual_state.sample.agent).params),
            strict=True,
        )
    )


def test_seed_replay_and_multiple_configs(capsys):
    """One imported module can run independent configurations reproducibly."""
    module = runpy.run_path(str(EXAMPLE))
    cfg = module["Config"](
        n_iters=2,
        n_envs=2,
        n_eval_envs=2,
        seq_len=4,
        n_epochs=1,
        n_batches=2,
        hidden_size=8,
        max_eps_len=8,
        eval_freq=1,
    )
    first = module["train"](cfg)
    output = capsys.readouterr().out
    other = module["train"](replace(cfg, seed=1))
    capsys.readouterr()
    replay = module["train"](cfg)
    assert capsys.readouterr().out == output
    assert output.count("iter=") == 2
    assert_same(first, replay, exact=True)
    assert not np.array_equal(
        jax.random.key_data(first.key), jax.random.key_data(other.key)
    )
