"""Probability-distribution values for action selection.

Distributions are parameter pytrees, typically produced by policy heads. Their
operations are pure and require no separate state::

    dist = Categorical(logits=logits)
    act = dist.sample(key)
    evaluation = dist.evaluate(act)
    mode = dist.mode()
"""

from __future__ import annotations

import math
from typing import Protocol

import jax
import jax.numpy as jnp
from chex import dataclass

_LOG_TWO_PI = math.log(2.0 * math.pi)


class Distribution[Act, Eval](Protocol):
    """Pure distribution mapping keys to actions and actions to evaluations."""

    def sample(self, key: jax.Array) -> Act: ...
    def evaluate(self, act: Act) -> Eval: ...
    def mode(self) -> Act: ...


@dataclass
class Evaluation:
    """Distribution statistics evaluated for a supplied action.

    Attributes:
        logp: Log-probability of the supplied action.
        entropy: Entropy of the distribution at the same batch indices.
    """

    logp: jax.Array
    entropy: jax.Array


@dataclass
class DiagNormal:
    """Diagonal Normal distribution over one vector-valued event axis.

    Attributes:
        loc: Distribution location, shaped ``(..., action_dim)``.
        log_scale: Log standard deviation, broadcastable to ``loc``.

    Public methods:
        sample: Draw an action.
        evaluate: Compute joint log-probability and entropy.
        mode: Return the distribution location.
    """

    loc: jax.Array
    log_scale: jax.Array

    def _broadcast(self) -> tuple[jax.Array, jax.Array]:
        """Broadcast location and log-scale to a common parameter shape."""
        loc, log_scale = jnp.broadcast_arrays(self.loc, self.log_scale)
        return loc, log_scale

    @staticmethod
    def _logp(
        loc: jax.Array,
        log_scale: jax.Array,
        act: jax.Array,
    ) -> jax.Array:
        """Compute joint log-probability over the final event axis."""
        standardized = (act - loc) * jnp.exp(-log_scale)
        logp = -0.5 * (standardized**2 + 2.0 * log_scale + _LOG_TWO_PI)
        return jnp.sum(logp, axis=-1)

    @staticmethod
    def _entropy(log_scale: jax.Array) -> jax.Array:
        """Compute joint entropy over the final event axis."""
        entropy = log_scale + 0.5 * (1.0 + _LOG_TWO_PI)
        return jnp.sum(entropy, axis=-1)

    def sample(self, key: jax.Array) -> jax.Array:
        """Draw an action."""
        loc, log_scale = self._broadcast()
        return loc + jnp.exp(log_scale) * jax.random.normal(key, loc.shape)

    def evaluate(self, act: jax.Array) -> Evaluation:
        """Compute joint log-probability and entropy for ``act``."""
        loc, log_scale = self._broadcast()
        return Evaluation(
            logp=self._logp(loc, log_scale, act),
            entropy=self._entropy(log_scale),
        )

    def mode(self) -> jax.Array:
        """Return the distribution location."""
        loc, _ = self._broadcast()
        return loc


@dataclass
class Categorical:
    """Categorical distribution over the final logits axis.

    Attributes:
        logits: Unnormalized logits, shaped ``(..., n_categories)``.

    Public methods:
        sample: Draw a category.
        evaluate: Compute log-probability and entropy.
        mode: Return the highest-logit category.
    """

    logits: jax.Array

    def _log_probs(self) -> jax.Array:
        """Normalize logits in log space."""
        return jax.nn.log_softmax(self.logits, axis=-1)

    @staticmethod
    def _select(log_probs: jax.Array, act: jax.Array) -> jax.Array:
        """Select the log-probability of each supplied category."""
        indices = jnp.expand_dims(act.astype(jnp.int32), axis=-1)
        return jnp.take_along_axis(log_probs, indices, axis=-1).squeeze(-1)

    def sample(self, key: jax.Array) -> jax.Array:
        """Draw a category."""
        return jax.random.categorical(key, self.logits, axis=-1)

    def evaluate(self, act: jax.Array) -> Evaluation:
        """Compute log-probability and entropy for ``act``."""
        log_probs = self._log_probs()
        probs = jax.nn.softmax(self.logits, axis=-1)
        safe_logits = jnp.where(jnp.isfinite(self.logits), self.logits, 0)
        entropy = jax.nn.logsumexp(self.logits, axis=-1) - jnp.sum(
            probs * safe_logits,
            axis=-1,
        )
        return Evaluation(
            logp=self._select(log_probs, act),
            entropy=entropy,
        )

    def mode(self) -> jax.Array:
        """Return the highest-logit category."""
        return jnp.argmax(self.logits, axis=-1)
