"""Probability-distribution values for action selection.

Distributions are pytrees of parameter arrays, typically produced by policy
heads. Their methods are pure: sampling receives an explicit random key, and
no distribution has persistent state or an initialization lifecycle.
"""

from __future__ import annotations

import math
from typing import Protocol

import jax
import jax.numpy as jnp
from chex import dataclass

_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class Sample:
    """One sampled action and its log-probability.

    Attributes:
        act: Sampled action following the distribution's batch/event shape.
        logp: Log-probability of ``act``, with shape equal to the batch shape.
    """

    act: jax.Array
    logp: jax.Array


@dataclass
class Evaluation:
    """Distribution statistics evaluated for a supplied action.

    Attributes:
        logp: Log-probability of the supplied action.
        entropy: Entropy of the distribution at the same batch indices.
    """

    logp: jax.Array
    entropy: jax.Array


class Distribution(Protocol):
    """Pure action-distribution capability over stored parameter arrays."""

    def sample(self, key: jax.Array) -> Sample: ...

    def evaluate(self, act: jax.Array) -> Evaluation: ...

    def mode(self) -> jax.Array: ...


@dataclass
class DiagNormal:
    """Diagonal Normal distribution over one vector-valued event axis.

    Attributes:
        loc: Distribution location, shaped ``(..., action_dim)``.
        log_scale: Log standard deviation, broadcastable to ``loc``.

    Public methods:
        sample: Draw an action and return its joint log-probability.
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

    def sample(self, key: jax.Array) -> Sample:
        """Draw an action and return its joint log-probability."""
        loc, log_scale = self._broadcast()
        act = loc + jnp.exp(log_scale) * jax.random.normal(key, loc.shape)
        return Sample(act=act, logp=self._logp(loc, log_scale, act))

    def evaluate(self, act: jax.Array) -> Evaluation:
        """Compute joint log-probability and entropy for ``act``."""
        loc, log_scale = self._broadcast()
        return Evaluation(
            logp=self._logp(loc, log_scale, act),
            entropy=self._entropy(log_scale),
        )

    def mode(self) -> jax.Array:
        """Return the most likely action."""
        loc, _ = self._broadcast()
        return loc


@dataclass
class Categorical:
    """Categorical distribution over the final logits axis.

    Attributes:
        logits: Unnormalized logits, shaped ``(..., n_categories)``.

    Public methods:
        sample: Draw a category and return its log-probability.
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

    def sample(self, key: jax.Array) -> Sample:
        """Draw a category and return its log-probability."""
        act = jax.random.categorical(key, self.logits, axis=-1)
        return Sample(act=act, logp=self._select(self._log_probs(), act))

    def evaluate(self, act: jax.Array) -> Evaluation:
        """Compute log-probability and entropy for ``act``."""
        log_probs = self._log_probs()
        entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
        return Evaluation(
            logp=self._select(log_probs, act),
            entropy=entropy,
        )

    def mode(self) -> jax.Array:
        """Return the highest-logit category."""
        return jnp.argmax(self.logits, axis=-1)
