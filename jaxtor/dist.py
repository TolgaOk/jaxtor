"""Stateless probability-distribution components for action selection.

Distribution parameters are explicit input pytrees, typically produced by a
policy network elsewhere. Components contain no network parameters, random
state, or initialization lifecycle.
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


class Distribution[ParamsT](Protocol):
    """Stateless action-distribution capability.

    Implementations consume explicit parameter pytrees and therefore require
    neither persistent state nor initialization.
    """

    def sample(self, key: jax.Array, params: ParamsT) -> Sample: ...

    def evaluate(self, params: ParamsT, act: jax.Array) -> Evaluation: ...

    def mode(self, params: ParamsT) -> jax.Array: ...


@dataclass
class DiagNormal:
    """Diagonal Normal distribution over one vector-valued event axis.

    Public dataclasses:
        Params: Location and log-scale arrays with a final event axis.

    Public methods:
        sample: Draw an action and return its joint log-probability.
        evaluate: Compute joint log-probability and entropy.
        mode: Return the distribution location.
    """

    @dataclass
    class Params:
        """Parameters of a diagonal Normal distribution.

        Attributes:
            loc: Distribution location, shaped ``(..., action_dim)``.
            log_scale: Log standard deviation, broadcastable to ``loc``.
        """

        loc: jax.Array
        log_scale: jax.Array

    @staticmethod
    def _broadcast(params: DiagNormal.Params) -> tuple[jax.Array, jax.Array]:
        """Broadcast location and log-scale to a common parameter shape."""
        loc, log_scale = jnp.broadcast_arrays(params.loc, params.log_scale)
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

    def sample(self, key: jax.Array, params: DiagNormal.Params) -> Sample:
        """Draw an action and return its joint log-probability."""
        loc, log_scale = self._broadcast(params)
        act = loc + jnp.exp(log_scale) * jax.random.normal(key, loc.shape)
        return Sample(act=act, logp=self._logp(loc, log_scale, act))

    def evaluate(
        self,
        params: DiagNormal.Params,
        act: jax.Array,
    ) -> Evaluation:
        """Compute joint log-probability and entropy for ``act``."""
        loc, log_scale = self._broadcast(params)
        return Evaluation(
            logp=self._logp(loc, log_scale, act),
            entropy=self._entropy(log_scale),
        )

    def mode(self, params: DiagNormal.Params) -> jax.Array:
        """Return the most likely action."""
        loc, _ = self._broadcast(params)
        return loc


@dataclass
class Categorical:
    """Categorical distribution over the final logits axis.

    Public dataclasses:
        Params: Unnormalized logits with a final category axis.

    Public methods:
        sample: Draw a category and return its log-probability.
        evaluate: Compute log-probability and entropy.
        mode: Return the highest-logit category.
    """

    @dataclass
    class Params:
        """Parameters of a categorical distribution.

        Attributes:
            logits: Unnormalized logits, shaped ``(..., n_categories)``.
        """

        logits: jax.Array

    @staticmethod
    def _log_probs(params: Categorical.Params) -> jax.Array:
        """Normalize logits in log space."""
        return jax.nn.log_softmax(params.logits, axis=-1)

    @staticmethod
    def _select(log_probs: jax.Array, act: jax.Array) -> jax.Array:
        """Select the log-probability of each supplied category."""
        indices = jnp.expand_dims(act.astype(jnp.int32), axis=-1)
        return jnp.take_along_axis(log_probs, indices, axis=-1).squeeze(-1)

    def sample(self, key: jax.Array, params: Categorical.Params) -> Sample:
        """Draw a category and return its log-probability."""
        act = jax.random.categorical(key, params.logits, axis=-1)
        return Sample(act=act, logp=self._select(self._log_probs(params), act))

    def evaluate(
        self,
        params: Categorical.Params,
        act: jax.Array,
    ) -> Evaluation:
        """Compute log-probability and entropy for ``act``."""
        log_probs = self._log_probs(params)
        entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
        return Evaluation(
            logp=self._select(log_probs, act),
            entropy=entropy,
        )

    def mode(self, params: Categorical.Params) -> jax.Array:
        """Return the highest-logit category."""
        return jnp.argmax(params.logits, axis=-1)
