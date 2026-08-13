"""Composable models, heads, acting agents, distributions, and inference.

Configured components are static, while their nested ``State`` dataclasses
mirror the dynamic child-state tree. Trainable leaves are marked by
:class:`Param`; :func:`partition` selects them only at the optimizer boundary.

Neural-library callables are adapted at the leaves. For example, given an
observation normalizer and semantic value and policy heads, an Equinox body is
partitioned before entering the component tree::

    body_eqx = eqx.nn.Sequential(
        [
            eqx.nn.Linear(obs_size, hidden_size, key=body_key),
            eqx.nn.Lambda(jax.nn.tanh),
        ]
    )
    body_params, body_static = eqx.partition(body_eqx, eqx.is_array)

    body_net = Module(static=body_static)
    body_net_state = body_net.init(body_params)
    body = NormModel(norm=obs_norm, model=body_net)
    body_state = body.init(obs_norm_state, body_net_state)

    agent = VPi(
        body=body,
        v=value_head,
        pi=policy_head,
        select=Draw(),
    )

``Module`` keeps the callable's static structure on the configured component
and its array parameters in ``Module.State``. Outer components add semantics
and mirror the same nesting in ``NormModel.State``, head states, and finally
``VPi.State``. Calling ``agent.apply`` follows that tree inward and returns the
updated state with the same structure.
"""

from jaxtor.agent.agent import Draw as Draw
from jaxtor.agent.agent import Mode as Mode
from jaxtor.agent.agent import Selector as Selector
from jaxtor.agent.agent import VPi as VPi
from jaxtor.agent.agent import VQPi as VQPi
from jaxtor.agent.dist import Categorical as Categorical
from jaxtor.agent.dist import DiagNormal as DiagNormal
from jaxtor.agent.dist import Distribution as Distribution
from jaxtor.agent.dist import Evaluation as Evaluation
from jaxtor.agent.dist import Sample as Sample
from jaxtor.agent.head import CategoricalHead as CategoricalHead
from jaxtor.agent.head import DiagNormalHead as DiagNormalHead
from jaxtor.agent.head import QHead as QHead
from jaxtor.agent.head import QsaHead as QsaHead
from jaxtor.agent.head import VHead as VHead
from jaxtor.agent.inference import VNextVInference as VNextVInference
from jaxtor.agent.inference import VPiNextVInference as VPiNextVInference
from jaxtor.agent.model import Function as Function
from jaxtor.agent.model import Model as Model
from jaxtor.agent.model import Module as Module
from jaxtor.agent.model import Normalizer as Normalizer
from jaxtor.agent.model import NormModel as NormModel
from jaxtor.agent.model import Param as Param
from jaxtor.agent.model import Partition as Partition
from jaxtor.agent.model import Transform as Transform
from jaxtor.agent.model import combine as combine
from jaxtor.agent.model import partition as partition
