from __future__ import annotations

import copy
from typing import *
from typing import Tuple

import gymnasium
import hydra
import numpy as np
import torch
from lightning import Fabric
from torch import nn, Tensor
from torch.distributions.utils import probs_to_logits

from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state
from sheeprl.algos.dreamer_v3.agent import CNNEncoder, CNNDecoder, Actor
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
from sheeprl.models.models import MLP
from sheeprl.utils.model import ArgsType, ModuleType


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        representation_model: RepresentationModel,
        observation_model: nn.Module,
        reward_model: nn.Module,
        continue_model: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.representation_model = representation_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.continue_model = continue_model

class RepresentationModel(nn.Module):
    def __init__(
        self,
        input_dims: int,
        action_dims: int,
        stochastic_size: int,
        discrete_size: int,
        hidden_sizes: Sequence[int],
        distribution_cfg: Dict[str, Any],
        unimix: float = 0.01,
        layer_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
    ) -> None:
        super().__init__()
        self.distribution_cfg = distribution_cfg
        self.stochastic_size = stochastic_size
        self.discrete_size = discrete_size
        self.unimix = unimix
        self.latent_model = MLP(
            input_dims=input_dims,
            output_dim=stochastic_size * discrete_size,
            hidden_sizes=hidden_sizes,
            layer_args=layer_args,
            norm_layer=norm_layer,
            norm_args=norm_args,
            activation=activation,
            act_args=act_args,
        )
        self.transition_model = MLP(
            input_dims=stochastic_size * discrete_size + int(sum(action_dims)),
            output_dim=stochastic_size * discrete_size,
            hidden_sizes=hidden_sizes,
            layer_args=layer_args,
            norm_layer=norm_layer,
            norm_args=norm_args,
            activation=activation,
            act_args=act_args,
        )

    def dynamic(self, embedded_obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, stochastic_state = self._representation(embedded_obs)
        next_logits, next_stochastic_state = self._transition(logits, actions)
        return logits, stochastic_state, next_logits, next_stochastic_state

    def _representation(self, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits: Tensor = self.latent_model(embedded_obs)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete_size)

    def _transition(self, logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        mixed = torch.concat([logits, actions], -1)
        next_logits = self.transition_model(mixed)
        next_logits = self._uniform_mix(next_logits)
        return next_logits, compute_stochastic_state(next_logits, discrete=self.discrete_size)

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 2:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete_size)
        elif dim != 3:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete_size
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def imagination(self, latent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        pass

class PlayerDV3(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        latent_state_model: RepresentationModel,
        actor: Actor | nn.Module,
        actions_dim: Sequence[int],
        num_envs: int,
        stochastic_size: int,
        device: str | torch.device,
        discrete_size: int = 32,
        actor_type: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_state_model = latent_state_model
        self.actor = actor
        self.actions_dim = actions_dim
        self.num_envs = num_envs
        self.stochastic_size = stochastic_size
        self.device = device
        self.discrete_size = discrete_size
        self.actor_type = actor_type

    def get_actions(
        self,
        obs: Dict[str, Tensor],
        greedy: bool = False,
        mask: Optional[Dict[str, Tensor]] = None,
    ) -> Sequence[Tensor]:
        embedded_obs = self.encoder(obs)
        _, stochastic_state = self.latent_state_model._representation(embedded_obs)
        stochastic_state = stochastic_state.view(
            *self.stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(stochastic_state, greedy, mask)
        self.actions = torch.cat(actions, -1)
        return actions


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
) -> Tuple[WorldModel, nn.Module, nn.Module, PlayerDV3]:
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic

    latent_state_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size

    cnn_stages = int(np.log2(cfg.env.screen_size) - np.log2(4))
    cnn_encoder = CNNEncoder(
        keys=cfg.algo.cnn_keys.encoder,
        input_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cfg.algo.cnn_keys.encoder],
        image_size=obs_space[cfg.algo.cnn_keys.encoder[0]].shape[-2:],
        channels_multiplier=world_model_cfg.encoder.cnn_channels_multiplier,
        layer_norm_cls=hydra.utils.get_class(world_model_cfg.encoder.cnn_layer_norm.cls),
        layer_norm_kw=world_model_cfg.encoder.cnn_layer_norm.kw,
        activation=hydra.utils.get_class(world_model_cfg.encoder.cnn_act),
        stages=cnn_stages,
    )
    encoder = cnn_encoder

    representation_ln_cls = hydra.utils.get_class(world_model_cfg.representation_model.layer_norm.cls)
    representation_model = RepresentationModel(
        input_dims=encoder.output_dim,
        action_dims=actions_dim,
        stochastic_size=world_model_cfg.stochastic_size,
        discrete_size=world_model_cfg.discrete_size,
        hidden_sizes=[world_model_cfg.representation_model.hidden_size],
        distribution_cfg=world_model_cfg.representation_model.distribution_cfg,
        activation=hydra.utils.get_class(world_model_cfg.representation_model.dense_act),
        layer_args={"bias": representation_ln_cls == nn.Identity},
        norm_layer=[representation_ln_cls],
        norm_args=[
            {
                **world_model_cfg.representation_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.representation_model.hidden_size,
            }
        ],
    )

    cnn_decoder = CNNDecoder(
        keys=cfg.algo.cnn_keys.decoder,
        output_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cfg.algo.cnn_keys.decoder],
        channels_multiplier=world_model_cfg.observation_model.cnn_channels_multiplier,
        latent_state_size=latent_state_size,
        cnn_encoder_output_dim=cnn_encoder.output_dim,
        image_size=obs_space[cfg.algo.cnn_keys.decoder[0]].shape[-2:],
        activation=hydra.utils.get_class(world_model_cfg.observation_model.cnn_act),
        layer_norm_cls=hydra.utils.get_class(world_model_cfg.observation_model.cnn_layer_norm.cls),
        layer_norm_kw=world_model_cfg.observation_model.mlp_layer_norm.kw,
        stages=cnn_stages,
    )
    observation_model = cnn_decoder

    reward_ln_cls = hydra.utils.get_class(world_model_cfg.reward_model.layer_norm.cls)
    reward_model = MLP(
        input_dims=latent_state_size,
        output_dim=1, # world_model_cfg.reward_model.bins,
        hidden_sizes=[world_model_cfg.reward_model.dense_units] * world_model_cfg.reward_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.reward_model.dense_act),
        layer_args={"bias": reward_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=reward_ln_cls,
        norm_args={
            **world_model_cfg.reward_model.layer_norm.kw,
            "normalized_shape": world_model_cfg.reward_model.dense_units,
        },
    )

    discount_ln_cls = hydra.utils.get_class(world_model_cfg.discount_model.layer_norm.cls)
    continue_model = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[world_model_cfg.discount_model.dense_units] * world_model_cfg.discount_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.discount_model.dense_act),
        layer_args={"bias": discount_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=discount_ln_cls,
        norm_args={
            **world_model_cfg.discount_model.layer_norm.kw,
            "normalized_shape": world_model_cfg.discount_model.dense_units,
        },
    )
    world_model = WorldModel(
        encoder.apply(init_weights),
        representation_model.apply(init_weights),
        observation_model.apply(init_weights),
        reward_model.apply(init_weights),
        continue_model.apply(init_weights),
    )

    actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)
    actor: Actor = actor_cls(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        dense_units=actor_cfg.dense_units,
        activation=hydra.utils.get_class(actor_cfg.dense_act),
        mlp_layers=actor_cfg.mlp_layers,
        distribution_cfg=cfg.distribution,
        layer_norm_cls=hydra.utils.get_class(actor_cfg.layer_norm.cls),
        layer_norm_kw=actor_cfg.layer_norm.kw,
        unimix=cfg.algo.unimix,
        action_clip=actor_cfg.action_clip,
    )

    critic_ln_cls = hydra.utils.get_class(critic_cfg.layer_norm.cls)
    critic = MLP(
        input_dims=latent_state_size,
        output_dim=1, # critic_cfg.bins,
        hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
        activation=hydra.utils.get_class(critic_cfg.dense_act),
        layer_args={"bias": critic_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=critic_ln_cls,
        norm_args={
            **critic_cfg.layer_norm.kw,
            "normalized_shape": critic_cfg.dense_units,
        },
    )
    actor.apply(init_weights)
    critic.apply(init_weights)

    if cfg.algo.hafner_initialization:
        actor.mlp_heads.apply(uniform_init_weights(1.0))
        critic.model[-1].apply(uniform_init_weights(0.0))
        world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
        world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
        world_model.representation_model.latent_model.model[-1].apply(uniform_init_weights(1.0))
        world_model.representation_model.transition_model.model[-1].apply(uniform_init_weights(1.0))
        if cnn_decoder is not None:
            cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

    player = PlayerDV3(
        copy.deepcopy(world_model.encoder),
        copy.deepcopy(representation_model),
        copy.deepcopy(actor),
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        fabric.device,
        discrete_size=cfg.algo.world_model.discrete_size,
    )
    # Tie weights between the agent and the player
    for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(actor.parameters(), player.actor.parameters()):
        p.data = agent_p.data

    return world_model, actor, critic, player
