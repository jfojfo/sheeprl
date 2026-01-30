from __future__ import annotations

import copy
from typing import *

import gymnasium
import hydra
import numpy as np
import torch
from lightning import Fabric
from torch import nn, Tensor
from torch.distributions import Independent, kl_divergence, OneHotCategoricalStraightThrough, OneHotCategorical, \
    Distribution
from torch.distributions.utils import probs_to_logits
from torch.optim import Optimizer

from sheeprl.algos.dreamer_ppo.utils import choose_latent_state
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state
from sheeprl.algos.dreamer_v3.agent import CNNEncoder, CNNDecoder, Actor
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights, Moments, compute_lambda_values
from sheeprl.models.models import MLP, LayerNorm
from sheeprl.utils.distribution import MSEDistribution, TwoHotEncodingDistribution, BernoulliSafeMode
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.utils import normalize_tensor


class NoRecurrentModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        recurrent_state_size: int,
        dense_units: int,
        activation_fn: nn.Module = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=recurrent_state_size,
            hidden_sizes=[dense_units],
            activation=activation_fn,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=[layer_norm_cls],
            norm_args=[{**layer_norm_kw, "normalized_shape": dense_units}],
        )
        # self.rnn = LayerNormGRUCell(
        #     dense_units,
        #     recurrent_state_size,
        #     bias=False,
        #     batch_first=False,
        #     layer_norm_cls=layer_norm_cls,
        #     layer_norm_kw=layer_norm_kw,
        # )
        self.recurrent_state_size = recurrent_state_size

    def forward(self, input: Tensor, recurrent_state: Tensor) -> Tensor:
        feat = self.mlp(input)
        # out = self.rnn(feat, recurrent_state)
        return feat

class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        rssm: RSSM,
        observation_model: nn.Module,
        reward_model: nn.Module,
        continue_model: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.continue_model = continue_model

class RSSM(nn.Module):
    def __init__(
        self,
        representation_model: nn.Module,
        transition_model: nn.Module,
        distribution_cfg: Dict[str, Any],
        discrete_size: int = 32,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.distribution_cfg = distribution_cfg
        self.discrete_size = discrete_size
        self.unimix = unimix

    def dynamic(self, embedded_obs: Tensor, actions: Tensor, terminated: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, stochastic_state = self._representation(embedded_obs)
        latent_state = choose_latent_state(logits, stochastic_state)
        next_logits, next_stochastic_state = self._transition(latent_state, actions, terminated)
        return logits, stochastic_state, next_logits, next_stochastic_state

    def _representation(self, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits: Tensor = self.representation_model(embedded_obs)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete_size)

    def _transition(self, latent_state: Tensor, actions: Tensor, terminated: Tensor = None) -> Tuple[Tensor, Tensor]:
        mixed = torch.concat([latent_state, actions], -1)
        if terminated is not None:
            mixed = mixed * (1 - terminated)
        next_logits = self.transition_model(mixed)
        next_logits = self._uniform_mix(next_logits)
        return next_logits, compute_stochastic_state(next_logits, discrete=self.discrete_size)

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete_size)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete_size
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def imagination(self, latent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        logits, stochastic_state = self._transition(latent_state, actions)
        return logits, stochastic_state

class PlayerDV3(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        rssm: RSSM,
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
        self.rssm = rssm
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
        logits, stochastic_state = self.rssm._representation(embedded_obs)
        latent_state = choose_latent_state(logits, stochastic_state)
        actions, _ = self.actor(latent_state, greedy, mask)
        self.actions = torch.cat(actions, -1)
        return actions

    def init_states(self, reset_envs: Optional[Sequence[int]] = None) -> None:
        pass


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
) -> Tuple[WorldModel, nn.Module, nn.Module, nn.Module, PlayerDV3]:
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
    representation_model = MLP(
        input_dims=encoder.output_dim,
        output_dim=latent_state_size,
        hidden_sizes=[world_model_cfg.representation_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.representation_model.dense_act),
        layer_args={"bias": representation_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=[representation_ln_cls],
        norm_args=[
            {
                **world_model_cfg.representation_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.representation_model.hidden_size,
            }
        ],
    )
    transition_ln_cls = hydra.utils.get_class(world_model_cfg.transition_model.layer_norm.cls)
    transition_model = MLP(
        input_dims=latent_state_size + int(sum(actions_dim)),
        output_dim=latent_state_size,
        hidden_sizes=[world_model_cfg.transition_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.transition_model.dense_act),
        layer_args={"bias": transition_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=[transition_ln_cls],
        norm_args=[
            {
                **world_model_cfg.transition_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.transition_model.hidden_size,
            }
        ],
    )
    rssm = RSSM(
        representation_model=representation_model.apply(init_weights),
        transition_model=transition_model.apply(init_weights),
        distribution_cfg=cfg.distribution,
        discrete_size=world_model_cfg.discrete_size,
        unimix=cfg.algo.unimix,
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
        output_dim=world_model_cfg.reward_model.bins,
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
        rssm,
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
        output_dim=critic_cfg.bins,
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
        world_model.rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
        world_model.rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
        world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
        world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
        if cnn_decoder is not None:
            cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

    player = PlayerDV3(
        world_model.encoder, # copy.deepcopy(world_model.encoder),
        rssm, # copy.deepcopy(rssm),
        actor, # copy.deepcopy(actor),
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        fabric.device,
        discrete_size=cfg.algo.world_model.discrete_size,
    )

    # Setup target critic with a SingleDeviceStrategy
    target_critic = copy.deepcopy(critic)

    return world_model, actor, critic, target_critic, player


def tie_player_weights(player, world_model, actor) -> None:
    # Tie weights between the agent and the player
    for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(world_model.rssm.parameters(), player.rssm.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(actor.parameters(), player.actor.parameters()):
        p.data = agent_p.data


def train_world_model(
    fabric: Fabric,
    world_model: WorldModel,
    world_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    shared_vars: Dict[str, Any],
) -> None:
    # The environment interaction goes like this:
    # Actions:           a0       a1       a2      a4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     v  /     v  /     v /
    # Observations:  o0       o1       o2       o3
    # Rewards:       0        r1       r2       r3
    # Dones:         0        d1       d2       d3
    # Is-first       1        i1       i2       i3
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_next_obs = {k: data[f"next_{k}"] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    actions = data["actions"]

    embedded_obs = world_model.encoder(batch_obs)
    logits, stochastic_state, next_prior_logits, next_prior_stochastic_state = world_model.rssm.dynamic(embedded_obs, actions, data["terminated"])
    latent_states = choose_latent_state(logits, stochastic_state)
    next_prior_latent_states = choose_latent_state(next_prior_logits, next_prior_stochastic_state)
    reconstructed_obs = world_model.observation_model(latent_states)

    next_embedded_obs = world_model.encoder(batch_next_obs)
    next_posterior_logits, _ = world_model.rssm._representation(next_embedded_obs)

    shared_vars["latent_states"] = latent_states

    # Compute the distribution over the reconstructed observations
    img_shape_start = 2
    po = {
        k: MSEDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[img_shape_start:]))
        for k in cfg.algo.cnn_keys.decoder
    }

    # Compute the distribution over the rewards
    dist_cls = MSEDistribution if cfg.algo.world_model.reward_model.bins == 1 else TwoHotEncodingDistribution
    pr = dist_cls(world_model.reward_model(latent_states), dims=1)

    # Compute the distribution over the terminal steps, if required
    pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states)), 1)
    continues_targets = 1 - data["terminated"]

    observation_loss = -sum([po[k].log_prob(batch_next_obs[k]) for k in po.keys()])
    reward_loss = -pr.log_prob(data["rewards"])

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    prior = next_prior_logits.view(*next_prior_logits.shape[:-1], stochastic_size, discrete_size)
    posterior = next_posterior_logits.view(*next_posterior_logits.shape[:-1], stochastic_size, discrete_size)

    # KL balancing
    kl_free_nats = cfg.algo.world_model.kl_free_nats
    kl_dynamic = cfg.algo.world_model.kl_dynamic
    kl_representation = cfg.algo.world_model.kl_representation

    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posterior.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=prior), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)

    repr_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posterior), 1),
        Independent(OneHotCategoricalStraightThrough(logits=prior.detach()), 1),
    )
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    kl_loss = dyn_loss + repr_loss

    continue_scale_factor = cfg.algo.world_model.continue_scale_factor
    if pc is not None and continues_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continues_targets)
    else:
        continue_loss = torch.zeros_like(reward_loss)

    kl_regularizer = cfg.algo.world_model.kl_regularizer
    reconstruction_loss = (kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss).mean()

    world_optimizer.zero_grad()
    reconstruction_loss.backward()
    world_model_grads = None
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_model_grads = torch.nn.utils.clip_grad_norm_(world_model.parameters(), cfg.algo.world_model.clip_gradients)
    world_optimizer.step()

    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/world_model_loss", reconstruction_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.mean().detach())
        aggregator.update("Loss/reward_loss", reward_loss.mean().detach())
        aggregator.update("Loss/continue_loss", continue_loss.mean().detach())
        aggregator.update("Loss/state_loss", kl_loss.mean().detach())
        aggregator.update("State/kl", kl.mean().detach())
        if world_model_grads:
            aggregator.update("Grads/world_model", world_model_grads.mean().detach())
        aggregator.update(
            "State/post_entropy",
            Independent(OneHotCategorical(logits=next_posterior_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(OneHotCategorical(logits=next_prior_logits.detach()), 1).entropy().mean().detach(),
        )


def train_ac(
    fabric: Fabric,
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    target_critic: torch.nn.Module,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    ac_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
    moments: Moments,
    shared_vars: Dict[str, Any],
) -> None:
    # Behaviour Learning
    latent_states = shared_vars['latent_states'].detach()
    device = fabric.device
    latent_states_size = latent_states.shape[-1]

    imagined_trajectories = torch.empty(
        cfg.algo.horizon + 1,
        np.prod(latent_states.shape[:2]),
        latent_states_size,
        device=device,
    )
    imagined_actions = torch.empty(
        cfg.algo.horizon + 1,
        np.prod(latent_states.shape[:2]),
        data["actions"].shape[-1],
        device=device,
    )
    imagined_latent_state = latent_states.reshape(1, -1, latent_states_size)
    actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    imagined_trajectories[0] = imagined_latent_state
    imagined_actions[0] = actions

    # The imagination goes like this, with H=3:
    # Actions:           a'0      a'1      a'2     a'4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:        z0 ---> z'1 ---> z'2 ---> z'3
    # Rewards:       r'0     r'1      r'2      r'3
    # Values:        v'0     v'1      v'2      v'3
    # Lambda-values:         l'1      l'2      l'3
    # Continues:     c0      c'1      c'2      c'3
    # where z0 comes from the posterior, while z'i is the imagined states (prior)

    # Imagine trajectories in the latent space
    for i in range(1, cfg.algo.horizon + 1):
        imagined_logits, imagined_stochastic_state = world_model.rssm.imagination(imagined_latent_state, actions)
        imagined_latent_state = choose_latent_state(imagined_logits, imagined_stochastic_state)
        imagined_trajectories[i] = imagined_latent_state
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions

    imagined_trajectories = imagined_trajectories.detach()
    imagined_actions = imagined_actions.detach()

    # Predict values, rewards and continues
    imagined_critic_values = critic(imagined_trajectories)
    dist_critic_cls = MSEDistribution if cfg.algo.critic.bins == 1 else TwoHotEncodingDistribution
    predicted_values = dist_critic_cls(imagined_critic_values, dims=1).mean

    dist_rewards_cls = MSEDistribution if cfg.algo.world_model.reward_model.bins == 1 else TwoHotEncodingDistribution
    predicted_rewards = dist_rewards_cls(world_model.reward_model(imagined_trajectories), dims=1).mean

    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
    true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
    continues = torch.cat((true_continue, continues[1:]))

    # Estimate lambda-values
    lambda_values = compute_lambda_values(
        predicted_rewards[1:],
        predicted_values[1:],
        continues[1:] * cfg.algo.gamma,
        lmbda=cfg.algo.lmbda,
    )

    # Compute the discounts to multiply the lambda values to
    with torch.no_grad():
        discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

    # Actor optimization step. Eq. 11 from the paper
    # Given the following diagram, with H=3
    # Actions:          [a'0]    [a'1]    [a'2]    a'3
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:       [z0] -> [z'1] -> [z'2] ->  z'3
    # Values:       [v'0]   [v'1]    [v'2]     v'3
    # Lambda-values:        [l'1]    [l'2]    [l'3]
    # Entropies:    [e'0]   [e'1]    [e'2]
    policies: Sequence[Distribution] = actor(imagined_trajectories)[1]

    baseline = predicted_values[:-1]
    offset, invscale = moments(lambda_values, fabric)
    normed_lambda_values = (lambda_values - offset) / invscale
    normed_baseline = (baseline - offset) / invscale
    advantage = normed_lambda_values - normed_baseline
    if is_continuous:
        objective = advantage
    else:
        objective = (
            torch.stack(
                [
                    p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                    for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                ],
                dim=-1,
            ).sum(dim=-1)
            * advantage.detach()
        )
    try:
        entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
    except NotImplementedError:
        entropy = torch.zeros_like(objective)
    policy_loss = -discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1])
    policy_loss = policy_loss.mean()

    qv = dist_critic_cls(imagined_critic_values[:-1], dims=1)
    value_loss = -qv.log_prob(lambda_values.detach())
    if target_critic is not None:
        predicted_target_values = dist_critic_cls(
            target_critic(imagined_trajectories[:-1]), dims=1
        ).mean
        value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1).detach())

    ac_loss = policy_loss + value_loss

    ac_optimizer.zero_grad()
    ac_loss.backward()
    actor_grads = None
    critic_grads = None
    ac_grads = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads = torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'))
        critic_grads = torch.nn.utils.clip_grad_norm_(critic.parameters(), float('inf'))
        ac_grads = torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.algo.actor.clip_gradients)
    ac_optimizer.step()

    if aggregator and not aggregator.disabled:
        if actor_grads:
            aggregator.update("Grads/actor", actor_grads.mean().detach())
        if critic_grads:
            aggregator.update("Grads/critic", critic_grads.mean().detach())
        if ac_grads:
            aggregator.update("Grads/ac", ac_grads.mean().detach())
        aggregator.update("Loss/policy_loss", policy_loss.detach())
        aggregator.update("Loss/value_loss", value_loss.detach())


# 数据顺序：前一个状态经过动作后进入到当前状态、当前done、当前reward，action是当前状态进入到下一个状态的action
# 返回前n-1个数的gae，最后一个无法计算，因为没有下一个状态
# dim: [seq_len, batch, *]
@torch.no_grad()
def compute_gae_with_dreamerv3(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    advantages: Tensor = torch.zeros_like(rewards)[:-1]
    gae = 0
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t+1] + values[t+1] * continues[t+1] * gamma - values[t]
        gae = delta + gae * continues[t+1] * gamma * gae_lambda
        advantages[t] = gae
    returns = advantages + values[:-1]
    return returns, advantages


def train_ac_with_ppo(
    fabric: Fabric,
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    target_critic: torch.nn.Module,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    ac_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
    moments: Moments,
    shared_vars: Dict[str, Any],
) -> None:
    # Behaviour Learning
    latent_states = shared_vars['latent_states']
    device = fabric.device
    batch_size = cfg.algo.per_rank_batch_size
    latent_states_size = latent_states.shape[-1]
    clip_param = cfg.algo.e_clip

    imagined_trajectories = torch.empty(
        cfg.algo.horizon + 1,
        batch_size,
        latent_states_size,
        device=device,
    )
    imagined_actions = torch.empty(
        cfg.algo.horizon + 1,
        batch_size,
        data["actions"].shape[-1],
        device=device,
    )
    imagined_latent_state = latent_states
    actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    imagined_trajectories[0] = imagined_latent_state
    imagined_actions[0] = actions

    # The imagination goes like this, with H=3:
    # Actions:           a'0      a'1      a'2     a'4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:        z0 ---> z'1 ---> z'2 ---> z'3
    # Rewards:       r'0     r'1      r'2      r'3
    # Values:        v'0     v'1      v'2      v'3
    # Lambda-values:         l'1      l'2      l'3
    # Continues:     c0      c'1      c'2      c'3
    # where z0 comes from the posterior, while z'i is the imagined states (prior)

    # Imagine trajectories in the latent space
    for i in range(1, cfg.algo.horizon + 1):
        imagined_logits, imagined_stochastic_state = world_model.rssm.imagination(imagined_latent_state, actions)
        imagined_latent_state = choose_latent_state(imagined_logits, imagined_stochastic_state)
        imagined_trajectories[i] = imagined_latent_state
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions
    imagined_trajectories, imagined_actions = imagined_trajectories.detach(), imagined_actions.detach()

    # Predict values, rewards and continues
    predicted_values = critic(imagined_trajectories)
    predicted_rewards = world_model.reward_model(imagined_trajectories)
    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode

    return_, _ = compute_gae_with_dreamerv3(predicted_rewards, predicted_values, continues, cfg.algo.gamma, cfg.algo.lmbda)
    advantage = normalize_tensor(return_ - predicted_values[:-1])
    return_, advantage = return_.detach(), advantage.detach()
    policies: Sequence[Distribution] = actor(imagined_trajectories)[1]
    log_probs = [
        p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1].detach()
        for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
    ]

    actor_grads, critic_grads, ac_grads = None, None, None
    for i in range(10):
        value = critic(imagined_trajectories)
        policies: Sequence[Distribution] = actor(imagined_trajectories)[1]
        actor_loss, entropy_loss = 0, 0
        for p, imgnd_act, old_log_probs in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1), log_probs):
            new_log_probs = p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
            ratio = (new_log_probs - old_log_probs).exp()  # new_prob/old_prob
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss += -torch.min(surr1, surr2)
            entropy_loss += p.entropy()[:-1]
        actor_loss, entropy_loss = actor_loss.mean(), entropy_loss.mean()
        policy_loss = actor_loss - 0.01 * entropy_loss

        qv = MSEDistribution(value[:-1], dims=1)
        value_loss = -qv.log_prob(return_)
        value_loss = 0.5 * torch.mean(value_loss)
        ac_loss = value_loss + policy_loss
        ac_optimizer.zero_grad()
        ac_loss.backward()
        if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
            actor_grads = torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'))
            critic_grads = torch.nn.utils.clip_grad_norm_(critic.parameters(), float('inf'))
            ac_grads = torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.algo.actor.clip_gradients)
        ac_optimizer.step()

    if aggregator and not aggregator.disabled:
        if actor_grads:
            aggregator.update("Grads/actor", actor_grads.mean().detach())
        if critic_grads:
            aggregator.update("Grads/critic", critic_grads.mean().detach())
        if ac_grads:
            aggregator.update("Grads/ac", ac_grads.mean().detach())
        aggregator.update("Loss/policy_loss", policy_loss.detach())
        aggregator.update("Loss/value_loss", value_loss.detach())


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    target_critic: torch.nn.Module,
    world_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    ac_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
    moments: Moments,
) -> None:
    assert (actor_optimizer is not None and critic_optimizer is not None and ac_optimizer is None
            or actor_optimizer is None and critic_optimizer is None and ac_optimizer is not None)
    shared_vars = {}
    train_world_model(fabric, world_model, world_optimizer, data, aggregator, cfg, shared_vars)
    train_ac(fabric, world_model, actor, critic, target_critic, actor_optimizer, critic_optimizer, ac_optimizer, data, aggregator, cfg, is_continuous, actions_dim, moments, shared_vars)
    # train_ac_with_ppo(fabric, world_model, actor, critic, target_critic, ac_optimizer, data, aggregator, cfg, is_continuous, actions_dim, moments, shared_vars)
    shared_vars.clear()

    # Reset everything
    world_optimizer.zero_grad()
    if actor_optimizer is not None:
        actor_optimizer.zero_grad()
    if critic_optimizer is not None:
        critic_optimizer.zero_grad()
    if ac_optimizer is not None:
        ac_optimizer.zero_grad()
