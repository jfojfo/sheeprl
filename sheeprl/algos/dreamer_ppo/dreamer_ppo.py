from __future__ import annotations

import copy
import os
from functools import partial
from typing import Any, Dict, Sequence

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Fabric
import gymnasium as gym
from torch import nn, Tensor
from torch.distributions import kl_divergence, OneHotCategoricalStraightThrough, Independent
from torch.optim import Optimizer
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_ppo.agent import build_agent, WorldModel
from sheeprl.algos.dreamer_v3.utils import prepare_obs, Moments, test
from sheeprl.data.buffers import EnvIndependentReplayBuffer, ReplayBuffer
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.utils.distribution import MSEDistribution
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_logger, get_log_dir
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import save_configs, Ratio


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    world_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
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
    logits, stochastic_state, next_logits, next_stochastic_state = world_model.representation_model.dynamic(embedded_obs, actions)
    latent_states = stochastic_state.view(*stochastic_state.shape[:-2], -1)
    next_latent_states = next_stochastic_state.view(*stochastic_state.shape[:-2], -1)
    reconstructed_obs = world_model.observation_model(latent_states)

    next_embedded_obs = world_model.encoder(batch_next_obs)
    next_real_logits, _ = world_model.representation_model._representation(next_embedded_obs)

    # Compute the distribution over the reconstructed observations
    img_shape_start = 1
    po = {
        k: MSEDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[img_shape_start:]))
        for k in cfg.algo.cnn_keys.decoder
    }

    # Compute the distribution over the rewards
    pr = MSEDistribution(world_model.reward_model(latent_states), dims=1)

    # Compute the distribution over the terminal steps, if required
    pc = MSEDistribution(world_model.continue_model(latent_states), dims=1)
    continues_targets = 1 - data["terminated"]

    # KL balancing
    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=next_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=next_real_logits.detach()), 1),
    )
    kl_free_nats = cfg.algo.world_model.kl_free_nats
    kl_dynamic = cfg.algo.world_model.kl_dynamic
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)

    items = [po[k].log_prob(batch_next_obs[k]) for k in po.keys()]
    observation_loss = -sum(items)
    reward_loss = -pr.log_prob(data["rewards"])
    continue_loss = cfg.algo.world_model.continue_scale_factor * -pc.log_prob(continues_targets)
    reconstruction_loss = (observation_loss + reward_loss + continue_loss + dyn_loss).mean()

    world_optimizer.zero_grad()
    reconstruction_loss.backward()
    world_optimizer.step()
    world_model_grads = None
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_model_grads = torch.nn.utils.clip_grad_norm_(world_model.parameters(), cfg.algo.world_model.clip_gradients)

    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/world_model_loss", reconstruction_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.mean().detach())
        aggregator.update("Loss/reward_loss", reward_loss.mean().detach())
        aggregator.update("Loss/continue_loss", continue_loss.mean().detach())
        aggregator.update("Loss/state_loss", dyn_loss.mean().detach())
        aggregator.update("State/kl", kl.mean().detach())
        if world_model_grads:
            aggregator.update("Grads/world_model", world_model_grads.mean().detach())

    # Behaviour Learning
    device = fabric.device
    batch_size = cfg.algo.per_rank_batch_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    stoch_state_size = stochastic_size * discrete_size

    imagined_trajectories = torch.empty(
        cfg.algo.horizon + 1,
        batch_size,
        stoch_state_size,
        device=device,
    )
    for i in range(1, cfg.algo.horizon + 1):


#     imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
#     recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
#     imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
#     imagined_trajectories = torch.empty(
#         cfg.algo.horizon + 1,
#         batch_size * sequence_length,
#         stoch_state_size + recurrent_state_size,
#         device=device,
#     )
#     imagined_trajectories[0] = imagined_latent_state
#     imagined_actions = torch.empty(
#         cfg.algo.horizon + 1,
#         batch_size * sequence_length,
#         data["actions"].shape[-1],
#         device=device,
#     )
#     actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
#     imagined_actions[0] = actions
#
#     # The imagination goes like this, with H=3:
#     # Actions:           a'0      a'1      a'2     a'4
#     #                    ^ \      ^ \      ^ \     ^
#     #                   /   \    /   \    /   \   /
#     #                  /     \  /     \  /     \ /
#     # States:        z0 ---> z'1 ---> z'2 ---> z'3
#     # Rewards:       r'0     r'1      r'2      r'3
#     # Values:        v'0     v'1      v'2      v'3
#     # Lambda-values:         l'1      l'2      l'3
#     # Continues:     c0      c'1      c'2      c'3
#     # where z0 comes from the posterior, while z'i is the imagined states (prior)
#
#     # Imagine trajectories in the latent space
#     for i in range(1, cfg.algo.horizon + 1):
#         imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
#         imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
#         imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
#         imagined_trajectories[i] = imagined_latent_state
#         actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
#         imagined_actions[i] = actions
#
#     # Predict values, rewards and continues
#     predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean
#     predicted_rewards = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean
#     continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
#     true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
#     continues = torch.cat((true_continue, continues[1:]))
#
#     # Estimate lambda-values
#     lambda_values = compute_lambda_values(
#         predicted_rewards[1:],
#         predicted_values[1:],
#         continues[1:] * cfg.algo.gamma,
#         lmbda=cfg.algo.lmbda,
#     )
#
#     # Compute the discounts to multiply the lambda values to
#     with torch.no_grad():
#         discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma
#
#     # Actor optimization step. Eq. 11 from the paper
#     # Given the following diagram, with H=3
#     # Actions:          [a'0]    [a'1]    [a'2]    a'3
#     #                    ^ \      ^ \      ^ \     ^
#     #                   /   \    /   \    /   \   /
#     #                  /     \  /     \  /     \ /
#     # States:       [z0] -> [z'1] -> [z'2] ->  z'3
#     # Values:       [v'0]   [v'1]    [v'2]     v'3
#     # Lambda-values:        [l'1]    [l'2]    [l'3]
#     # Entropies:    [e'0]   [e'1]    [e'2]
#     actor_optimizer.zero_grad(set_to_none=True)
#     policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]
#
#     baseline = predicted_values[:-1]
#     offset, invscale = moments(lambda_values, fabric)
#     normed_lambda_values = (lambda_values - offset) / invscale
#     normed_baseline = (baseline - offset) / invscale
#     advantage = normed_lambda_values - normed_baseline
#     if is_continuous:
#         objective = advantage
#     else:
#         objective = (
#             torch.stack(
#                 [
#                     p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
#                     for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
#                 ],
#                 dim=-1,
#             ).sum(dim=-1)
#             * advantage.detach()
#         )
#     try:
#         entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
#     except NotImplementedError:
#         entropy = torch.zeros_like(objective)
#     policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
#     fabric.backward(policy_loss)
#     actor_grads = None
#     if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
#         actor_grads = fabric.clip_gradients(
#             module=actor, optimizer=actor_optimizer, max_norm=cfg.algo.actor.clip_gradients, error_if_nonfinite=False
#         )
#     actor_optimizer.step()
#
#     # Predict the values
#     qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
#     predicted_target_values = TwoHotEncodingDistribution(
#         target_critic(imagined_trajectories.detach()[:-1]), dims=1
#     ).mean
#
#     # Critic optimization. Eq. 10 in the paper
#     critic_optimizer.zero_grad(set_to_none=True)
#     value_loss = -qv.log_prob(lambda_values.detach())
#     value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
#     value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))
#
#     fabric.backward(value_loss)
#     critic_grads = None
#     if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
#         critic_grads = fabric.clip_gradients(
#             module=critic,
#             optimizer=critic_optimizer,
#             max_norm=cfg.algo.critic.clip_gradients,
#             error_if_nonfinite=False,
#         )
#     critic_optimizer.step()
#
#     # Log metrics
#     if aggregator and not aggregator.disabled:
#         aggregator.update("Loss/world_model_loss", rec_loss.detach())
#         aggregator.update("Loss/observation_loss", observation_loss.detach())
#         aggregator.update("Loss/reward_loss", reward_loss.detach())
#         aggregator.update("Loss/state_loss", state_loss.detach())
#         aggregator.update("Loss/continue_loss", continue_loss.detach())
#         aggregator.update("State/kl", kl.mean().detach())
#         aggregator.update(
#             "State/post_entropy",
#             Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
#         )
#         aggregator.update(
#             "State/prior_entropy",
#             Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
#         )
#         aggregator.update("Loss/policy_loss", policy_loss.detach())
#         aggregator.update("Loss/value_loss", value_loss.detach())
#         if world_model_grads:
#             aggregator.update("Grads/world_model", world_model_grads.mean().detach())
#         if actor_grads:
#             aggregator.update("Grads/actor", actor_grads.mean().detach())
#         if critic_grads:
#             aggregator.update("Grads/critic", critic_grads.mean().detach())
#
#     # Reset everything
#     actor_optimizer.zero_grad(set_to_none=True)
#     critic_optimizer.zero_grad(set_to_none=True)
#     world_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank

    # These arguments cannot be changed
    cfg.env.frame_stack = -1
    if 2 ** int(np.log2(cfg.env.screen_size)) != cfg.env.screen_size:
        raise ValueError(f"The screen size must be a power of 2, got: {cfg.env.screen_size}")

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            partial(
                RestartOnException,
                make_env(
                    cfg,
                    cfg.seed + rank * cfg.env.num_envs + i,
                    rank * cfg.env.num_envs,
                    log_dir if rank == 0 else None,
                    "train",
                    vector_env_idx=i,
                ),
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    if len(set(cfg.algo.cnn_keys.decoder) - set(cfg.algo.cnn_keys.encoder)) > 0:
        raise RuntimeError(
            "The CNN keys of the decoder must be contained in the encoder ones. "
            f"Those keys are decoded without being encoded: {list(set(cfg.algo.cnn_keys.decoder))}"
        )

    world_model, actor, critic, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space
    )
    world_model.to(device)
    actor.to(device)
    critic.to(device)

    world_optimizer = hydra.utils.instantiate(cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all")
    actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters(), _convert_="all")
    critic_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic.parameters(), _convert_="all")

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)
        world_model.load_state_dict(state["world_model"])
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        # moments.load_state_dict(state["moments"])

    save_configs(cfg, log_dir)

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 2
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        obs_keys=cfg.algo.cnn_keys.encoder,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=ReplayBuffer,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        rb = state["rb"]

    # Global variables
    start_iter = state["iter_num"] + 1 if cfg.checkpoint.resume_from else 1
    policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
    total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
    prefill_steps = learning_starts - int(learning_starts > 0)
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"]
        learning_starts += start_iter
        prefill_steps += start_iter

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

    # Get the first environment observation and start the optimization
    step_data = {}
    obs = envs.reset(seed=cfg.seed)[0]
    obs_keys = cfg.algo.cnn_keys.encoder
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["terminated"])

    cumulative_per_rank_gradient_steps = 0
    for iter_num in range(start_iter, total_iters + 1):
        policy_step += policy_steps_per_iter

        with torch.inference_mode():
            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                # Sample an action given the observation received by the environment
                if (
                    iter_num <= learning_starts
                    and cfg.checkpoint.resume_from is None
                    and "minedojo" not in cfg.env.wrapper._target_.lower()
                ):
                    real_actions = actions = np.array(envs.action_space.sample())
                    if not is_continuous:
                        actions = np.concatenate(
                            [
                                F.one_hot(torch.as_tensor(act), act_dim).numpy()
                                for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
                            ],
                            axis=-1,
                        )
                else:
                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None
                    real_actions = actions = player.get_actions(torch_obs, mask=mask)
                    actions = torch.cat(actions, -1).cpu().numpy()
                    if is_continuous:
                        real_actions = torch.stack(real_actions, dim=-1).cpu().numpy()
                    else:
                        real_actions = (
                            torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                        )

                step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
                rb.add(step_data, validate_args=cfg.buffer.validate_args)

                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                dones = np.logical_or(terminated, truncated).astype(np.uint8)

            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:
                for i, agent_roe in enumerate(infos["restart_on_exception"]):
                    if agent_roe and not dones[i]:
                        last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                        rb.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["terminated"][last_inserted_idx]
                        )
                        rb.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                            rb.buffer[i]["truncated"][last_inserted_idx]
                        )
                        rb.buffer[i]["is_first"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["is_first"][last_inserted_idx]
                        )
                        step_data["is_first"][i] = np.ones_like(step_data["is_first"][i])

            if cfg.metric.log_level > 0 and "final_info" in infos:
                for i, agent_ep_info in enumerate(infos["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        if aggregator and not aggregator.disabled:
                            aggregator.update("Rewards/rew_avg", ep_rew)
                            aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

            # Save the real next observation
            real_next_obs = copy.deepcopy(next_obs)
            if "final_observation" in infos:
                for idx, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        for k, v in final_obs.items():
                            real_next_obs[k][idx] = v

            for k in obs_keys:
                step_data[k] = next_obs[k][np.newaxis]

            # next_obs becomes the new obs
            obs = next_obs

            rewards = rewards.reshape((1, cfg.env.num_envs, -1))
            step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
            step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
            step_data["rewards"] = clip_rewards_fn(rewards)

            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
                reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

                # Reset already inserted step data
                step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])

        # Train the agent
        if iter_num >= learning_starts:
            ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
            per_rank_gradient_steps = ratio(ratio_steps)
            if per_rank_gradient_steps > 0:
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    local_data = rb.sample_tensors(
                        cfg.algo.per_rank_batch_size,
                        n_samples=per_rank_gradient_steps,
                        sample_next_obs=True,
                        dtype=None,
                        device=fabric.device,
                        from_numpy=cfg.buffer.from_numpy,
                    )
                    for i in range(per_rank_gradient_steps):
                        # if (
                        #     cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                        #     == 0
                        # ):
                        #     tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                        #     for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
                        #         tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                        batch = {k: v[i].float() for k, v in local_data.items()}
                        train(
                            fabric,
                            world_model,
                            actor,
                            critic,
                            world_optimizer,
                            actor_optimizer,
                            critic_optimizer,
                            batch,
                            aggregator,
                            cfg,
                            is_continuous,
                            actions_dim,
                        )
                        cumulative_per_rank_gradient_steps += 1

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or iter_num == total_iters):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio", cumulative_per_rank_gradient_steps / policy_step, policy_step
            )

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step

        # Checkpoint Model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            iter_num == total_iters and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "world_model": world_model.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                # "target_critic": target_critic.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                # "moments": moments.state_dict(),
                "ratio": ratio.state_dict(),
                "iter_num": iter_num * fabric.world_size,
                "batch_size": cfg.algo.per_rank_batch_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir, greedy=False)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "actor": actor,
            "critic": critic,
            # "target_critic": target_critic,
            # "moments": moments,
        }
        register_model(fabric, log_models, cfg, models_to_log)
