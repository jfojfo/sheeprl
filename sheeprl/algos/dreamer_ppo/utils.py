# coding: utf8
from typing import Tuple

import torch
from torch import Tensor

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
    "Grads/ac",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic"}

def choose_latent_state(logits: Tensor, stochastic_state: Tensor) -> Tensor:
    return stochastic_state.view(*stochastic_state.shape[:-2], -1)

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
