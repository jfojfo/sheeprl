from functorch.dim import Tensor

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
