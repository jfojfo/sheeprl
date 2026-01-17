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
    # "State/post_entropy",
    # "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic"}
