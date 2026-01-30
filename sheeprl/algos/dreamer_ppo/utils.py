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

def generate_attention_mask(is_first_seq: Tensor):
    """
    Generates attention masks for gym game sequences with episode resets indicated by 1s.
    Rule for intermediate key_padding_mask:
    - Each '1' in terminated_seq marks the start of a new episode.
    - A query in episode E can attend to keys in episode E (set to False),
      but cannot attend to keys from any previous episode Ep < E (set to True).
    - Then, the standard causal mask is applied (logical OR).

    Args:
        is_first_seq (torch.Tensor): Batch of sequences where 1 indicates episode start/reset.
                                     Shape: [batch_size, seq_len]

    Returns:
        torch.Tensor: Boolean attention mask for the batch. True means MASK OUT (ignore).
                      Shape: [batch_size, seq_len, seq_len]
        torch.Tensor: key_padding_mask: Boolean attention mask for the batch. True means MASK OUT (ignore).
                      Shape: [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = is_first_seq.shape
    device = is_first_seq.device

    # Step 1: Identify episode boundaries using cumulative sum of '1's
    # This assigns an episode ID to each position.
    # For [0, 1, 0, 1, 0], cumsum is [0, 1, 1, 2, 2].
    # For [0, 0, 0, 1, 0], cumsum is [0, 0, 0, 1, 1].
    episode_ids = torch.cumsum(is_first_seq, dim=-1)  # Shape: [batch_size, seq_len]

    # Step 2: Create index grids for queries and keys
    # query_idx: [batch_size, seq_len, 1]
    query_idx = torch.arange(seq_len, device=device).view(1, -1, 1).expand(batch_size, -1, -1)
    # key_idx: [batch_size, 1, seq_len]
    key_idx = torch.arange(seq_len, device=device).view(1, 1, -1).expand(batch_size, -1, -1)

    # episode_id_for_query: [batch_size, seq_len, 1]
    episode_id_for_query = episode_ids.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
    # episode_id_for_key: [batch_size, 1, seq_len]
    episode_id_for_key = episode_ids.unsqueeze(-2)  # Shape: [batch_size, 1, seq_len]

    # Step 3: Generate intermediate key padding mask based on episode rules
    # Rule: query can attend to key if episode_id(query) <= episode_id(key), else mask out.
    # So, mask_out = episode_id(query) > episode_id(key)
    # Shape: [batch_size, seq_len, seq_len]
    key_padding_mask = episode_id_for_query > episode_id_for_key  # True means mask out

    # Step 4: Generate standard causal mask
    # Shape: [seq_len, seq_len]
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    # Expand to batch size via broadcasting in the next step

    # Step 5: Combine intermediate mask with causal mask using logical OR
    # Broadcasting: causal_mask [seq_len, seq_len] -> [1, seq_len, seq_len] -> [batch_size, seq_len, seq_len]
    final_mask = key_padding_mask | causal_mask

    return final_mask, key_padding_mask


if __name__ == "__main__":
    is_first = torch.tensor([
        [0, 0, 0, 1, 0], # Batch 0: As per original example
        [1, 0, 0, 0, 0], # Batch 1: Terminate at start
        [0, 0, 0, 0, 0], # Batch 2: No termination
        [0, 1, 0, 1, 0], # Batch 3: Multiple terminations, first at idx 1
        [0, 0, 0, 0, 1],
    ])
    mask, key_padding_mask = generate_attention_mask(is_first)
    result_key_padding_mask = torch.tensor([[[False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True, False, False]],

        [[False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False]],

        [[False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False]],

        [[False, False, False, False, False],
         [ True, False, False, False, False],
         [ True, False, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True, False, False]],

        [[False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [ True,  True,  True,  True, False]]])
    result_mask = torch.tensor([[[False,  True,  True,  True,  True],
         [False, False,  True,  True,  True],
         [False, False, False,  True,  True],
         [ True,  True,  True, False,  True],
         [ True,  True,  True, False, False]],

        [[False,  True,  True,  True,  True],
         [False, False,  True,  True,  True],
         [False, False, False,  True,  True],
         [False, False, False, False,  True],
         [False, False, False, False, False]],

        [[False,  True,  True,  True,  True],
         [False, False,  True,  True,  True],
         [False, False, False,  True,  True],
         [False, False, False, False,  True],
         [False, False, False, False, False]],

        [[False,  True,  True,  True,  True],
         [ True, False,  True,  True,  True],
         [ True, False, False,  True,  True],
         [ True,  True,  True, False,  True],
         [ True,  True,  True, False, False]],

        [[False,  True,  True,  True,  True],
         [False, False,  True,  True,  True],
         [False, False, False,  True,  True],
         [False, False, False, False,  True],
         [ True,  True,  True,  True, False]]])

    assert (key_padding_mask == result_key_padding_mask).all()
    assert (mask == result_mask).all()
    print(key_padding_mask)
    print(mask)
