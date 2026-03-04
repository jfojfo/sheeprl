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

def generate_single_query_attention_mask(is_first_seq: Tensor) -> Tensor:
    episode_ids = torch.cumsum(is_first_seq, dim=-1)  # Shape: [batch_size, seq_len]
    episode_id_for_key = episode_ids.unsqueeze(-2)  # Shape: [batch_size, 1, seq_len]
    episode_ids_last = episode_ids[:, -1:].unsqueeze(1)  # Shape: [batch_size, 1, 1]
    key_padding_mask = episode_ids_last > episode_id_for_key  # True means mask out
    return key_padding_mask

def generate_attention_mask(is_first_seq: Tensor) -> Tuple[Tensor, Tensor]:
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

def apply_lower_triangle_mask(mask: Tensor) -> Tensor:
    """
    Applies a lower triangular mask to the attention mask, setting all elements
    in the lower-left triangle (strictly below the main diagonal) to True.

    This is useful when you want to mask out past positions within episodes
    while still respecting episode boundaries defined by the key_padding_mask.

    Args:
        mask (torch.Tensor): Boolean attention mask. Shape: [batch_size, seq_len, key_seq_len]
                             or [seq_len, key_seq_len]. True means MASK OUT (ignore).

    Returns:
        torch.Tensor: Modified attention mask with lower triangle (excluding diagonal) set to True.
                      Shape: same as input.
    """
    seq_len = mask.shape[-2]
    key_seq_len = mask.shape[-1]
    device = mask.device

    # Create lower triangular mask for (possibly non-square) matrix
    # For position (i, j), it's in lower triangle if i > j (strictly below diagonal)
    lower_tri_mask = torch.zeros((seq_len, key_seq_len), dtype=torch.bool, device=device)
    for i in range(seq_len):
        # Mark positions where j < i as True (strictly lower triangle, excluding diagonal)
        lower_tri_mask[i, : min(i, key_seq_len)] = True

    # Expand to batch size if needed
    if mask.dim() == 3:
        lower_tri_mask = lower_tri_mask.unsqueeze(0).expand(mask.shape[0], -1, -1)

    # Set lower triangle positions to True (mask out)
    # mask | lower_tri_mask: combine existing mask with lower triangle mask
    result = mask | lower_tri_mask

    return result


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

    # Test apply_lower_triangle_mask
    print("\n--- Testing apply_lower_triangle_mask (square) ---")
    processed_mask = apply_lower_triangle_mask(mask)
    print(processed_mask)

    # Verify that lower triangle (excluding diagonal) is now all True
    seq_len = mask.shape[-1]
    lower_tri_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=-1)
    for i in range(mask.shape[0]):
        # Check that lower triangle positions are True in the result
        # Only check positions where lower_tri_mask is True
        batch_lower_tri = processed_mask[i][lower_tri_mask]
        assert batch_lower_tri.all(), f"Lower triangle should be all True for batch {i}"
    print("Lower triangle mask test (square) passed!")

    # ============================================================
    # Non-square mask tests
    # ============================================================
    
    # Test 1: key_seq_len = 2 * seq_len (expanded key sequence)
    print("\n--- Testing apply_lower_triangle_mask (non-square, expanded) ---")
    key_seq_len_exp = 2 * seq_len
    # Create a non-square mask for testing
    mask_ns_exp = torch.zeros((5, seq_len, key_seq_len_exp), dtype=torch.bool)
    mask_ns_exp[:, :, :seq_len] = mask  # Copy square part
    
    print("Original expanded mask:")
    print(mask_ns_exp)
    
    processed_mask_ns_exp = apply_lower_triangle_mask(mask_ns_exp)
    print(f"Expanded mask shape: {mask_ns_exp.shape} -> {processed_mask_ns_exp.shape}")
    print("Processed expanded mask:")
    print(processed_mask_ns_exp)
    
    # Verify lower triangle is True for expanded mask
    for i in range(processed_mask_ns_exp.shape[0]):
        for row in range(seq_len):
            for col in range(min(row, key_seq_len_exp)):
                assert processed_mask_ns_exp[i, row, col], f"Position ({row}, {col}) should be True"
    print("Lower triangle mask test (non-square, expanded) passed!")

    # Test 2: key_seq_len < seq_len (truncated key sequence)
    print("\n--- Testing apply_lower_triangle_mask (non-square, truncated) ---")
    key_seq_len_short = 3
    mask_ns_short = mask[:, :, :key_seq_len_short].clone()
    
    processed_mask_ns_short = apply_lower_triangle_mask(mask_ns_short)
    print(f"Truncated mask shape: {mask_ns_short.shape} -> {processed_mask_ns_short.shape}")
    print(processed_mask_ns_short)
    
    # Verify lower triangle is True for truncated mask
    for i in range(processed_mask_ns_short.shape[0]):
        for row in range(seq_len):
            for col in range(min(row, key_seq_len_short)):
                assert processed_mask_ns_short[i, row, col], f"Position ({row}, {col}) should be True"
    print("Lower triangle mask test (non-square, truncated) passed!")
    
    print("\n=== All tests passed! ===")

    # ============================================================
    # Test generate_single_query_attention_mask
    # ============================================================
    print("\n--- Testing generate_single_query_attention_mask ---")
    
    # Test case 1: Original is_first tensor
    _, key_padding_mask = generate_attention_mask(is_first)
    single_query_mask = generate_single_query_attention_mask(is_first)
    
    # The single query mask should match the last query row of key_padding_mask
    expected_mask = key_padding_mask[:, -1:, :]
    
    print(f"single_query_mask shape: {single_query_mask.shape}")
    print(f"expected_mask shape: {expected_mask.shape}")
    print(f"single_query_mask:\n{single_query_mask}")
    print(f"expected_mask:\n{expected_mask}")
    
    assert single_query_mask.shape == expected_mask.shape, \
        f"Shape mismatch: {single_query_mask.shape} vs {expected_mask.shape}"
    assert (single_query_mask == expected_mask).all(), \
        "single_query_mask does not match the last row of key_padding_mask"
    print("Test case 1 passed!")
    
    # Test case 2: Various edge cases
    print("\n--- Testing generate_single_query_attention_mask with edge cases ---")
    
    # Edge case 1: All zeros (no episode boundaries)
    is_first_all_zeros = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    _, kpm_all_zeros = generate_attention_mask(is_first_all_zeros)
    sqm_all_zeros = generate_single_query_attention_mask(is_first_all_zeros)
    assert (sqm_all_zeros == kpm_all_zeros[:, -1:, :]).all(), \
        "Failed for all zeros case"
    print("Edge case 1 (all zeros) passed!")
    
    # Edge case 2: All ones (every position is a new episode)
    is_first_all_ones = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ])
    _, kpm_all_ones = generate_attention_mask(is_first_all_ones)
    sqm_all_ones = generate_single_query_attention_mask(is_first_all_ones)
    assert (sqm_all_ones == kpm_all_ones[:, -1:, :]).all(), \
        "Failed for all ones case"
    print("Edge case 2 (all ones) passed!")
    
    # Edge case 3: Single element sequence
    is_first_single = torch.tensor([
        [0],
        [1],
        [0],
    ])
    _, kpm_single = generate_attention_mask(is_first_single)
    sqm_single = generate_single_query_attention_mask(is_first_single)
    assert (sqm_single == kpm_single[:, -1:, :]).all(), \
        "Failed for single element case"
    print("Edge case 3 (single element) passed!")
    
    # Edge case 4: Long sequence with multiple episode boundaries
    is_first_long = torch.tensor([
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    _, kpm_long = generate_attention_mask(is_first_long)
    sqm_long = generate_single_query_attention_mask(is_first_long)
    assert (sqm_long == kpm_long[:, -1:, :]).all(), \
        "Failed for long sequence case"
    print("Edge case 4 (long sequence) passed!")
    
    # Edge case 5: Episode boundary at the last position
    is_first_last = torch.tensor([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ])
    _, kpm_last = generate_attention_mask(is_first_last)
    sqm_last = generate_single_query_attention_mask(is_first_last)
    assert (sqm_last == kpm_last[:, -1:, :]).all(), \
        "Failed for episode boundary at last position"
    print("Edge case 5 (episode at last position) passed!")
    
    # Edge case 6: Random test with multiple runs
    print("\n--- Testing generate_single_query_attention_mask with random inputs ---")
    torch.manual_seed(42)
    for trial in range(10):
        batch_size = torch.randint(1, 5, (1,)).item()
        seq_len = torch.randint(1, 20, (1,)).item()
        is_first_random = torch.randint(0, 2, (batch_size, seq_len))
        _, kpm_random = generate_attention_mask(is_first_random)
        sqm_random = generate_single_query_attention_mask(is_first_random)
        assert (sqm_random == kpm_random[:, -1:, :]).all(), \
            f"Failed for random test trial {trial}: batch_size={batch_size}, seq_len={seq_len}"
    print(f"All {trial + 1} random tests passed!")
    
    print("\n=== All generate_single_query_attention_mask tests passed! ===")
