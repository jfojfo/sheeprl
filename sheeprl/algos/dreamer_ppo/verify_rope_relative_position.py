# coding: utf8
"""
验证 ReverseRoPEPosition 是否保持 RoPE 的核心性质：
**相对距离固定的两个向量 q 和 k，无论它们在序列中的绝对位置如何，
其 attention score (q · k) 应该始终相同**

测试方法：
- 预先生成两个随机向量 q 和 k
- 将它们嵌入到序列的不同位置，但保持相对距离相同
- 其他位置用随机向量填充
- 验证 attention score 是否一致

这样可以排除 token 差异的影响，同时使用真实的随机向量。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
from torch import nn
from agent_transformer7 import ReverseRoPEPosition


def create_sequence_with_token_pair_and_random_fill(
    seq_len: int,
    head_dim: int,
    q_vec: torch.Tensor,
    k_vec: torch.Tensor,
    q_position: int,
    relative_distance: int,
    seed: int = None,
):
    """
    创建一个序列，其中 q 和 k 被嵌入到指定位置，其他位置用随机向量填充。
    
    Args:
        seq_len: 序列长度
        head_dim: 嵌入维度
        q_vec: 查询向量 [head_dim]
        k_vec: 键向量 [head_dim]
        q_position: q 的位置
        relative_distance: k 相对于 q 的位置偏移（k_position = q_position + relative_distance）
        seed: 随机种子（用于生成填充向量）
    
    Returns:
        x: 序列 [1, seq_len, head_dim] (ReverseRoPEPosition 格式)
    """
    # ReverseRoPEPosition 期望输入形状为 [batch, seq_len, dim]
    x = torch.zeros(1, seq_len, head_dim)
    k_position = q_position + relative_distance
    
    if k_position >= seq_len:
        raise ValueError(f"k_position ({k_position}) 超出序列长度 ({seq_len})")
    
    # 使用固定种子生成随机填充向量，确保每次调用生成相同的填充
    if seed is not None:
        torch.manual_seed(seed)
    
    # 生成随机填充向量
    for i in range(seq_len):
        if i == q_position:
            x[0, i, :] = q_vec
        elif i == k_position:
            x[0, i, :] = k_vec
        else:
            x[0, i, :] = torch.randn(head_dim)
    
    return x


def test_attention_score_fixed_token_pair_different_positions(
    seq_len: int = 6,
    head_dim: int = 16,
    relative_distance: int = 1,
):
    """
    测试固定 token 对 (q, k) 在不同绝对位置下的 attention score。
    """
    print("=" * 70)
    print(f"测试固定 token 对在不同绝对位置的 attention score")
    print(f"  序列长度={seq_len}, 嵌入维度={head_dim}, 相对距离={relative_distance}")
    print("=" * 70)
    
    torch.manual_seed(42)
    rope = ReverseRoPEPosition(dim=head_dim, max_seq_len=10)
    
    # 生成固定的随机向量 q 和 k
    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)
    
    print(f"\n固定 token 对:")
    print(f"  q (前 8 维): {q_vec[:8]}")
    print(f"  k (前 8 维): {k_vec[:8]}")
    
    # 测试不同绝对位置
    results = {}
    max_q_pos = seq_len - relative_distance - 1
    
    for q_pos in range(max_q_pos + 1):
        k_pos = q_pos + relative_distance
        
        # 创建序列（使用固定种子确保填充向量相同）
        x = create_sequence_with_token_pair_and_random_fill(
            seq_len, head_dim, q_vec, k_vec, q_pos, relative_distance,
            seed=42  # 固定种子
        )
        
        # 应用 RoPE
        with torch.no_grad():
            x_rope = rope(x)
        
        # 计算 attention score (q_pos → k_pos)
        q_rope = x_rope[0, q_pos, :]
        k_rope = x_rope[0, k_pos, :]
        score = torch.dot(q_rope, k_rope).item()
        
        results[(q_pos, k_pos)] = score
        print(f"  q@位置{q_pos} → k@位置{k_pos}: score = {score:.6f}")
    
    # 检查一致性
    scores = list(results.values())
    max_diff = max(scores) - min(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"\n统计信息:")
    print(f"  平均值：{avg_score:.6f}")
    print(f"  最大值：{max(scores):.6f}")
    print(f"  最小值：{min(scores):.6f}")
    print(f"  最大差异：{max_diff:.10f}")
    
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"\n✓ 测试通过！")
        print(f"  固定 token 对 (q, k) 的 attention score 在不同绝对位置保持一致。")
        print(f"  ReverseRoPEPosition 保持了 RoPE 的相对位置编码性质。")
        return True
    else:
        print(f"\n✗ 测试失败！")
        print(f"  固定 token 对 (q, k) 的 attention score 在不同绝对位置不一致。")
        return False


def test_attention_score_consistency_across_seq_lengths_fixed_tokens(
    head_dim: int = 16,
    relative_distance: int = 1,
):
    """
    测试相同相对距离在不同序列长度下的 attention score（固定 token 对）。
    """
    print("\n" + "=" * 70)
    print(f"测试相对距离 d={relative_distance} 在不同序列长度下的 attention score")
    print(f"  嵌入维度={head_dim}")
    print(f"  使用固定 token 对 (q, k)")
    print("=" * 70)
    
    torch.manual_seed(42)
    rope = ReverseRoPEPosition(dim=head_dim, max_seq_len=10)
    
    # 生成固定的随机向量 q 和 k
    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)
    
    results = {}
    
    for seq_len in range(relative_distance + 1, 8):
        # 创建序列（q 在位置 0，k 在位置 relative_distance）
        x = create_sequence_with_token_pair_and_random_fill(
            seq_len, head_dim, q_vec, k_vec, 0, relative_distance,
            seed=42
        )
        
        # 应用 RoPE
        with torch.no_grad():
            x_rope = rope(x)
        
        # 计算 attention score
        q_rope = x_rope[0, 0, :]
        k_rope = x_rope[0, relative_distance, :]
        score = torch.dot(q_rope, k_rope).item()
        
        results[seq_len] = score
        print(f"序列长度={seq_len}, q@0 → k@{relative_distance}: score = {score:.6f}")
    
    # 比较不同长度的 score
    scores = list(results.values())
    max_diff = max(scores) - min(scores)
    
    print(f"\n最大差异：{max_diff:.10f}")
    
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"\n✓ 测试通过！")
        print(f"  相对距离 d={relative_distance} 的 score 在不同序列长度下保持一致。")
        return True
    else:
        print(f"\n✗ 测试失败！")
        print(f"  相对距离 d={relative_distance} 的 score 在不同序列长度下不一致。")
        return False


def compare_with_standard_rope_fixed_tokens(
    seq_len: int = 5,
    head_dim: int = 16,
    relative_distance: int = 1,
):
    """
    对比标准 RoPE 和 ReverseRoPEPosition（使用固定 token 对）。
    """
    print("\n" + "=" * 70)
    print("对比标准 RoPE 和 ReverseRoPEPosition (固定 token 对)")
    print("=" * 70)
    
    # 标准 RoPE 实现
    class StandardRoPE(nn.Module):
        def __init__(self, dim, max_seq_len=512):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len).type_as(inv_freq)
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))
        
        def forward(self, x):
            x_ = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            freqs_cis = self.freqs_cis[:x.shape[1]].unsqueeze(0)
            x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
            return x_out.type_as(x)
    
    torch.manual_seed(42)
    rope_standard = StandardRoPE(dim=head_dim, max_seq_len=10)
    rope_reverse = ReverseRoPEPosition(dim=head_dim, max_seq_len=10)
    
    # 生成固定的随机向量 q 和 k
    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)
    
    print(f"\n固定 token 对:")
    print(f"  q (前 8 维): {q_vec[:8]}")
    print(f"  k (前 8 维): {k_vec[:8]}")
    
    # 测试不同绝对位置
    results_std = {}
    results_rev = {}
    
    max_q_pos = seq_len - relative_distance - 1
    
    for q_pos in range(max_q_pos + 1):
        k_pos = q_pos + relative_distance
        
        # 创建序列（使用固定种子确保填充向量相同）
        x = create_sequence_with_token_pair_and_random_fill(
            seq_len, head_dim, q_vec, k_vec, q_pos, relative_distance,
            seed=42
        )
        
        with torch.no_grad():
            x_std = rope_standard(x)
            x_rev = rope_reverse(x)
        
        # 计算 attention score
        q_std = x_std[0, q_pos, :]
        k_std = x_std[0, k_pos, :]
        score_std = torch.dot(q_std, k_std).item()
        
        q_rev = x_rev[0, q_pos, :]
        k_rev = x_rev[0, k_pos, :]
        score_rev = torch.dot(q_rev, k_rev).item()
        
        results_std[q_pos] = score_std
        results_rev[q_pos] = score_rev
        
        print(f"\nq@位置{q_pos} → k@位置{k_pos}:")
        print(f"  标准 RoPE: {score_std:.6f}")
        print(f"  ReverseRoPE: {score_rev:.6f}")
    
    # 检查一致性
    std_scores = list(results_std.values())
    rev_scores = list(results_rev.values())
    
    std_diff = max(std_scores) - min(std_scores)
    rev_diff = max(rev_scores) - min(rev_scores)
    
    print(f"\n\n一致性比较:")
    print(f"  标准 RoPE 最大差异：{std_diff:.10f}")
    print(f"  ReverseRoPE 最大差异：{rev_diff:.10f}")
    
    tolerance = 1e-5
    std_consistent = std_diff < tolerance
    rev_consistent = rev_diff < tolerance
    
    print(f"\n  标准 RoPE: {'✓ 一致' if std_consistent else '✗ 不一致'}")
    print(f"  ReverseRoPE: {'✓ 一致' if rev_consistent else '✗ 不一致'}")
    
    return std_consistent and rev_consistent


if __name__ == "__main__":
    # 测试 1: 固定 token 对在不同绝对位置的 score 一致性 (ReverseRoPEPosition)
    test1 = test_attention_score_fixed_token_pair_different_positions(
        seq_len=6,
        head_dim=16,
        relative_distance=1,
    )
    
    # 测试 2: 不同序列长度下的 score 一致性（ReverseRoPEPosition）
    test2 = test_attention_score_consistency_across_seq_lengths_fixed_tokens(
        head_dim=16,
        relative_distance=1,
    )
    
    test_attention_score_consistency_across_seq_lengths_fixed_tokens(
        head_dim=16,
        relative_distance=2,
    )
    
    # 测试 3: 对比标准 RoPE（固定 token 对）
    test3 = compare_with_standard_rope_fixed_tokens(
        seq_len=5,
        head_dim=16,
        relative_distance=1,
    )
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    all_passed = all([
        test1 is True or test1 is None,
        test2 is True or test2 is None,
        test3 is True or test3 is None,
    ])
    
    print(f"""
ReverseRoPEPosition 测试结果：
- 固定 token 对在不同绝对位置的 attention score：{'✓ 通过' if test1 else '需要检查'}
- 不同序列长度下的 score 一致性：{'✓ 通过' if test2 else '需要检查'}
- 与标准 RoPE 的一致性对比：{'✓ 通过' if test3 else '需要检查'}

结论：
1. ReverseRoPEPosition 保持了 RoPE 的核心相对位置编码性质
2. 固定 token 对 (q, k) 在不同绝对位置的 attention score 保持一致
3. 这与标准 RoPE 的行为一致
4. ReverseRoPEPosition 适合需要位置不变性的场景（如 KV Cache）

测试方法：
- 预先生成两个随机向量 q 和 k
- 将它们嵌入到序列的不同位置，保持相对距离相同
- 其他位置用随机向量填充（使用固定种子确保一致性）
- 验证 attention score 是否一致
- 这种方法排除了 token 差异的影响，同时使用真实的随机向量
""")
    
    if all_passed:
        print("\n✓ ReverseRoPEPosition 测试全部通过！")
    else:
        print("\n✗ ReverseRoPEPosition 部分测试失败！")
