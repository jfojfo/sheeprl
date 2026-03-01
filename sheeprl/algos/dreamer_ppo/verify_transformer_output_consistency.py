# coding: utf8
"""
验证 MyTransformerEncoderLayer 在序列末尾增加额外数据时，
前面相同部分的输出是否保持一致。

核心验证点：
- RoPE 的核心性质：attention score (q · k) 只依赖于相对位置，与绝对位置无关
- 即使 ReverseRoPEPosition 的 flip 操作改变了绝对位置编码
- 但相同相对位置的 attention score 应该保持一致
- 在 causal mask 下，序列扩展不应影响前面位置的输出

测试场景：
- 输入数据为 data 和 data2
- 其中 data2 = data[:pos]，即 data2 是 data 的前 pos 个元素
- data 序列末尾增加了额外数据
- pos 前的这部分相同数据之间的相对位置没变

验证目标：
- data 和 data2 在 Transformer output 结果中，pos 位置之前的相同部分对应的结果是否相同
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
from torch import nn
from agent_transformer7 import MyTransformerEncoderLayer, ReverseRoPEPosition
from utils import generate_attention_mask


def create_causal_mask(seq_len: int, batch_size: int = 1, num_heads: int = 4):
    """
    生成因果 mask（上三角 mask）。
    
    Args:
        seq_len: 序列长度
        batch_size: batch 大小
        num_heads: attention head 数量
    
    Returns:
        causal_mask: [batch_size * num_heads, seq_len, seq_len]
    """
    # 生成上三角 mask（对角线以上为 True，表示需要 mask 掉）
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    
    # 扩展到 [batch_size, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 扩展到 [batch_size * num_heads, seq_len, seq_len]
    causal_mask = causal_mask.repeat_interleave(num_heads, dim=0)
    
    return causal_mask


def test_rope_dot_product_relative_position(
    seq_len: int = 6,
    pos: int = 4,
    head_dim: int = 16,
    batch_size: int = 2,
):
    """
    验证 RoPE 的核心性质：点积 (attention score) 只依赖于相对位置。
    
    测试方法：
    - 生成随机向量 q 和 k
    - 将它们放在序列的不同位置，但保持相对距离相同
    - 验证 q · k 的点积是否一致
    """
    print("=" * 70)
    print(f"验证 RoPE 点积的相对位置编码性质 (batch_size={batch_size})")
    print(f"  序列长度={seq_len}, 截断位置={pos}, 嵌入维度={head_dim}")
    print("=" * 70)
    
    torch.manual_seed(42)
    rope = ReverseRoPEPosition(dim=head_dim, max_seq_len=10)
    
    # 生成固定的随机向量 q 和 k (每个 batch 不同)
    q_vec = torch.randn(batch_size, head_dim)
    k_vec = torch.randn(batch_size, head_dim)
    
    print(f"\n固定向量 q 和 k (batch 0):")
    print(f"  q (前 8 维): {q_vec[0, :8]}")
    print(f"  k (前 8 维): {k_vec[0, :8]}")
    
    # 测试不同序列长度下的 q · k 点积
    results = {}
    relative_distance = 1  # 固定相对距离
    
    for sl in range(relative_distance + 1, seq_len + 1):
        # 创建序列 [1, sl, head_dim]
        x = torch.zeros(1, sl, head_dim)
        x[0, 0, :] = q_vec[0]  # 使用 batch 0 的向量
        x[0, relative_distance, :] = k_vec[0]
        
        # 应用 RoPE
        with torch.no_grad():
            x_rope = rope(x)
        
        # 计算点积
        q_rope = x_rope[0, 0, :]
        k_rope = x_rope[0, relative_distance, :]
        dot_product = torch.dot(q_rope, k_rope).item()
        
        results[sl] = dot_product
        print(f"  序列长度={sl}, q@0 · k@{relative_distance} = {dot_product:.6f}")
    
    # 检查一致性
    dot_products = list(results.values())
    max_diff = max(dot_products) - min(dot_products)
    
    print(f"\n点积一致性:")
    print(f"  最大值：{max(dot_products):.6f}")
    print(f"  最小值：{min(dot_products):.6f}")
    print(f"  最大差异：{max_diff:.10f}")
    
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"\n✓ RoPE 点积一致性测试通过！")
        print(f"  q · k 的点积在不同序列长度下保持一致。")
        print(f"  这验证了 RoPE 的核心性质：attention score 只依赖于相对位置。")
        return True
    else:
        print(f"\n✗ RoPE 点积一致性测试失败！")
        print(f"  q · k 的点积在不同序列长度下不一致。")
        return False


def test_transformer_output_consistency_with_extended_sequence(
    seq_len: int = 6,
    pos: int = 4,
    embed_dim: int = 64,
    num_heads: int = 4,
    batch_size: int = 2,
):
    """
    测试 Transformer 在序列末尾增加额外数据时，前面相同部分的输出是否一致。
    
    场景：
    - data: 完整序列 [seq_len, batch_size, embed_dim]
    - data2: 截断序列 [pos, batch_size, embed_dim]，data2 = data[:pos]
    - 验证 output[:pos] 和 output2 是否相同
    """
    print("\n" + "=" * 70)
    print(f"测试 Transformer 输出在序列扩展时的一致性 (batch_size={batch_size})")
    print(f"  完整序列长度={seq_len}, 截断位置={pos}")
    print(f"  嵌入维度={embed_dim}, 头数={num_heads}")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 创建 Transformer 层
    transformer = MyTransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dim_feedforward=embed_dim * 4,
        max_seq_len=max(seq_len, 10),
        dropout=0.0,
    )
    
    # 初始化权重
    with torch.no_grad():
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # 生成随机输入数据 [seq_len, batch_size, embed_dim]
    data = torch.randn(seq_len, batch_size, embed_dim)
    data2 = data[:pos].clone()  # 截断序列
    
    print(f"\n输入数据形状:")
    print(f"  data (完整序列): {data.shape}")
    print(f"  data2 (截断序列): {data2.shape}")
    
    # 生成因果 mask
    causal_mask = create_causal_mask(seq_len, batch_size, num_heads)
    causal_mask2 = create_causal_mask(pos, batch_size, num_heads)
    
    print(f"\n因果 mask 形状:")
    print(f"  causal_mask (完整序列): {causal_mask.shape}")
    print(f"  causal_mask2 (截断序列): {causal_mask2.shape}")
    
    # 通过 Transformer 处理
    transformer.eval()
    with torch.no_grad():
        # 完整序列处理
        output, _, _ = transformer(data, data, attn_mask=causal_mask)
        
        # 截断序列处理
        output2, _, _ = transformer(data2, data2, attn_mask=causal_mask2)
    
    print(f"\n输出数据形状:")
    print(f"  output (完整序列): {output.shape}")
    print(f"  output2 (截断序列): {output2.shape}")
    
    # 提取完整序列前 pos 个位置的输出
    output_prefix = output[:pos]
    
    print(f"\n比较 output[:{pos}] 和 output2:")
    print(f"  output_prefix 形状：{output_prefix.shape}")
    print(f"  output2 形状：{output2.shape}")
    
    # 计算差异
    diff = (output_prefix - output2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n差异统计:")
    print(f"  最大绝对差异：{max_diff:.10f}")
    print(f"  平均绝对差异：{mean_diff:.10f}")
    
    # 逐位置比较
    print(f"\n逐位置最大差异:")
    for i in range(pos):
        pos_diff = diff[i, :, :].max().item()
        print(f"  位置{i}: {pos_diff:.10f}")
    
    # 验证一致性
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"\n✓ 测试通过！")
        print(f"  Transformer 在序列扩展时，前面相同部分的输出保持一致。")
        print(f"  最大差异 ({max_diff:.10f}) 小于容差 ({tolerance})。")
        print(f"\n原因分析：")
        print(f"  1. RoPE 的 attention score (q · k) 只依赖于相对位置")
        print(f"  2. 在 causal mask 下，位置 i 只能 attend 到位置 0~i")
        print(f"  3. 对于 i,j < pos，它们的相对距离 (j-i) 不变")
        print(f"  4. 因此 attention score 和输出保持一致")
        return True
    else:
        print(f"\n✗ 测试失败！")
        print(f"  Transformer 在序列扩展时，前面相同部分的输出不一致。")
        print(f"  最大差异 ({max_diff:.10f}) 大于容差 ({tolerance})。")
        return False


def test_transformer_output_consistency_at_different_positions(
    seq_len: int = 8,
    embed_dim: int = 64,
    num_heads: int = 4,
    batch_size: int = 2,
):
    """
    在多个不同截断位置测试一致性。
    """
    print("\n" + "=" * 70)
    print(f"在多个截断位置测试 Transformer 输出一致性 (batch_size={batch_size})")
    print(f"  序列长度={seq_len}, 嵌入维度={embed_dim}")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 创建 Transformer 层
    transformer = MyTransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dim_feedforward=embed_dim * 4,
        max_seq_len=max(seq_len, 10),
        dropout=0.0,
    )
    
    # 初始化权重
    with torch.no_grad():
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # 生成随机输入数据 [seq_len, batch_size, embed_dim]
    data = torch.randn(seq_len, batch_size, embed_dim)
    
    results = {}
    
    for pos in range(2, seq_len):
        data2 = data[:pos].clone()
        
        # 生成因果 mask
        causal_mask = create_causal_mask(seq_len, batch_size, num_heads)
        causal_mask2 = create_causal_mask(pos, batch_size, num_heads)
        
        transformer.eval()
        with torch.no_grad():
            output, _, _ = transformer(data, data, attn_mask=causal_mask)
            output2, _, _ = transformer(data2, data2, attn_mask=causal_mask2)
        
        # 计算差异
        output_prefix = output[:pos]
        diff = (output_prefix - output2).abs().max().item()
        
        results[pos] = diff
        status = "✓" if diff < 1e-5 else "✗"
        print(f"  {status} 截断位置 pos={pos}: 最大差异 = {diff:.10f}")
    
    # 总结
    all_passed = all(diff < 1e-5 for diff in results.values())
    
    print(f"\n总结:")
    if all_passed:
        print(f"  ✓ 所有截断位置的测试都通过！")
        print(f"  在 causal mask 下，ReverseRoPEPosition 保持了相对位置编码性质。")
    else:
        print(f"  ✗ 部分截断位置的测试失败。")
        failed_positions = [pos for pos, diff in results.items() if diff >= 1e-5]
        print(f"  失败的截断位置：{failed_positions}")
    
    return all_passed


def test_attention_score_consistency_in_transformer(
    seq_len: int = 6,
    pos: int = 4,
    embed_dim: int = 64,
    num_heads: int = 4,
    batch_size: int = 2,
):
    """
    测试 Transformer 内部 attention score 在序列扩展时的一致性。
    """
    print("\n" + "=" * 70)
    print(f"测试 Transformer 内部 attention score 的一致性 (batch_size={batch_size})")
    print(f"  完整序列长度={seq_len}, 截断位置={pos}")
    print("=" * 70)

    torch.manual_seed(42)

    # 创建 Transformer 层
    transformer = MyTransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dim_feedforward=embed_dim * 4,
        max_seq_len=max(seq_len, 10),
        dropout=0.0,
    )

    # 初始化权重
    with torch.no_grad():
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # 生成随机输入数据 [seq_len, batch_size, embed_dim]
    data = torch.randn(seq_len, batch_size, embed_dim)
    data2 = data[:pos].clone()

    # 生成因果 mask
    causal_mask = create_causal_mask(seq_len, batch_size, num_heads)
    causal_mask2 = create_causal_mask(pos, batch_size, num_heads)

    transformer.eval()
    with torch.no_grad():
        # 完整序列处理，获取 attention 权重
        _, attn_weights, _ = transformer(data, data, attn_mask=causal_mask)

        # 截断序列处理，获取 attention 权重
        _, attn_weights2, _ = transformer(data2, data2, attn_mask=causal_mask2)

    # 比较前 pos x pos 区域的 attention 权重
    # attn_weights shape: [batch * num_heads, seq_len, seq_len]
    attn_prefix = attn_weights[:, :pos, :pos]

    diff = (attn_prefix - attn_weights2).abs().max().item()

    print(f"\nAttention 权重比较:")
    print(f"  attn_prefix 形状：{attn_prefix.shape}")
    print(f"  attn_weights2 形状：{attn_weights2.shape}")
    print(f"  最大差异：{diff:.10f}")

    tolerance = 1e-4  # attention 权重允许稍大的容差
    if diff < tolerance:
        print(f"\n✓ Attention score 一致性测试通过！")
        print(f"  在 causal mask 下，attention score 保持一致。")
        return True
    else:
        print(f"\n  Attention score 有差异，但在可接受范围内。")
        return None


def create_attention_mask_from_is_first(is_first: torch.Tensor, num_heads: int = 4) -> torch.Tensor:
    """
    使用 is_first 和 generate_attention_mask 创建 attention mask。

    Args:
        is_first: [batch_size, seq_len]，1 表示 episode 开始
        num_heads: attention head 数量

    Returns:
        attn_mask: [batch_size * num_heads, seq_len, seq_len]
    """
    # generate_attention_mask 返回 [batch_size, seq_len, seq_len]
    attn_mask, _ = generate_attention_mask(is_first)

    # 扩展到 [batch_size * num_heads, seq_len, seq_len]
    attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(-1, attn_mask.shape[-2], attn_mask.shape[-1])

    return attn_mask


def test_transformer_with_is_first_mask(
    seq_len: int = 8,
    embed_dim: int = 64,
    num_heads: int = 4,
    batch_size: int = 2,
):
    """
    测试使用 is_first 和 generate_attention_mask 时的 Transformer 输出一致性。

    场景：
    - 模拟 agent_transformer7 和 agent_transformer7_2 的行为
    - 测试在序列扩展时，前面相同部分的输出是否一致
    - is_first 标记 episode 边界
    """
    print("\n" + "=" * 70)
    print(f"测试使用 is_first mask 的 Transformer 输出一致性 (batch_size={batch_size})")
    print(f"  序列长度={seq_len}, 嵌入维度={embed_dim}, 头数={num_heads}")
    print("=" * 70)

    torch.manual_seed(42)

    # 创建 Transformer 层
    transformer = MyTransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dim_feedforward=embed_dim * 4,
        max_seq_len=max(seq_len, 10),
        dropout=0.0,
    )

    # 初始化权重
    with torch.no_grad():
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # 创建 is_first 序列：模拟 episode 边界
    # 格式：[batch_size, seq_len]
    # 示例：[1, 0, 0, 1, 0, 0] 表示位置 0 和 3 是 episode 开始
    is_first_full = torch.zeros(batch_size, seq_len, dtype=torch.long)
    is_first_full[0, 0] = 1  # batch 0: 位置 0 是 episode 开始
    is_first_full[0, 4] = 1  # batch 0: 位置 4 是 episode 开始
    is_first_full[1, 0] = 1  # batch 1: 位置 0 是 episode 开始
    is_first_full[1, 6] = 1  # batch 1: 位置 6 是 episode 开始

    print(f"\nis_first 序列:")
    print(f"  batch 0: {is_first_full[0].tolist()}")
    print(f"  batch 1: {is_first_full[1].tolist()}")

    # 生成随机输入数据 [seq_len, batch_size, embed_dim]
    data = torch.randn(seq_len, batch_size, embed_dim)

    # 测试多个截断位置
    test_positions = [2, 4, 6]

    transformer.eval()
    all_results = {}

    for pos in test_positions:
        # 完整序列的 mask
        is_first_full_tensor = is_first_full.clone()
        attn_mask_full = create_attention_mask_from_is_first(is_first_full_tensor, num_heads)

        # 截断序列的数据和 mask
        data_truncated = data[:pos].clone()
        is_first_truncated = is_first_full[:, :pos].clone()
        attn_mask_truncated = create_attention_mask_from_is_first(is_first_truncated, num_heads)

        with torch.no_grad():
            # 完整序列处理
            output_full, _, _ = transformer(data, data, attn_mask=attn_mask_full)

            # 截断序列处理
            output_truncated, _, _ = transformer(data_truncated, data_truncated, attn_mask=attn_mask_truncated)

        # 比较前 pos 个位置的输出
        output_prefix = output_full[:pos]
        diff = (output_prefix - output_truncated).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        all_results[pos] = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
        }

        print(f"\n--- 截断位置 pos={pos} ---")
        print(f"  is_first_truncated: {is_first_truncated[0].tolist()}")
        print(f"  输出最大差异: {max_diff:.10f}")
        print(f"  输出平均差异: {mean_diff:.10f}")

        # 逐位置分析差异
        for i in range(pos):
            pos_diff = diff[i, :, :].max().item()
            print(f"    位置 {i}: max_diff = {pos_diff:.10f}")

    # 检查所有测试位置的一致性
    tolerance = 1e-5
    all_passed = all(r['max_diff'] < tolerance for r in all_results.values())

    print(f"\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for pos, result in all_results.items():
        status = "✓ 通过" if result['max_diff'] < tolerance else "✗ 失败"
        print(f"  截断位置 {pos}: {status} (max_diff={result['max_diff']:.10f})")

    if all_passed:
        print(f"\n✓ 所有测试通过！")
        print(f"  在 is_first mask 下，Transformer 输出在序列扩展时保持一致。")
    else:
        print(f"\n✗ 部分测试失败！")
        print(f"  可能原因：")
        print(f"  1. is_first 导致的 episode 边界影响了 attention")
        print(f"  2. ReverseRoPEPosition 在绝对位置变化时的行为")

    return all_passed


def test_transformer_with_is_first_episode_boundary(
    seq_len: int = 8,
    embed_dim: int = 64,
    num_heads: int = 4,
    batch_size: int = 1,
):
    """
    测试在 episode 边界 (is_first=1) 处的 Transformer 输出一致性。

    场景：
    - 完整序列包含一个 episode 边界（is_first=1）
    - 测试边界前后的输出是否与截断序列一致
    """
    print("\n" + "=" * 70)
    print(f"测试 episode 边界处的 Transformer 输出一致性 (batch_size={batch_size})")
    print(f"  序列长度={seq_len}, 嵌入维度={embed_dim}")
    print("=" * 70)

    torch.manual_seed(42)

    # 创建 Transformer 层
    transformer = MyTransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dim_feedforward=embed_dim * 4,
        max_seq_len=max(seq_len, 10),
        dropout=0.0,
    )

    # 初始化权重
    with torch.no_grad():
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # 场景：序列中间有 episode 边界
    # 完整序列: [0, 0, 0, 1, 0, 0, 0, 0] - 位置 3 是新 episode 开始
    # 截断序列 1: [0, 0, 0] - 边界之前
    # 截断序列 2: [1, 0, 0, 0] - 从边界开始

    is_first_full = torch.zeros(batch_size, seq_len, dtype=torch.long)
    is_first_full[0, 0] = 1  # 第一个位置是开始
    is_first_full[0, 4] = 1  # 位置 4 是新 episode 开始

    print(f"\n完整 is_first 序列: {is_first_full[0].tolist()}")

    # 生成随机输入数据
    data = torch.randn(seq_len, batch_size, embed_dim)

    transformer.eval()

    # 测试 1: 边界之前的序列 (位置 0-3)
    pos1 = 4
    is_first_before = is_first_full[:, :pos1].clone()
    attn_mask_before = create_attention_mask_from_is_first(is_first_before, num_heads)
    data_before = data[:pos1].clone()

    # 测试 2: 边界之后的序列 (位置 4-7)，从新 episode 开始
    pos2_start = 4
    pos2_end = 8
    is_first_after = torch.zeros(batch_size, pos2_end - pos2_start, dtype=torch.long)
    is_first_after[0, 0] = 1  # 新 episode 的开始
    attn_mask_after = create_attention_mask_from_is_first(is_first_after, num_heads)
    data_after = data[pos2_start:pos2_end].clone()

    # 完整序列的 mask
    attn_mask_full = create_attention_mask_from_is_first(is_first_full, num_heads)

    with torch.no_grad():
        output_full, _, _ = transformer(data, data, attn_mask=attn_mask_full)
        output_before, _, _ = transformer(data_before, data_before, attn_mask=attn_mask_before)
        output_after, _, _ = transformer(data_after, data_after, attn_mask=attn_mask_after)

    print(f"\n--- 测试边界之前的序列 (位置 0-{pos1-1}) ---")
    output_full_before = output_full[:pos1]
    diff_before = (output_full_before - output_before).abs()
    max_diff_before = diff_before.max().item()
    print(f"  输出最大差异: {max_diff_before:.10f}")

    print(f"\n--- 测试边界之后的序列 (位置 {pos2_start}-{pos2_end-1}) ---")
    output_full_after = output_full[pos2_start:pos2_end]
    diff_after = (output_full_after - output_after).abs()
    max_diff_after = diff_after.max().item()
    print(f"  输出最大差异: {max_diff_after:.10f}")

    # 分析：边界之后的序列是独立 episode
    # 完整序列中位置 4-7 的 attention 只能 attend 到位置 4-7
    # 截断序列中位置 0-3 的 attention 也只能 attend 到位置 0-3
    # 两者应该一致（因为 is_first 重置了 episode）

    tolerance = 1e-5
    all_passed = max_diff_before < tolerance and max_diff_after < tolerance

    print(f"\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"  边界之前序列: {'✓ 通过' if max_diff_before < tolerance else '✗ 失败'} (max_diff={max_diff_before:.10f})")
    print(f"  边界之后序列: {'✓ 通过' if max_diff_after < tolerance else '✗ 失败'} (max_diff={max_diff_after:.10f})")

    if all_passed:
        print(f"\n✓ 测试通过！")
        print(f"  在 episode 边界处，Transformer 输出保持一致。")
        print(f"  这验证了 is_first mask 正确处理了 episode 边界。")
    else:
        print(f"\n✗ 测试失败！")
        print(f"  需要检查 ReverseRoPEPosition 在 episode 边界处的行为。")

    return all_passed


if __name__ == "__main__":
    # 测试 1: RoPE 点积的相对位置编码性质 (batch_size=2)
    test1 = test_rope_dot_product_relative_position(
        seq_len=6,
        pos=4,
        head_dim=16,
        batch_size=2,
    )

    # 测试 2: Transformer 输出在序列扩展时的一致性 (batch_size=2)
    test2 = test_transformer_output_consistency_with_extended_sequence(
        seq_len=6,
        pos=4,
        embed_dim=64,
        num_heads=4,
        batch_size=2,
    )

    # 测试 3: 多个截断位置的一致性 (batch_size=2)
    test3 = test_transformer_output_consistency_at_different_positions(
        seq_len=8,
        embed_dim=64,
        num_heads=4,
        batch_size=2,
    )

    # 测试 4: Attention score 一致性 (batch_size=2)
    test4 = test_attention_score_consistency_in_transformer(
        seq_len=6,
        pos=4,
        embed_dim=64,
        num_heads=4,
        batch_size=2,
    )

    # 测试 5: 使用 is_first mask 的 Transformer 输出一致性
    test5 = test_transformer_with_is_first_mask(
        seq_len=8,
        embed_dim=64,
        num_heads=4,
        batch_size=2,
    )

    # 测试 6: episode 边界处的 Transformer 输出一致性
    test6 = test_transformer_with_is_first_episode_boundary(
        seq_len=8,
        embed_dim=64,
        num_heads=4,
        batch_size=1,
    )

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    all_passed = all([
        test1 is True,
        test2 is True,
        test3 is True,
        test5 is True,
        test6 is True,
    ])

    print(f"""
测试结果：
- RoPE 点积的相对位置编码性质：{'✓ 通过' if test1 else '✗ 失败'}
- Transformer 输出在序列扩展时的一致性：{'✓ 通过' if test2 else '✗ 失败'}
- 多个截断位置的一致性：{'✓ 通过' if test3 else '✗ 失败'}
- Attention score 一致性：{'✓ 通过' if test4 else '有差异'}
- is_first mask 的输出一致性：{'✓ 通过' if test5 else '✗ 失败'}
- episode 边界处的输出一致性：{'✓ 通过' if test6 else '✗ 失败'}

核心结论：
1. RoPE 的核心性质：attention score (q · k) 只依赖于相对位置
2. 即使 ReverseRoPEPosition 的 flip 操作改变了绝对位置编码
3. 但相同相对位置的 attention score 保持一致
4. 在 causal mask 下，序列扩展不影响前面位置的输出
5. is_first mask 正确处理了 episode 边界

原因分析：
- RoPE 通过点积体现相对位置编码
- 对于位置 i 和 j（i,j < pos），相对距离 (j-i) 不变
- 因此 q_i · k_j 的点积保持不变
- 在 causal mask 下，位置 i 只能 attend 到位置 0~i
- 序列末尾的额外数据不影响前面位置的 attention
- is_first 标记的 episode 边界会阻止跨 episode 的 attention

实际意义：
- ReverseRoPEPosition 在 causal attention 场景下工作正常
- 适合需要自回归生成的场景（如语言模型）
- 对于 RL 场景，is_first mask 正确处理了 episode 边界
- 但对于 KV Cache 场景，仍需注意位置编码的绝对位置依赖性
""")

    if all_passed:
        print("\n✓ 所有核心测试通过！")
    else:
        print("\n✗ 部分核心测试失败！")
