# coding: utf8
"""
验证 Transformer attention mask 在序列截断后的行为一致性。

测试场景：
- 输入数据为 data、is_first 和 data2、is_first2
- 其中 data2 = data[pos:]、is_first2 = is_first[pos:]
- is_first 在 pos 位置为 True，对应 is_first2 第一个位置元素为 True
- 通过 MyTransformerEncoderLayer(agent_transformer7.py) 对这两份数据序列进行处理后，
  由于 is_first 对 mask 的作用，处理后结果应该是一样的
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
from torch import nn
from typing import Dict, Any

from agent_transformer7 import MyTransformerEncoderLayer
from utils import generate_attention_mask


def create_test_data(seq_len: int = 5, embed_dim: int = 64, batch_size: int = 1, pos: int = 2):
    """
    创建测试数据。
    
    Args:
        seq_len: 原始序列长度
        embed_dim: 嵌入维度
        batch_size: batch 大小
        pos: 截断位置，is_first 在此位置为 True
        
    Returns:
        data: 原始数据 [seq_len, batch_size, embed_dim]
        is_first: 原始 is_first 标记 [batch_size, seq_len]
        data2: 截断后的数据 [seq_len-pos, batch_size, embed_dim]
        is_first2: 截断后的 is_first 标记 [batch_size, seq_len-pos]
        pos: 截断位置
    """
    torch.manual_seed(42)
    
    # 创建原始数据
    data = torch.randn(seq_len, batch_size, embed_dim)
    
    # 创建 is_first 标记，在 pos 位置为 True
    is_first = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    is_first[:, pos] = True  # 在 pos 位置标记为新 episode 开始
    
    # 创建截断后的数据
    data2 = data[pos:].clone()
    is_first2 = is_first[:, pos:].clone()
    
    # 确保 is_first2 的第一个位置为 True
    assert is_first2[0, 0] == True, "is_first2 的第一个位置应该为 True"
    
    print(f"原始序列长度：{seq_len}")
    print(f"截断位置：{pos}")
    print(f"截断后序列长度：{data2.shape[0]}")
    print(f"\nis_first: {is_first}")
    print(f"is_first2: {is_first2}")
    
    return data, is_first, data2, is_first2, pos


def generate_causal_mask_with_is_first(is_first: torch.Tensor, num_heads: int = 1) -> torch.Tensor:
    """
    根据 is_first 生成 attention mask。
    
    Args:
        is_first: 形状为 [batch_size, seq_len] 的布尔张量，True 表示新 episode 开始
        num_heads: attention head 数量
        
    Returns:
        attn_mask: 形状为 [batch_size * num_heads, seq_len, seq_len] 的 attention mask
    """
    batch_size, seq_len = is_first.shape
    
    # 使用 generate_attention_mask 生成 mask
    # final_mask: [batch_size, seq_len, seq_len], True 表示需要 mask 掉的位置
    final_mask, _ = generate_attention_mask(is_first)
    
    # 扩展为 multi-head 格式：[batch_size, seq_len, seq_len] -> [batch_size * num_heads, seq_len, seq_len]
    if num_heads > 1:
        final_mask = final_mask.repeat_interleave(num_heads, dim=0)
    
    return final_mask


def test_transformer_attention_mask_consistency(
    seq_len: int = 5,
    embed_dim: int = 64,
    num_heads: int = 4,
    batch_size: int = 1,
    pos: int = 2,
):
    """
    测试 Transformer 在完整序列和截断序列上的输出一致性。
    
    原理：
    - 对于完整序列，attention mask 会阻止跨 episode 的 attention
    - 对于截断序列（从新 episode 开始位置截断），由于 is_first[0]=True，
      第一个位置只能 attend 到自己
    - 因此，完整序列从 pos 开始的输出应该与截断序列的输出一致
    """
    print("=" * 60)
    print("测试 Transformer attention mask 的序列截断一致性")
    print("=" * 60)
    
    # 创建测试数据
    data, is_first, data2, is_first2, pos = create_test_data(
        seq_len=seq_len, embed_dim=embed_dim, batch_size=batch_size, pos=pos
    )
    
    # 创建 Transformer 层
    transformer = MyTransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dim_feedforward=embed_dim * 4,
        max_seq_len=seq_len,
        dropout=0.0,
    )
    
    # 初始化权重以保证确定性
    with torch.no_grad():
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # 生成 attention mask
    attn_mask = generate_causal_mask_with_is_first(is_first, num_heads=num_heads)
    attn_mask2 = generate_causal_mask_with_is_first(is_first2, num_heads=num_heads)
    
    print(f"\n完整序列的 attention mask 形状：{attn_mask.shape}")
    print(f"截断序列的 attention mask 形状：{attn_mask2.shape}")
    
    # 打印 mask 以便调试
    print("\n完整序列 attention mask (batch 0):")
    print(attn_mask[0].int())
    
    print("\n截断序列 attention mask (batch 0):")
    print(attn_mask2[0].int())
    
    # 通过 Transformer 处理
    transformer.eval()
    with torch.no_grad():
        # 完整序列处理
        output_full, _, _ = transformer(data, data, attn_mask=attn_mask)
        
        # 截断序列处理
        output_trunc, _, _ = transformer(data2, data2, attn_mask=attn_mask2)
    
    print(f"\n完整序列输出形状：{output_full.shape}")
    print(f"截断序列输出形状：{output_trunc.shape}")
    
    # 提取完整序列从 pos 位置开始的输出
    output_full_from_pos = output_full[pos:]
    
    print(f"\n完整序列从 pos={pos} 开始的输出形状：{output_full_from_pos.shape}")
    
    # 比较输出
    print("\n比较输出（完整序列从 pos 开始 vs 截断序列）:")
    print("完整序列输出 (从 pos 开始):")
    print(output_full_from_pos[:, 0, :8])  # 只显示前 8 个维度
    
    print("截断序列输出:")
    print(output_trunc[:, 0, :8])
    
    # 计算差异
    diff = (output_full_from_pos - output_trunc).abs().max()
    print(f"\n最大绝对差异：{diff.item():.10f}")
    
    # 验证一致性
    tolerance = 1e-5
    if diff.item() < tolerance:
        print(f"✓ 测试通过！差异 ({diff.item():.10f}) 小于容差 ({tolerance})")
        return True
    else:
        print(f"✗ 测试失败！差异 ({diff.item():.10f}) 大于容差 ({tolerance})")
        return False


def test_with_multiple_positions():
    """在多个不同截断位置测试一致性。"""
    print("\n" + "=" * 60)
    print("在多个截断位置测试一致性")
    print("=" * 60)
    
    seq_len = 6
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    
    all_passed = True
    
    for pos in range(1, seq_len):
        print(f"\n--- 测试截断位置 pos={pos} ---")
        passed = test_transformer_attention_mask_consistency(
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_size=batch_size,
            pos=pos,
        )
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！✓")
    else:
        print("部分测试失败！✗")
    print("=" * 60)
    
    return all_passed


def test_attention_mask_pattern():
    """测试 attention mask 的模式是否正确。"""
    print("\n" + "=" * 60)
    print("测试 attention mask 模式")
    print("=" * 60)
    
    # 创建一个简单的 is_first 序列
    is_first = torch.tensor([
        [False, False, True, False, False]  # 在位置 2 开始新 episode
    ])
    
    final_mask, key_padding_mask = generate_attention_mask(is_first)
    
    print("\nis_first:", is_first)
    print("\nkey_padding_mask (episode 边界):")
    print(key_padding_mask.int())
    
    print("\nfinal_mask (包含因果 mask):")
    print(final_mask.int())
    
    # 验证 mask 模式
    # 位置 0,1 属于 episode 0，位置 2,3,4 属于 episode 1
    # 位置 2 只能 attend 到自己（因为是 episode 1 的开始）
    # 位置 3 可以 attend 到 2,3
    # 位置 4 可以 attend 到 2,3,4
    
    expected_mask = torch.tensor([[[
        [False, True,  True,  True,  True],   # 位置 0: 只能 attend 0
        [False, False, True,  True,  True],   # 位置 1: 可以 attend 0,1
        [True,  True,  False, True,  True],   # 位置 2: 只能 attend 2 (新 episode)
        [True,  True,  False, False, True],   # 位置 3: 可以 attend 2,3
        [True,  True,  False, False, False],  # 位置 4: 可以 attend 2,3,4
    ]]])
    
    if (final_mask == expected_mask).all():
        print("\n✓ Attention mask 模式正确！")
        return True
    else:
        print("\n✗ Attention mask 模式错误！")
        print("期望:")
        print(expected_mask.int())
        return False


if __name__ == "__main__":
    # 测试 1: attention mask 模式
    test1_passed = test_attention_mask_pattern()
    
    # 测试 2: 单个截断位置
    test2_passed = test_transformer_attention_mask_consistency(
        seq_len=5,
        embed_dim=64,
        num_heads=4,
        batch_size=1,
        pos=2,
    )
    
    # 测试 3: 多个截断位置
    test3_passed = test_with_multiple_positions()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"Attention mask 模式测试：{'通过 ✓' if test1_passed else '失败 ✗'}")
    print(f"单位置一致性测试：{'通过 ✓' if test2_passed else '失败 ✗'}")
    print(f"多位置一致性测试：{'通过 ✓' if test3_passed else '失败 ✗'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n所有测试通过！✓✓✓")
    else:
        print("\n部分测试失败！✗✗✗")
