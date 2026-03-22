from __future__ import annotations

import copy
from typing import *

import gymnasium
import hydra
import numpy as np
import torch
from lightning import Fabric
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Independent, kl_divergence, OneHotCategoricalStraightThrough, OneHotCategorical, \
    Distribution
from torch.distributions.utils import probs_to_logits
from torch.optim import Optimizer

from sheeprl.algos.dreamer_ppo.utils import choose_latent_state, generate_attention_mask, \
    generate_single_query_attention_mask
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state
from sheeprl.algos.dreamer_v3.agent import CNNEncoder, CNNDecoder, Actor
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights, compute_lambda_values, Moments
from sheeprl.models.models import MLP
from sheeprl.utils.distribution import MSEDistribution, TwoHotEncodingDistribution, BernoulliSafeMode
from sheeprl.utils.metric import MetricAggregator


class ReverseRoPEPosition(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        base = max_seq_len * 10

        # 使用更高精度的频率计算
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

        # 使用高精度的时间步长计算
        t = torch.arange(max_seq_len, dtype=torch.float64)
        freqs = torch.outer(t, inv_freq)

        # 使用高精度计算复数表示
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)

        # 存储cos和sin分量以提高数值稳定性
        self.register_buffer("cos_cached", cos_freqs)
        self.register_buffer("sin_cached", sin_freqs)

    def _rotate_half(self, x):
        """旋转后半部分向量"""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        # 获取序列长度
        seq_len = x.shape[-2]
        assert seq_len <= self.cos_cached.shape[0], "Sequence length exceeds max_seq_len"

        # 获取对应位置的cos和sin值
        cos_m = self.cos_cached[-seq_len:].to(x.dtype)
        sin_m = self.sin_cached[-seq_len:].to(x.dtype)

        # 重塑x以便进行广播操作
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]

        # 应用RoPE变换
        # 对于每个维度对 (x_i, x_{i+1})，应用旋转矩阵
        # [cos -sin; sin cos] * [x_i; x_{i+1}] = [cos*x_i - sin*x_{i+1}; sin*x_i + cos*x_{i+1}]
        cos_part = cos_m * x_even - sin_m * x_odd
        sin_part = sin_m * x_even + cos_m * x_odd

        # 组合结果
        result = torch.stack([cos_part, sin_part], dim=-1)

        # 重塑回原始形状
        output = result.view(*x.shape)
        return output

# class ReverseRoPEPosition(nn.Module):
#     def __init__(self, dim, max_seq_len=512):
#         super().__init__()
#         base = max_seq_len * 10
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
#         t = torch.arange(max_seq_len).type_as(inv_freq)
#         freqs = torch.outer(t, inv_freq)
#         # freqs_cis: complex - (seq_len, head_dim / 2)
#         self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))
#
#     def _reshape_for_broadcast(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         freqs_cis: complex - (seq_len, head_dim / 2)
#         x: complex - (bsz, seq_len, head_dim / 2)
#         """
#         ndim = x.ndim
#         freqs_cis = self.freqs_cis[:x.shape[1]]
#         assert 1 < ndim
#         assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
#             freqs_cis.shape,
#             (x.shape[1], x.shape[-1]),
#         )
#         shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#         return freqs_cis.view(*shape)
#
#     def forward(self, x):
#         # x = x.flip(dims=[1])
#         x_ = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
#         freqs_cis = self._reshape_for_broadcast(x_)
#         x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
#         # x_out = x_out.flip(dims=[1])
#         return x_out.type_as(x)

    # def forward_qk(self, xq, xk):
    #     """
    #     Args:
    #         xq (Tensor): Query tensor, shape (batch_size, seq_len, embed_dim).
    #         xk (Tensor): Key tensor, shape (batch_size, seq_len, embed_dim).
    #     Returns:
    #         Tuple[Tensor, Tensor]: RoPE-applied Query and Key tensors, same shapes as inputs.
    #     """
    #     xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    #     xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    #     freqs_cis = self._reshape_for_broadcast(xq_)
    #     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    #     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    #     return xq_out.type_as(xq), xk_out.type_as(xk)

class MySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len=512, dropout=None):
        super(MySelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        assert self.head_dim * self.num_heads == self.embed_dim

        # Linear transformations for Query, Key, and Value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = ReverseRoPEPosition(self.head_dim, max_seq_len=max_seq_len)

        # Output linear transformation
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, kv_cache=None, attn_mask=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=0)
            v = torch.cat([v_cache, v], dim=0)
            kv_cache = (k, v)

        # Reshape Query, Key, and Value tensors
        tgt_len, batch_size, _ = q.size()
        q = q.view(tgt_len, batch_size * self.num_heads, -1).transpose(0, 1)

        seq_len, batch_size, _ = k.size()
        k = k.view(seq_len, batch_size * self.num_heads, -1).transpose(0, 1)

        seq_len, batch_size, _ = v.size()
        v = v.view(seq_len, batch_size * self.num_heads, -1).transpose(0, 1)

        q_rope, k_rope = self.rope(q), self.rope(k)

        # Compute scaled dot-product attention
        attn_weights = torch.bmm(q_rope, k_rope.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            # a a a, b b b, c c c
            # attn_mask = attn_mask.repeat_interleave(repeats=self.num_heads, dim=0)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(-1, *attn_mask.shape[-2:])
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))
            attn_weights += attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout is not None:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.bmm(attn_weights, v)
        output = output.transpose(0, 1).contiguous()
        output = output.view(tgt_len, batch_size, self.embed_dim)
        output = self.out_proj(output)
        return output, attn_weights, kv_cache

class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=1024, max_seq_len=512, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1    
        """
        # self.self_attn = MySelfAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn = MySelfAttention(embed_dim, num_heads, max_seq_len=max_seq_len, dropout=None)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.activation = F.silu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                # 对于权重矩阵，使用 Xavier Uniform
                nn.init.xavier_uniform_(p)
            else:
                # 对于偏置，初始化为 0
                nn.init.zeros_(p)

    def forward(self, q_input, kv_input, kv_cache=None, attn_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param src_mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return: # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        src = q_input
        norm_src = self.norm1(src)  # [src_len,batch_size,num_heads*kdim]
        if kv_input is q_input:
            norm_src_kv = norm_src
        else:
            norm_src_kv = self.norm1(kv_input)
        attn_output, w, kv_cache = self.self_attn(norm_src, norm_src_kv, norm_src_kv,
                                        kv_cache=kv_cache,
                                        attn_mask=attn_mask)
        # src2: [src_len,batch_size,num_heads*kdim] num_heads*kdim = embed_dim
        src2 = src + self.dropout1(attn_output)  # 残差连接

        norm_src2 = self.norm2(src2)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(norm_src2))))
        output = src2 + self.dropout2(ffn_output)
        return output, w, kv_cache  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


class WorldModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any],
                 actions_dim: Sequence[int],
                 obs_space: gymnasium.spaces.Dict,
                 is_continuous: bool,
                 learnable_initial_recurrent_state: bool = True):
        super().__init__()
        self.cfg = cfg
        self.discrete_size = cfg.algo.world_model.discrete_size
        self.unimix = cfg.algo.unimix

        world_model_cfg = cfg.algo.world_model
        stoch_state_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
        latent_state_size = stoch_state_size + world_model_cfg.transformer.embed_dim

        if learnable_initial_recurrent_state:
            self.initial_recurrent_state = nn.Parameter(
                torch.zeros(world_model_cfg.transformer.embed_dim, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_recurrent_state", torch.zeros(world_model_cfg.transformer.embed_dim, dtype=torch.float32)
            )

        # smaller cnn_channels_multiplier 4 or 8
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
            output_dim=stoch_state_size,
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
        embed_action = nn.Linear(int(sum(actions_dim)), world_model_cfg.transformer.action_embed_dim)
        embed_state = nn.Linear(stoch_state_size, world_model_cfg.transformer.state_embed_dim)
        transformer = MyTransformerEncoderLayer(
            world_model_cfg.transformer.embed_dim,
            world_model_cfg.transformer.num_heads,
            world_model_cfg.transformer.embed_dim * 4,
            max_seq_len=cfg.algo.per_rank_sequence_length,
        )

        transition_ln_cls = hydra.utils.get_class(world_model_cfg.transition_model.layer_norm.cls)
        transition_model = MLP(
            input_dims=world_model_cfg.transformer.embed_dim,
            output_dim=stoch_state_size,
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

        self.encoder = encoder.apply(init_weights)
        self.representation_model = representation_model.apply(init_weights)
        self.embed_action = embed_action
        self.embed_state = embed_state
        self.transformer = transformer
        self.transition_model = transition_model.apply(init_weights)
        self.observation_model = observation_model.apply(init_weights)
        self.reward_model = reward_model.apply(init_weights)
        self.continue_model = continue_model.apply(init_weights)

        if cfg.algo.hafner_initialization:
            representation_model.model[-1].apply(uniform_init_weights(1.0))
            transition_model.model[-1].apply(uniform_init_weights(1.0))
            reward_model.model[-1].apply(uniform_init_weights(0.0))
            continue_model.model[-1].apply(uniform_init_weights(1.0))
            if cnn_decoder is not None:
                cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

    def get_initial_states(self, batch_shape: Sequence[int] | torch.Size) -> Tensor:
        initial_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        return initial_state
        # initial_posterior = self._transition(initial_state)[1]
        # return initial_state, initial_posterior

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

    def dynamic(self, embedded_obs: Tensor, actions: Tensor, is_first: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        posterior_logits, posterior_stochastic_state = self._representation(embedded_obs)
        attn_output, _ = self._attn_output(posterior_stochastic_state, actions, is_first)

        attn_output = torch.cat((torch.zeros_like(attn_output[:1]), attn_output[:-1]), dim=0)
        # 对于is_first为1的位置，重置attn_output为初始状态
        is_first_mask = is_first.squeeze(-1).bool()
        if is_first_mask.any():
            initial_state = self.get_initial_states([])
            attn_output[is_first_mask] = initial_state

        prior_logits, prior_stochastic_state = self._transition(attn_output)
        return posterior_logits, posterior_stochastic_state, prior_logits, prior_stochastic_state, attn_output

    def latent(self, stochastic_state: Tensor, attn_output: Tensor) -> Tensor:
        posterior = stochastic_state.view(*stochastic_state.shape[:-2], -1)
        latent_state = torch.cat([posterior, attn_output], dim=-1)
        return latent_state

    def _attn_output(self, stochastic_state: Tensor, actions: Tensor, is_first: Tensor) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        posterior = stochastic_state.view(*stochastic_state.shape[:-2], -1)

        embed_seq = torch.cat([self.embed_state(posterior), self.embed_action(actions)], dim=-1)
        attn_mask, _ = generate_attention_mask(is_first.squeeze(-1).transpose(0, 1))

        # Pass empty kv_cache to collect k, v for all positions
        batch_size, embed_dim = embed_seq.shape[1], embed_seq.shape[-1]
        empty_kv_cache = (
            torch.empty(0, batch_size, embed_dim, device=embed_seq.device, dtype=embed_seq.dtype),
            torch.empty(0, batch_size, embed_dim, device=embed_seq.device, dtype=embed_seq.dtype),
        )
        attn_output, _, kv_cache = self.transformer(embed_seq, embed_seq, kv_cache=empty_kv_cache, attn_mask=attn_mask)
        return attn_output, kv_cache

    def _representation(self, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.representation_model(embedded_obs)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete_size)

    def _transition(self, attn_state: Tensor) -> Tuple[Tensor, Tensor]:
        next_logits = self.transition_model(attn_state)
        next_logits = self._uniform_mix(next_logits)
        return next_logits, compute_stochastic_state(next_logits, discrete=self.discrete_size)

    def imagination(self, stochastic_state: Tensor, action: Tensor, kv_cache: tuple[Tensor, Tensor], is_first: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        prior = stochastic_state.view(*stochastic_state.shape[:-2], -1)
        embed = torch.cat([self.embed_state(prior), self.embed_action(action)], dim=-1)
        attn_mask = generate_single_query_attention_mask(is_first.squeeze(-1).transpose(0, 1))
        attn_output, _, kv_cache = self.transformer(embed, embed, kv_cache=kv_cache, attn_mask=attn_mask)
        logits, stochastic_state = self._transition(attn_output)
        return logits, stochastic_state, attn_output, kv_cache


class PlayerDV3(nn.Module):
    def __init__(
        self,
        world_model: WorldModel,
        actor: Actor,
        actions_dim: Sequence[int],
        cfg: Dict[str, Any],
        device: str | torch.device,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.actor = actor
        self.actions_dim = actions_dim
        self.cfg = cfg
        self.device = device
        self.seq_obs = None
        self.seq_is_first = None
        self.seq_action = None

    @torch.no_grad()
    def init_states(self, reset_envs: Optional[Sequence[int]] = None) -> None:
        pass

    def get_actions(
        self,
        obs: Dict[str, Tensor],
        is_first: Tensor = None,
        greedy: bool = False,
        mask: Optional[Dict[str, Tensor]] = None,
    ) -> Sequence[Tensor]:
        cfg = self.cfg
        seq_len = cfg.algo.per_rank_sequence_length
        num_envs = cfg.env.num_envs
        if self.seq_obs is None:
            self.seq_obs = {k: torch.empty(0, num_envs, *obs[k].shape[2:]).to(self.device) for k in obs}
            self.seq_is_first = torch.empty(0, num_envs, 1).to(self.device)
            self.seq_action = torch.empty(0, num_envs, sum(self.actions_dim)).to(self.device)
        self.seq_obs = {k: torch.cat([self.seq_obs[k], obs[k]], dim=0)[-seq_len:] for k in obs}
        self.seq_is_first = torch.cat([self.seq_is_first, is_first], dim=0)[-seq_len:]

        seq_embedded_obs = self.world_model.encoder(self.seq_obs)
        logits, stochastic_state = self.world_model._representation(seq_embedded_obs)
        if self.seq_action.shape[0] == 0:
            attn_output = self.world_model.get_initial_states([1, num_envs])
        else:
            attn_output, _ = self.world_model._attn_output(stochastic_state[:-1], self.seq_action, self.seq_is_first[:-1])

        is_first_mask = self.seq_is_first[1:].squeeze(-1).bool()
        if is_first_mask.any():
            initial_state = self.world_model.get_initial_states([])
            attn_output[is_first_mask] = initial_state

        latent_state = self.world_model.latent(stochastic_state[-1:], attn_output[-1:])

        actions, _ = self.actor(latent_state[-1:], greedy, mask)
        self.seq_action = torch.cat([self.seq_action, torch.cat(actions, dim=-1)], dim=0)[-(seq_len-1):]
        return actions


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

    stochastic_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    world_model_cfg.transformer.embed_dim = world_model_cfg.transformer.state_embed_dim + world_model_cfg.transformer.action_embed_dim
    latent_state_size = stochastic_size + world_model_cfg.transformer.embed_dim

    world_model = WorldModel(cfg, actions_dim, obs_space, is_continuous,
                             learnable_initial_recurrent_state=cfg.algo.world_model.learnable_initial_recurrent_state)

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
        output_dim=critic_cfg.bins, # critic_cfg.bins,
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

    player = PlayerDV3(
        world_model,
        actor,
        actions_dim,
        cfg,
        fabric.device,
    )

    # Setup target critic with a SingleDeviceStrategy
    target_critic = copy.deepcopy(critic)

    return world_model, actor, critic, target_critic, player


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
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
    # The environment interaction goes like this:
    # Actions:           a0       a1       a2      a4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     v  /     v  /     v /
    # Observations:  o0       o1       o2       o3
    # Rewards:       0        r1       r2       r3
    # Dones:         0        d1       d2       d3
    # Is-first       1        i1       i2       i3

    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    # batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size

    # Embed observations from the environment
    embedded_obs = world_model.encoder(batch_obs)
    posterior_logits, posterior_stochastic_state, prior_logits, _, attn_output = world_model.dynamic(
        embedded_obs, data["actions"], data["is_first"])

    # latent_states = torch.cat([posterior, attn_output], dim=-1)
    latent_states = world_model.latent(posterior_stochastic_state, attn_output)
    reconstructed_obs = world_model.observation_model(latent_states)

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

    observation_loss = -sum([po[k].log_prob(batch_obs[k]) for k in po.keys()])
    reward_loss = -pr.log_prob(data["rewards"])

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    prior_logits = prior_logits.view(*prior_logits.shape[:-1], stochastic_size, discrete_size)
    posterior_logits = posterior_logits.view(*posterior_logits.shape[:-1], stochastic_size, discrete_size)

    # KL balancing
    kl_free_nats = cfg.algo.world_model.kl_free_nats
    kl_dynamic = cfg.algo.world_model.kl_dynamic
    kl_representation = cfg.algo.world_model.kl_representation

    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posterior_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)

    repr_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posterior_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1),
    )
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    # kl_loss = (dyn_loss + repr_loss) * continues_targets[:-1].squeeze(-1) # exclude state(terminated -> is_first)
    kl_loss = dyn_loss + repr_loss

    continue_scale_factor = cfg.algo.world_model.continue_scale_factor
    if pc is not None and continues_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continues_targets)
    else:
        continue_loss = torch.zeros_like(reward_loss)

    kl_regularizer = cfg.algo.world_model.kl_regularizer
    # kl_loss seq dim is minus 1
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
            Independent(OneHotCategorical(logits=posterior_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(OneHotCategorical(logits=prior_logits.detach()), 1).entropy().mean().detach(),
        )


    # Behaviour Learning
    latent_states_size = latent_states.shape[-1]
    new_batch_size = np.prod(latent_states.shape[:2])
    imagined_trajectories = torch.empty(
        cfg.algo.horizon + 1,
        new_batch_size,
        latent_states_size,
        device=device,
    )
    imagined_actions = torch.empty(
        cfg.algo.horizon + 1,
        new_batch_size,
        data["actions"].shape[-1],
        device=device,
    )
    imagined_stochastic_state = posterior_stochastic_state.reshape(1, -1, *posterior_stochastic_state.shape[2:])
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

    # Compute kv_cache_candidate from original sequence data
    # posterior_stochastic_state shape: (seq_len, batch_size, ...)
    # data["actions"] shape: (seq_len, batch_size, ...)
    # data["is_first"] shape: (seq_len, batch_size, 1)
    _, kv_cache_candidate = world_model._attn_output(posterior_stochastic_state, data["actions"], data["is_first"])
    # kv_cache_candidate: (k_cache, v_cache), each with shape (seq_len, batch_size, embed_dim)

    # new_batch_size = seq_len * batch_size
    k_cache_new = torch.zeros(sequence_length, new_batch_size, cfg.algo.world_model.transformer.embed_dim).to(device)
    v_cache_new = torch.zeros(sequence_length, new_batch_size, cfg.algo.world_model.transformer.embed_dim).to(device)
    new_is_first = torch.zeros(sequence_length, new_batch_size, 1, device=device)

    for i in range(sequence_length):
        start_batch = i * batch_size
        end_batch = (i + 1) * batch_size
        seq_len_needed = i + 1
        # Fill from the end
        k_cache_new[-seq_len_needed:, start_batch:end_batch] = kv_cache_candidate[0][0:seq_len_needed]
        v_cache_new[-seq_len_needed:, start_batch:end_batch] = kv_cache_candidate[1][0:seq_len_needed]
        new_is_first[-seq_len_needed:, start_batch:end_batch] = data["is_first"][0:seq_len_needed]
        if seq_len_needed < sequence_length - 1:
            # Mark the position where new query starts (the -seq_len_needed position) as is_first
            new_is_first[-seq_len_needed, start_batch:end_batch] = torch.ones_like(data["is_first"][-1])

    kv_cache = (k_cache_new[1:-1], v_cache_new[1:-1])
    new_is_first = new_is_first[1:]

    # Imagine trajectories in the latent space
    for i in range(1, cfg.algo.horizon + 1):
        imagined_logits, imagined_stochastic_state, imagined_attn_output, kv_cache = world_model.imagination(
            imagined_stochastic_state, actions, kv_cache, is_first=new_is_first)
        imagined_latent_state = world_model.latent(imagined_stochastic_state, imagined_attn_output)
        imagined_trajectories[i] = imagined_latent_state
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions
        kv_cache = (t[-sequence_length+2:] for t in kv_cache)
        new_is_first = torch.cat((new_is_first[-sequence_length+2:], torch.zeros_like(new_is_first[-1:])), dim=0)

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

    # Reset everything
    if actor_optimizer is not None:
        actor_optimizer.zero_grad(set_to_none=True)
    if critic_optimizer is not None:
        critic_optimizer.zero_grad(set_to_none=True)
    if ac_optimizer is not None:
        ac_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)


