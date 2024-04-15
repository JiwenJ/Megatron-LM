# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import math

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from megatron.metee_plugin.long_context import retrieve_database
from megatron.training import get_args


class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

        self.is_memorizing_layer = False
        if get_args().retrieve_transformer and layer_number == self.config.memorizing_layer:
                self.is_memorizing_layer = True
                self.retrieve_database = retrieve_database()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs
    ):
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # retrieve history key-value pairs
        if self.is_memorizing_layer and 'is_last_chunk' in kwargs.keys():

            # use forward_only as an identifier of evaluation
            if 'forward_only' in kwargs.keys() and kwargs['forward_only']==True:
                self.retrieve_database.set_eval_mode()
            else:
                self.retrieve_database.set_train_mode()

            if len(self.retrieve_database): # is first chunk or not
                key_ret, val_ret = self.retrieve_database.get(query, kwargs['top_k'])
                key_ret = torch.from_numpy(key_ret).to(device=key.device, dtype=key.dtype)
                val_ret = torch.from_numpy(val_ret).to(device=value.device, dtype=value.dtype)
                first_chunk = False
            else:
                first_chunk = True

            if kwargs['is_last_chunk']:
                self.retrieve_database.clear()
            else:
                self.retrieve_database.add(key, value)

        if self.is_memorizing_layer and 'is_last_chunk' in kwargs.keys():
            if first_chunk:
                # [b, np, sq, sk]
                output_size = (
                    query.size(1),
                    query.size(2),
                    query.size(0),
                    key.size(0),
                )

                # [sq, b, np, hn] -> [sq, b * np, hn]
                # This will be a simple view when doing normal attention, but in group query attention
                # the key and value tensors are repeated to match the queries so you can't use simple strides
                # to extract the queries.
                query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
                # [sk, b, np, hn] -> [sk, b * np, hn]
                key = key.view(output_size[3], output_size[0] * output_size[1], -1)

                # preallocting input tensor: [b * np, sq, sk]
                matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
                    (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
                )

                # Raw attention scores. [b * np, sq, sk]
                matmul_result = torch.baddbmm(
                    matmul_input_buffer,
                    query.transpose(0, 1),  # [b * np, sq, hn]
                    key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                    beta=0.0,
                    alpha=(1.0 / self.norm_factor),
                )

                # change view to [b, np, sq, sk]
                attention_scores = matmul_result.view(*output_size)

                # ===========================
                # Attention probs and dropout
                # ===========================

                # attention scores and attention mask [b, np, sq, (top_k+1)*sk]
                # attention_mask = torch.cat()
                attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

                # This is actually dropping out entire tokens to attend to, which might
                # seem a bit unusual, but is taken from the original Transformer paper.

                if not self.config.sequence_parallel:
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        attention_probs = self.attention_dropout(attention_probs)
                else:
                    attention_probs = self.attention_dropout(attention_probs)

                # =========================
                # Context layer. [sq, b, hp]
                # =========================

                # value -> context layer.
                # [sk, b, np, hn] --> [b, np, sq, hn]

                # context layer shape: [b, np, sq, hn]
                output_size = (
                    value.size(1),
                    value.size(2),
                    query.size(0),
                    value.size(3),
                )

                # change view [sk, b * np, hn]
                value = value.view(value.size(0), output_size[0] * output_size[1], -1)

                # change view [b * np, sq, (top_k+1)sk]
                attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

                # matmul: [b * np, sq, hn]
                context = torch.bmm(attention_probs, value.transpose(0, 1))

                # change view [b, np, sq, hn]
                context = context.view(*output_size)

                # [b, np, sq, hn] --> [sq, b, np, hn]
                context = context.permute(2, 0, 1, 3).contiguous()

                # [sq, b, np, hn] --> [sq, b, hp]
                new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
                context = context.view(*new_context_shape)

                return context

            else:
	        # for attention_retrieve, launch twice kernel
                output_size = (
                    query.size(1),
                    query.size(2),
                    query.size(0),
                    key.size(0),
                )
                query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
                key = key.view(output_size[3], output_size[0] * output_size[1], -1)
                matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
                    (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
                )
                matmul_result = torch.baddbmm(
                    matmul_input_buffer,
                    query.transpose(0, 1),
                    key.transpose(0, 1).transpose(1, 2),
                    beta=0.0,
                    alpha=(1.0 / self.norm_factor),
                )
                attention_scores = matmul_result.view(*output_size)

                output_size = output_size[:3]+(key_ret.size(0),)
                key_ret = key_ret.view(output_size[3], output_size[0] * output_size[1], -1)
                matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
                    (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
                )
                matmul_result = torch.baddbmm(
                    matmul_input_buffer,
                    query.transpose(0, 1),
                    key_ret.transpose(0, 1).transpose(1, 2),
                    beta=0.0,
                    alpha=(1.0 / self.norm_factor),
                )
                del key_ret
                attention_scores_ret = matmul_result.view(*output_size)
                #attention_scores = torch.cat((attention_scores, attention_scores_ret), dim=-1)
                #del attention_scores_ret

                attention_mask_ret = torch.ones((attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2], kwargs['top_k']*attention_mask.shape[3])) < 0.5
                attention_mask_ret = attention_mask_ret.to(device=attention_mask.device)
                ## attention scores and attention mask [b, np, sq, (1+top_k)*sk]
                #attention_mask = torch.cat((attention_mask, attention_mask_ret), dim=3)
                #del attention_mask_ret
                attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)
                attention_probs_ret: Tensor = self.scale_mask_softmax(attention_scores_ret, attention_mask_ret)

                # This is actually dropping out entire tokens to attend to, which might
                # seem a bit unusual, but is taken from the original Transformer paper.

                if not self.config.sequence_parallel:
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        attention_probs = self.attention_dropout(attention_probs)
                        attention_probs_ret = self.attention_dropout(attention_probs_ret)
                else:
                    attention_probs = self.attention_dropout(attention_probs)
                    attention_probs_ret = self.attention_dropout(attention_probs_ret)

                # =========================
                # Context layer. [sq, b, hp]
                # =========================

                # value -> context layer.
                # [sk, b, np, hn] --> [b, np, sq, hn]

                # context layer shape: [b, np, sq, hn]
                output_size = (
                    value.size(1),
                    value.size(2),
                    query.size(0),
                    value.size(3),
                )

                # change view [sk, b * np, hn]
                value = value.view(value.size(0), output_size[0] * output_size[1], -1)
                # change view [top_k*sk, b * np, hn]
                val_ret = val_ret.view(val_ret.size(0), output_size[0] * output_size[1], -1)
                #value = torch.cat((value, val_ret), dim=0)
                #del val_ret

                # change view [b * np, sq, (top_k+1)sk]
                attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
                attention_probs_ret = attention_probs_ret.view(output_size[0] * output_size[1], output_size[2], -1)

                # matmul: [b * np, sq, hn]
                context = torch.bmm(attention_probs, value.transpose(0, 1))
                context_ret = torch.bmm(attention_probs_ret, val_ret.transpose(0, 1))
                del val_ret

                # change view [b, np, sq, hn]
                context = context.view(*output_size)
                context_ret = context_ret.view(*output_size)

                # [b, np, sq, hn] --> [sq, b, np, hn]
                context = context.permute(2, 0, 1, 3).contiguous()
                context_ret = context_ret.permute(2, 0, 1, 3).contiguous()

                belta = torch.rand(context.shape[2], dtype=context.dtype, device=context.device, requires_grad=True)
                belta = belta.view(1, 1, context.shape[2], 1)
                context = torch.sigmoid(belta)*context_ret + (1-torch.sigmoid(belta))*context
                del context_ret

                # [sq, b, np, hn] --> [sq, b, hp]
                new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
                context = context.view(*new_context_shape)

                return context
        else:
            # [b, np, sq, sk]
            output_size = (
                query.size(1),
                query.size(2),
                query.size(0),
                key.size(0),
            )

            # [sq, b, np, hn] -> [sq, b * np, hn]
            # This will be a simple view when doing normal attention, but in group query attention
            # the key and value tensors are repeated to match the queries so you can't use simple strides
            # to extract the queries.
            query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key = key.view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
                (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query.transpose(0, 1),  # [b * np, sq, hn]
                key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.

            if not self.config.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    attention_probs = self.attention_dropout(attention_probs)
            else:
                attention_probs = self.attention_dropout(attention_probs)

            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (
                value.size(1),
                value.size(2),
                query.size(0),
                value.size(3),
            )

            # change view [sk, b * np, hn]
            value = value.view(value.size(0), output_size[0] * output_size[1], -1)

            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

            # matmul: [b * np, sq, hn]
            context = torch.bmm(attention_probs, value.transpose(0, 1))

            # change view [b, np, sq, hn]
            context = context.view(*output_size)

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context = context.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
            context = context.view(*new_context_shape)

            return context
