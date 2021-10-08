import math
from functools import partial

import torch
from entmax import entmax15, sparsemax, entmax_bisect
from transformers import XLMRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention, RobertaAttention


class SparseXLMRobertaModel(XLMRobertaModel):
    def __init__(self, config, add_pooling_layer=True, alpha=1.5, output_norm=False, norm_strategy='weighted_norm', effective=False):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention = SparseRobertaAttention(
                config, alpha=alpha, output_norm=output_norm, norm_strategy=norm_strategy, effective=effective
            )
        self.init_weights()  # reinit weights (important if we want to train from scratch)


class SparseRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config, alpha=1.5, output_norm=False, effective=False):
        super().__init__(config)
        if alpha == 1.0:
            self.transform_fn = torch.softmax
        elif alpha == 1.5:
            self.transform_fn = entmax15
        elif alpha == 2.0:
            self.transform_fn = sparsemax
        else:
            self.transform_fn = partial(entmax_bisect, alpha=alpha)
        self.output_norm = output_norm
        self.effective = effective

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.transform_fn(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # https://github.com/KaiserWhoLearns/Effective-Attention-Interpretability/
        ### Here, effective attention begins
        ## Calculate Project_(LN(T))(A)
        if self.effective:
            # Compute T
            T = value_layer
            # Make a SVD of T, U(dimension: [batch_size, hidden_size, hidden_size], [d_s, d_s]
            # S(dimension: square, and is of min(hidden_size, all_head_size), min(d_s, d))
            U, S, V = torch.Tensor.svd(T, some=False, compute_uv=True)
            # Find the bound of S, when S value less than bound, we treat it as a 0
            bound = torch.finfo(S.dtype).eps * max(U.shape[1], V.shape[1])
            # Find the basis of LN(T), null_space dimension: [batch_size, hidden_size, hidden_size - rank], [d_s, d_s-r]
            basis_start_index = torch.max(torch.sum(S > bound, dim=2).long())
            null_space = U[:, :, :, basis_start_index:]
            # Multiply attention with null_space, dimension: [batch_size, hidden_size, hidden_size - rank], [d_s, d_s-r]
            B = torch.matmul(attention_probs, null_space)
            # Transpose B [batch_size, hidden_size - rank, hidden_size]
            transpose_B = torch.transpose(B, -1, -2)
            # Multiply null_space and transposed B [batch_size, hidden_size, hiddensize]
            projection_attention = torch.matmul(null_space, transpose_B)
            # Then do tranpose for projection of LN(T)
            projection_attention = torch.transpose(projection_attention, -1, -2)
            # Compute the effective attention
            attention_probs = torch.sub(attention_probs, projection_attention)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.output_norm:
            outputs = (context_layer, attention_probs, value_layer)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# https://github.com/gorokoba560/norm-analysis-of-transformer/blob/master/transformers/src/transformers/modeling_bert.py
class AttentionNormOutput(torch.nn.Module):
    def __init__(self, config, norm_strategy='weighted_norm'):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.norm_strategy = norm_strategy

    def forward(self, hidden_states, attention_probs, value_layer, dense):
        # hidden_states: (batch, seq_length, all_head_size)
        # attention_probs: (batch, num_heads, seq_length, seq_length)
        # value_layer: (batch, num_heads, seq_length, head_size)
        # dense: nn.Linear(all_head_size, all_head_size)

        with torch.no_grad():
            # value_layer is converted to (batch, seq_length, num_heads, 1, head_size)
            value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
            value_shape = value_layer.size()
            value_layer = value_layer.view(value_shape[:-1] + (1, value_shape[-1],))

            # dense weight is converted to (num_heads, head_size, all_head_size)
            dense = dense.weight
            dense = dense.view(self.all_head_size, self.num_attention_heads, self.attention_head_size)
            dense = dense.permute(1, 2, 0).contiguous()

            # Make transformed vectors f(x) from Value vectors (value_layer) and weight matrix (dense).
            transformed_layer = value_layer.matmul(dense)
            transformed_shape = transformed_layer.size()  # (batch, seq_length, num_heads, 1, all_head_size)
            transformed_layer = transformed_layer.view(transformed_shape[:-2] + (transformed_shape[-1],))
            transformed_layer = transformed_layer.permute(0, 2, 1, 3).contiguous()
            # (batch, num_heads, seq_length, all_head_size)
            transformed_shape = transformed_layer.size()
            # (batch, num_heads, seq_length)
            transformed_norm = torch.norm(transformed_layer, dim=-1)
            # (batch, num_heads, seq_length, seq_length)
            weighted_norm = attention_probs * transformed_norm.unsqueeze(2)  # probas >= 0 always, no need of abs()

            # outputs:
            # transformed_norm: ||f(x)||
            # weighted_norm: ||αf(x)||
            # summed_weighted_norm: ||Σαf(x)||
            if self.norm_strategy == 'transformed_norm':
                outputs = transformed_norm

            elif self.norm_strategy == 'weighted_norm':
                outputs = weighted_norm

            elif self.norm_strategy == 'summed_weighted_norm':
                # Make weighted vectors αf(x) from transformed vectors (transformed_layer) and attention weights
                weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs, transformed_layer)
                # Sum each αf(x) over all heads: (batch, seq_length, seq_length, all_head_size)
                summed_weighted_layer = weighted_layer.sum(dim=1)
                # Calculate L2 norm of summed weighted vectors: (batch, seq_length, seq_length)
                summed_weighted_norm = torch.norm(summed_weighted_layer, dim=-1)
                outputs = summed_weighted_norm.unsqueeze(1)  # add head dimension
                del transformed_shape, transformed_layer, weighted_layer, summed_weighted_layer
                torch.cuda.empty_cache()

        return outputs


class SparseRobertaAttention(RobertaAttention):
    def __init__(self, config, alpha=1.5, output_norm=False, norm_strategy='weighted_norm', effective=False):
        super().__init__(config)
        self.self = SparseRobertaSelfAttention(config, alpha=alpha, output_norm=output_norm, effective=effective)
        self.norm = AttentionNormOutput(config, norm_strategy=norm_strategy)
        self.output_norm = output_norm

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            **kwargs
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states, **kwargs)

        if self.output_norm:
            norms_output = self.norm(hidden_states, self_outputs[1], self_outputs[2], self.output.dense)
            outputs = (attention_output, norms_output,) + self_outputs[2:]  # replace attention_probs by attn_norms
            return outputs

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
