from functools import partial

import torch
from entmax import entmax15, sparsemax, entmax_bisect
from model.utils import masked_average


class BottleneckSummary(torch.nn.Module):

    def __init__(
        self,
        hidden_size,
        aggregation='none',
        kv_rep='embeddings',
        alpha=1.0,
        classwise=False,
        alpha_merge=0.5,
        squared_attn=False
    ):
        super().__init__()
        self.classwise = classwise
        self.hidden_size = hidden_size + int(self.classwise)
        self.aggregation = aggregation
        self.kv_rep = kv_rep
        self.alpha = alpha
        self.alpha_merge = alpha_merge
        self.squared_attn = squared_attn
        if alpha < 1.0:
            self.transform_fn = torch.sigmoid
        elif alpha == 1.0:
            self.transform_fn = partial(torch.softmax, dim=-1)
        elif alpha == 1.5:
            self.transform_fn = partial(entmax15, dim=-1)
        elif alpha == 2.0:
            self.transform_fn = partial(sparsemax, dim=-1)
        else:
            self.transform_fn = partial(entmax_bisect, alpha=alpha, dim=-1)
        self.q_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.k_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.v_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self._init_weights()

    def _init_weights(self):
        pass
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)

    def forward(self,
                hidden_states: torch.Tensor,
                embeddings: torch.Tensor,
                attention_mask: torch.Tensor,
                first_sentence_mask: torch.Tensor,
                first_piece_mask: torch.Tensor,
                separate_mt_and_src=True
                ):
        """
        Produce an estimation by adding a bottleneck computation on top of hidden states from selected layers.

        Args:
            hidden_states: selected hidden states with shape of (bs, ts, hdim)
            embeddings: output of embeddings layer of BERT. Shape of (bs, ts, hdim)
            attention_mask: binary mask, 1 indicates the positions of valid (non pad) inputs. Shape of (bs, ts)
            first_sentence_mask: binary mask, 1 indicates the positions of the first sentence. Shape of (bs, ts)
            first_piece_mask:  binary mask, 1 indicates the positions of the first word piece. Shape of (bs, ts)

        Returns:
            torch.Tensor with a shape of (bs, hdim)
        """
        # optionally, aggregate word pieces of k and v
        if self.aggregation != 'none':
            bounds, bounds_mask = self.get_bounds_from_first_piece_mask(first_piece_mask)
            embeddings = self.aggregate_word_pieces(embeddings, bounds, method=self.aggregation)
            hidden_states = self.aggregate_word_pieces(hidden_states, bounds, method=self.aggregation)
            r = torch.arange(bounds.shape[0], device=bounds.device).unsqueeze(-1)
            first_sentence_mask = first_sentence_mask[r, bounds]
            first_piece_mask = first_piece_mask[r, bounds]
            attention_mask = bounds_mask

        # select which vectors to use to represent keys and values
        kv_rep = embeddings if self.kv_rep == 'embeddings' else hidden_states

        if self.alpha_merge < 0:  # do attention to mt and src jointly
            if not self.squared_attn:
                # masked average over word pieces to get a single query representation
                hidden_states_avg = masked_average(hidden_states, attention_mask)
                q = self.q_layer(hidden_states_avg).unsqueeze(1)
            else:
                q = self.q_layer(hidden_states)

            # linear map for k and v
            k = self.k_layer(kv_rep)
            v = self.v_layer(kv_rep)

            # attention
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / k.shape[-1] ** 0.5
            attn_scores = attn_scores.masked_fill(~attention_mask.unsqueeze(1).bool(), -10000.0)
            attn_probas = self.transform_fn(attn_scores)
            if not self.squared_attn:
                attn_alphas = attn_probas if self.alpha >= 1.0 else attn_probas.clone() / attn_probas.sum(-1).unsqueeze(-1)
                combined_summary = torch.matmul(attn_alphas, v).squeeze(1)
                # we select scores for alpha=-1 because we will use it with BCELoss and fp16
                attn_probas = attn_probas.squeeze(1) if self.alpha >= 1.0 else attn_scores.squeeze(1)
            else:
                attn_alphas = attn_probas if self.alpha >= 1.0 else attn_probas.clone() / attn_probas.sum(-1).unsqueeze(-1)
                combined_summary = masked_average(torch.matmul(attn_alphas, v), attention_mask)
                # we select scores for alpha=-1 because we will use it with BCELoss and fp16
                attn_probas = attn_probas if self.alpha >= 1.0 else attn_scores
                attn_probas = masked_average(attn_probas, attention_mask)

            if separate_mt_and_src:
                # break attn probas into mt and src
                mt_probas, src_probas, mt_mask, src_mask = self.separate_mt_and_src(attn_probas, first_sentence_mask,
                                                                                    attention_mask)

                # set pad probas to zero (we might have created new pad positions when splitting <mt> from <src>)
                mt_probas = mt_probas * mt_mask.float()
                src_probas = src_probas * src_mask.float()

                # renormalize
                # mt_probas = mt_probas / mt_probas.sum(-1).unsqueeze(-1)
                # src_probas = src_probas / src_probas.sum(-1).unsqueeze(-1)

                attentions = (mt_probas, src_probas)

            else:
                attentions = attn_probas

        else:  # do attention to mt and src separately
            hidden_states_mt, hidden_states_src, _, _ = self.separate_mt_and_src(hidden_states, first_sentence_mask,
                                                                                 attention_mask)
            kv_rep_mt, kv_rep_src, mt_mask, src_mask = self.separate_mt_and_src(kv_rep, first_sentence_mask,
                                                                                attention_mask)

            # avg hidden states
            hidden_states_avg_mt = masked_average(hidden_states_mt, mt_mask)
            hidden_states_avg_src = masked_average(hidden_states_src, src_mask)

            # do attn of src over mt
            if not self.squared_attn:
                q = self.q_layer(hidden_states_avg_src)
            else:
                q = self.q_layer(hidden_states_src)
            k = self.k_layer(hidden_states_mt)
            v = self.v_layer(hidden_states_mt)
            s = torch.matmul(q, k.transpose(-1, -2)) / k.shape[-1] ** 0.5
            s = s.masked_fill(~mt_mask.unsqueeze(1).bool(), -10000.0)
            mt_probas = self.transform_fn(s)
            if not self.squared_attn:
                mt_summary = torch.matmul(mt_probas, v).squeeze(1)
                mt_probas = mt_probas.squeeze(1).clone() if self.alpha >= 1.0 else s.squeeze(1).clone()
            else:
                mt_summary = torch.matmul(mt_probas, v)
                mt_summary = masked_average(mt_summary, src_mask)
                mt_probas = mt_probas.clone() if self.alpha >= 1.0 else s.clone()
                mt_probas = masked_average(mt_probas, src_mask)

            # do attn of mt over src
            if not self.squared_attn:
                q = self.q_layer(hidden_states_avg_mt)
            else:
                q = self.q_layer(hidden_states_mt)
            k = self.k_layer(hidden_states_src)
            v = self.v_layer(hidden_states_src)
            s = torch.matmul(q, k.transpose(-1, -2)) / k.shape[-1] ** 0.5
            s = s.masked_fill(~src_mask.unsqueeze(1).bool(), -10000.0)
            src_probas = self.transform_fn(s)
            if not self.squared_attn:
                src_summary = torch.matmul(src_probas, v).squeeze(1)
                src_probas = src_probas.squeeze(1).clone() if self.alpha >= 1.0 else s.squeeze(1).clone()
            else:
                src_summary = torch.matmul(src_probas, v)
                src_summary = masked_average(src_summary, mt_mask)
                src_probas = src_probas if self.alpha >= 1.0 else s
                src_probas = masked_average(src_probas, mt_mask)

            combined_summary = self.alpha_merge * mt_summary + (1 - self.alpha_merge) * src_summary
            attentions = (mt_probas, src_probas)

        return combined_summary, attentions

    @staticmethod
    def separate_mt_and_src(tensor, first_sentence_mask, attention_mask):
        """
        Split a tensor into two according to the bool first_sentence_mask tensor.
        It will use attention_mask to get rid of pad positions.

        It assumes a concatenated input tensor: <mt> <src>.

        Args:
            tensor (torch.Tensor): shape of (bs, ts, hdim)
            first_sentence_mask (torch.LongTensor): boolean tensor, with 1s indicating the positions of <mt>
                and 0s of <src>. Shape of (bs, ts)
            attention_mask (torch.LongTensor): mask of pad positions, 1s indicate valid and 0s indicatep pad positions.
                Shape of (bs, ts)

        Returns:
            <mt> torch.Tensor (bs, mt_len, hdim)
            <src> torch.Tensor (bs, src_len, hdim)
            mt_mask torch.BoolTensor (bs, mt_len) with 1s indicating valid positions, and 0s pad positions
            src_mask torch.BoolTensor (bs, src_len) with 1s indicating valid positions, and 0s pad positions
        """
        first_sentence_mask = first_sentence_mask.long()
        attention_mask = attention_mask.long()
        # recover mt tensor
        mt_len = first_sentence_mask.sum(-1).max().item()
        tensor_mt = tensor[:, :mt_len].clone()
        mt_mask = attention_mask[:, :mt_len] & first_sentence_mask[:, :mt_len]
        # recover src tensor + rest of padding (which will be dealt later in the loss fn)
        src_first = first_sentence_mask.sum(-1).min().item()
        tensor_src = tensor[:, src_first:].clone()
        src_mask = attention_mask[:, src_first:] & (1 - first_sentence_mask)[:, src_first:]

        return tensor_mt, tensor_src, mt_mask, src_mask

    @staticmethod
    def get_bounds_from_first_piece_mask(first_piece_mask):
        """
        Transforms a binary mask of first word piece positions to 0-indexed bounds tensor.
        E.g.
        [[1, 0, 0, 1, 0, 1, 0, 1],
         [1, 1, 0, 0, 0, 1, 0, 0]]
        will be transformed to
        [[0, 3, 5, 6],
         [0, 1, 5, -1]]
        where -1 indicates pad positions.
        """
        bs, seq_len = first_piece_mask.shape
        device = first_piece_mask.device
        bounds_highest_index = 999999
        bounds = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
        bounds = bounds.masked_fill(~first_piece_mask.bool(), bounds_highest_index)
        bounds = bounds.sort(dim=-1)[0]
        bounds_mask = (bounds != bounds_highest_index).bool()
        max_num_bounds = bounds_mask.long().sum(-1).max().item()
        bounds = bounds[:, :max_num_bounds]
        bounds[bounds == bounds_highest_index] = -1
        bounds_mask = (bounds != -1).bool()
        return bounds, bounds_mask

    @staticmethod
    def aggregate_word_pieces(hidden_states, bounds, method='first'):
        """
        Aggregate hidden states according to word piece tokenization.

        Args:
            hidden_states (torch.Tensor): output of BERT. Shape of (bs, ts, hdim)
            bounds (torch.LongTensor): the indexes where the word pieces start.
                Shape of (bs, ts)
                e.g. Welcome to the jungle -> _Wel come _to _the _jungle
                     bounds[0] = [0, 2, 3, 4]
                indexes for padding positions are expected to be equal to -1
            method (str): the strategy used to get a representation of a word
                based on its word pices. Possible choices are:
                'first' = take the vector of the first word piece
                'sum' = take the sum of word pieces vectors
                'mean' = take the average of word pieces vectors
                'max' = take the max of word pieces vectors

        Returns:
            torch.Tensor (bs, original_sequence_length, hdim)
        """
        bs, ts, hidden_dim = hidden_states.size()
        r = torch.arange(bs, device=hidden_states.device).unsqueeze(1)

        if method == 'first':
            return hidden_states[r, bounds]

        elif method == 'sum' or method == 'mean':
            neg_one_indexes = bounds.new_zeros(bs, 1) - 1
            extended_bounds = torch.cat((bounds[:, 1:], neg_one_indexes), dim=1)
            last_idx = (extended_bounds != -1).sum(dim=1).unsqueeze(-1) - 1
            extended_bounds[r, last_idx + 1] = extended_bounds[r, last_idx] + 1
            shifted_bounds = extended_bounds - 1
            cumsum = hidden_states.cumsum(dim=1)
            cumsum = cumsum[r, shifted_bounds]
            zero_values = cumsum.new_zeros(bs, 1, hidden_dim)
            shifted_cumsum = torch.cat((zero_values, cumsum[:, :-1]), dim=1)
            selected_pieces = cumsum - shifted_cumsum
            if method == 'mean':
                lens = shifted_bounds + 1 - bounds
                lens[lens == 0] = 1  # we should not have a case where lens_ij=0
                selected_pieces = selected_pieces / lens.unsqueeze(-1).float()
            return selected_pieces

        elif method == 'max':
            max_bounds_size = (bounds != -1).sum(1).max().item()
            max_wordpieces = torch.zeros(bs, max_bounds_size, hidden_dim, device=bounds.device)
            for i in range(bs):
                bounds_len = (bounds[i] != -1).sum().item()
                valid_bounds = bounds[i, :bounds_len].tolist()
                valid_bounds.append(valid_bounds[-1] + 1)
                slices = zip(valid_bounds[:-1], valid_bounds[1:])
                for j, (k1, k2) in enumerate(slices):
                    x, _ = torch.max(hidden_states[i, k1:k2], dim=0)
                    max_wordpieces[i, j] = x
            return max_wordpieces

        else:
            raise Exception('Method {} is not implemented'.format(method))
