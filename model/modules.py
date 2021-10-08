import math
import numpy as np
import torch
from torch import nn

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn import Linear, Sequential

class SelfAdditiveScorer(nn.Module):
    """
    Simple scorer of the form:
    v^T tanh(Wx + b) / sqrt(d)
    """

    def __init__(self, vector_size, attn_hidden_size):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(attn_hidden_size, vector_size))
        self.b = nn.Parameter(torch.Tensor(attn_hidden_size))
        self.v = nn.Parameter(torch.Tensor(1, attn_hidden_size))
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.W.shape[1])
        nn.init.uniform_(self.b, -bound, bound)
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

    def forward(self, query, keys):
        """Computes scores for each key of size n given the queries of size m.

        Args:
            query (torch.FloatTensor): query matrix (bs, ..., target_len, d_q)
            keys (torch.FloatTensor): keys matrix (bs, ..., source_len, d_k)

        Returns:
            torch.FloatTensor: scores between source and target words: (bs, ..., target_len, source_len)
        """
        # assume query = keys
        x = torch.matmul(query, self.W.t()) + self.b
        x = self.activation(x)
        score = torch.matmul(x, self.v.t()).squeeze(-1)
        return score / math.sqrt(keys.size(-1))

class RelaxedBernoulliGate(nn.Module):
    """
    Computes a Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, out_features=1):
        super(RelaxedBernoulliGate, self).__init__()

        self.layer = Sequential(Linear(in_features, out_features))

    def forward(self, x, mask):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """        
        logits = self.layer(x)  # [B, T, 1]
        logits = logits.squeeze(-1) * mask
        logits = logits.unsqueeze(-1)
        dist = RelaxedBernoulli(temperature=torch.tensor([0.1], device=logits.device), logits=logits)
        return dist