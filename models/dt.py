"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill 
which is fixed in the following code
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.behavior import Behavior

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, config, max_timestep=4096):
        super().__init__()

        self.state_dim = config.state_dim
        self.act_dim = config.action_dim
        h_dim = config.n_embd
        self.h_dim = h_dim
        context_len = config.seq_len
        self.seq_len = config.seq_len
        n_heads = config.n_head
        n_blocks = config.n_layer
        drop_p = config.dropout
        self.max_timestep = max_timestep


        ### transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)


        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(self.act_dim, h_dim)
        use_action_tanh = False# True for continuous actions

        ### prediction heads
        self.predict_action = nn.Sequential(
            *([nn.Linear(3*self.seq_len*h_dim, self.act_dim)] + ([nn.Softmax()] if use_action_tanh else []))
        )


    def forward(self, returns_to_go, states, actions,verbose=False):
        if verbose: 
            print(f"returns_to_go: {returns_to_go.shape}")
            print(f"states: {states.shape}")
            print(f"actions: {actions.shape}")
            print(f"returns_to_go: {returns_to_go[0]}")
            print(f"states: {states[0]}")
            print(f"actions: {actions[0]}")
        B, T, _ = states.shape
        # time_embeddings = self.embed_timestep(timesteps)
        # assert not torch.isnan(time_embeddings).any(), f'time embedding contains a nan {time_embeddings}'
        
        timesteps = torch.arange(T, device=states.device).long()  # shape (T,)
        time_embeddings = self.embed_timestep(timesteps)          # (T, h_dim)
        time_embeddings = time_embeddings[None, :, :].expand(B, -1, -1) # (B, T, h_dim)
        if verbose:
            print(f"time embeddings: {time_embeddings.shape}")
            print(f"time embeddings: {time_embeddings[0]}")
        if verbose: print(f"states before embedding: {states.shape}")
        state_embeddings = self.embed_state(states) + time_embeddings
        if verbose: print(f"states after embedding: {state_embeddings.shape}")
        if verbose: print(f"actions before embedding: {actions.shape}")
        action_embeddings = self.embed_action(actions) + time_embeddings
        if verbose: print(f"actions after embedding: {action_embeddings.shape}")

        if verbose: print(f"returns_to_go before embedding: {returns_to_go.shape}")
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        if verbose: print(f"returns_to_go after embedding: {returns_embeddings.shape}")
        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)
        # transformer and prediction
        h = self.transformer(h)
        assert not torch.isnan(h).any() ,f'NAN after transformer'
        action_preds = self.predict_action(h.view(B,-1))  # predict action given r, s
        assert not torch.isnan(action_preds).any() ,f'NAN after predict action'

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        # state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        # action_preds = self.predict_action(h[:,1])  # predict action given r, s

        return action_preds

class BehaviorDT(DecisionTransformer,Behavior):
    def __init__(self, config):
        super().__init__(config)
        self.action_size = config.action_dim
        self.state_size = config.state_dim
        self.loss =  nn.CrossEntropyLoss()
        self.r_size = 1
        self.optim = None
        self.entropy_optim = None
        self.scheduler = None
    def init_optimizer(self,lr=0.3):
        self.optim = torch.optim.Adam(self.parameters(),lr=lr)
        self.entropy_optim = torch.optim.Adam(self.parameters(),lr,maximize=True)
        
        self.scheduler = ReduceLROnPlateau(self.optim, 'min', factor=0.5, patience=5, verbose=True)
