import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.adam
import torch.optim.adam
import numpy as np
import math
from torch.nn import functional as F
from dataclasses import dataclass
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
class DecisionTransformer(nn.Module):
    """A transformer model.

    User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        r_dim: the dimension of the returns-to-go
        state_dim: the dimension of the states
        action_size: the dimension of the actions
        d_model: the dimension of the model
        T: the sequence length
    """
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        transformer_layers = [Block(config) for _ in range(config.n_layer)]
        self.transformer_layers = nn.Sequential(*transformer_layers)
        self.state_embedding = nn.Linear( config.state_size,config.n_embd)
        self.action_embedding = nn.Linear(config.action_size,config.n_embd )
        self.return_embedding = nn.Linear(config.command_size, config.n_embd)
        self.positional_embeddings = nn.Embedding(self.seq_len,config.n_embd )
        self.linear = nn.Linear(3*self.seq_len*config.n_embd,config.action_size)
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.r_size = config.command_size

 
    def forward(self,R,s,a,t):

        # R,s,a,t : returns-to-go, states, actions, timesteps
        # R : (batch_size, seq_len)
        # s : (batch_size, seq_len, state_size)
        # a : (batch_size, seq_len, action_size)
        # t : (batch_size, seq_len)
        
        # assert R.shape[0] == s.shape[0] == a.shape[0] 
        # assert R.shape[1] == s.shape[1] == a.shape[1] 
        assert R.shape[0] == s.shape[0] and s.shape[0]==a.shape[0], f"Batch shape is not consistent Batch R={R.shape} Batch s={s.shape} Batch a={a.shape}"
        B = R.shape[0] 
        pos = torch.arange(0,self.seq_len, dtype=torch.long, device=device) # shape (t)
        pos_embeddings = self.positional_embeddings(pos)
        

        # print("sel.state_embedding.device",self.state_embedding.get_device())
        assert not torch.isnan(s).any(), f's contains a nan {s}'
        assert not torch.isnan(pos_embeddings).any(), f'pos embedding contains a nan {pos_embeddings}'
        s_embedding  = self.state_embedding(s) + pos_embeddings  

        assert not torch.isnan(s_embedding).any(), f"WHAT THE FUCK s_embedding is NAN"

        
        a_embedding = self.action_embedding(a) + pos_embeddings
        assert  not torch.isnan(s_embedding).any(), f"WHAT THE FUCK a_embedding is NAN"

        R_embedding = self.return_embedding(R) + pos_embeddings
        assert not torch.isnan(R_embedding.isnan()).any(), f'WHAT the FUCB R-embedding is NAN'

        assert s_embedding.shape[1]== R_embedding.shape[1] # We have as many state than returns to go
        assert a_embedding.shape[1]== s_embedding.shape[1] # We have an action less than states and returns to go
        input_embeddings = torch.stack([ s_embedding, R_embedding, a_embedding], dim=1) # (B,3,T,d_model)
        input_embeddings = input_embeddings.permute(0,2,1,3)
        input_embeddings = input_embeddings.reshape(B, self.seq_len * 3, -1) 
        B = R.shape[0]
        x=self.transformer_layers(input_embeddings)
        assert not torch.isnan(x).any() ,f'wtf it is nan after ???'
        # flatten
        x= x.reshape([B,-1])
        x = self.linear(x)
        return x
def padding(state, seq_len, embedding_size, device="cpu"):
    """
    Get a state and add padding if it is not the correct size.
    """
    # Handle the case where the input state is empty
    if state.nelement() == 0:  # Check for empty tensor
        return torch.zeros(seq_len, embedding_size, device=device)
    
    # Check if the state is already the correct shape
    if state.shape == (seq_len, embedding_size):
        return state.to(device)
    
    # If the sequence length is less than the target, pad at the beginning
    if state.shape[0] < seq_len:
        padding_size = seq_len - state.shape[0]
        padding_tensor = torch.zeros(padding_size, embedding_size, device=device)
        state = torch.cat((padding_tensor, state.to(device)), dim=0)
        return state
    if state.shape[0] > seq_len:
        state = state[-seq_len:]

    # Ensure the dimensions are valid
    assert state.shape[0] == seq_len, f"Expected seq_len={seq_len}, got {state.shape[0]}"
    assert state.shape[1] == embedding_size, f"Expected embedding_size={embedding_size}, got {state.shape[1]}"
    return state.to(device)

@dataclass
class DecisionTransformerConfig:
    block_size: int = 1024
    state_size: int = 8
    command_size: int= 2
    action_size: int= 4 
    n_layer: int = 5
    seq_len: int = 10
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
if __name__ == "__main__":
    d_model = 512
    state_dim = 10
    action_dim = 4
    r_dim = 2 
    seq_len = 10
    num_layers=15
    B = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DecisionTransformer( r_dim,state_dim,action_dim,d_model,seq_len,num_layers).to(device)
    # for name,layer in model.named_modules():
        # print(f"Layer Name  {name}, Layer: {layer}")k
    R = torch.randn(B,seq_len,r_dim).to(device)
    s = torch.randn(B,seq_len,state_dim).to(device)
    a = torch.randn(B,seq_len,action_dim).to(device)
    t = torch.tensor([1]).to(device)
    y=model(R,s,a,t)
    import torchviz 
    # y= y.detach().cpu()
    # print(y)
    torchviz.make_dot(y, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    # model.forward(R,s,a,t)