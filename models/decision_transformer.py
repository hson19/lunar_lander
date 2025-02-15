import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.adam
import torch.optim.adam
import numpy as np
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    def __init__(self, r_size,state_size,action_size,d_model,seq_len,num_layers):
        super().__init__()
        self.seq_len = seq_len
        transformer_layers = [nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False, bias=False) for _ in range(num_layers)]
        self.transformer_layers = nn.Sequential(*transformer_layers)
        self.state_embedding = nn.Linear( state_size,d_model)
        self.action_embedding = nn.Linear(action_size, d_model)
        self.return_embedding = nn.Linear(r_size, d_model)
        self.positional_embeddings = nn.Embedding(self.seq_len, d_model)
        self.linear = nn.Linear(3*seq_len*d_model,action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.r_size = r_size

    # def to_input_embeddings(self,R,s,a,t):
    #     # input  
    #     # R (B,T,Command)
    #     # s (B,T,State_size)
    #     # a (B,T,action_size)
    #     assert R.shape[0] == s.shape[0] and s.shape[0]==a.shape[0], f"Batch shape is not consistent Batch R={R.shape} Batch s={s.shape} Batch a={a.shape}"
    #     B = R.shape[0] 
    #     all_positions = torch.arange(self.seq_len).to(device)
    #     pos_embeddings = self.positional_embeddings(all_positions)
        

        
    #     s_embedding  = self.state_embedding(s) + pos_embeddings  
               
        
    #     a_embedding = self.action_embedding(a) + pos_embeddings

    #     R_embedding = self.return_embedding(R) + pos_embeddings
    #     assert s_embedding.shape[1]== R_embedding.shape[1] # We have as many state than returns to go
    #     assert a_embedding.shape[1]== s_embedding.shape[1] # We have an action less than states and returns to go
    #     input_embeddings = torch.stack([ s_embedding, R_embedding, a_embedding], dim=1) # (B,3,T,d_model)
    #     input_embeddings = input_embeddings.permute(0,2,1,3)
    #     input_embeddings = input_embeddings.reshape(B, self.seq_len * 3, -1) 
    #     return input_embeddings
 
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