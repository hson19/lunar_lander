import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.adam
import torch.optim.adam

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
    def __init__(self, r_size,state_size,action_size,d_model,seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.transformer_layers = nn.TransformerDecoderLayer(d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False, bias=False)
        self.transformer = nn.TransformerDecoder(self.transformer_layers, num_layers=6)
        self.output_layer = nn.Linear(d_model*seq_len*3, action_size)
        self.positional_embeddings = nn.Embedding(self.seq_len, d_model)
        self.state_embedding = nn.Linear( state_size,d_model,)
        self.action_embedding = nn.Linear(action_size, d_model)
        self.return_embedding = nn.Linear(r_size, d_model)
        self.state_size = state_size
        self.action_size = action_size
        self.r_size = r_size

    def forward(self,R,s,a,t):
        # R,s,a,t : returns-to-go, states, actions, timesteps
        # R : (batch_size, seq_len)
        # s : (batch_size, seq_len, state_size)
        # a : (batch_size, seq_len, action_size)
        # t : (batch_size, seq_len)
        
        assert R.shape[0] == s.shape[0] == a.shape[0] 
        assert R.shape[1] == s.shape[1] == a.shape[1] 
        B = R.size(0)
        
        all_positions = torch.arange(self.seq_len)
        pos_embeddings = self.positional_embeddings(all_positions)
        
        s_embedding  = self.state_embedding(s) + pos_embeddings  
               
        
        a_embedding = self.action_embedding(a) + pos_embeddings

        R_embedding = self.return_embedding(R) + pos_embeddings
        assert s_embedding.shape[1]== R_embedding.shape[1] # We have as many state than returns to go
        assert a_embedding.shape[1]== s_embedding.shape[1] # We have an action less than states and returns to go
        input_embeddings = torch.stack([ s_embedding, R_embedding, a_embedding], dim=1) # (B,3,T,d_model)
        input_embeddings = input_embeddings.permute(0,2,1,3) # (B,T,3,d_model)
        input_embeddings = input_embeddings.reshape(B, self.seq_len * 3, -1)  # [B, T*3, d_model]

        a_hidden = self.transformer.forward(input_embeddings, input_embeddings)
        a_hidden = a_hidden.view(B,-1)
        a_logits = self.output_layer(a_hidden)
        
        return a_logits

class BasicModel(nn.Module):

    def __init__(self,d_model=512,state_dim=10,action_dim=4,r_dim=10,seq_len=10,B=2):
        super().__init__()
        self.seq_len = seq_len
        self.model = DecisionTransformer(r_dim, state_dim, action_dim, d_model, self.seq_len)
        self.optim = None
        self.seq_len= seq_len

    def forward(self,R,s,a,t):
        return self.model(R,s,a,t)
    def init_optimizer(self,lr= torch.tensor(0.01)):
        self.optim = torch.optim.Adam([lr])

    
if __name__ == "__main__":
    d_model = 512
    state_dim = 10
    action_dim = 4
    r_dim = 1
    seq_len = 10
    B = 1
    model = BasicModel(d_model, state_dim, action_dim, r_dim, seq_len, B)
    R = torch.randn(B,seq_len,r_dim)
    s = torch.randn(B,seq_len,state_dim)
    a = torch.randn(B,seq_len,action_dim)
    t = torch.tensor([1])