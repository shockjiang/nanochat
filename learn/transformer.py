from torch import nn
import torch

import math

class DotAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        """
        in:
            Q, K, V in or (B, N, head#, D)
            mask.shape=(B, N)
        returns:
            attention.shape=(B, N, head#, N)
            out.shape=(B, N, D)
        """
        score = torch.matmul(Q, K.transpose(-2, -1)) #(B, N, N)
        score = Q @ K.transpose(-2, -1) #(B, N, head#, D) @(B, N, D, head#) -> (B, N, head#, head#)
        dk = Q.shape[-1]
        score = score/math.sqrt(dk)
        if mask is not None:
            score = score.mask_fill(mask[..., None, None]==0, -torch.inf)
        attention = score.softmax(dim=-1) # (B, N, head#, head#)
        attention = self.dropout(attention)
        
        out = attention @ V # (B, N, head#, head#) @ (B, N, head# , D) -> (B, N, head#, D)
        return attention, out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, D, dropout):
        super().__init__()
        self.num_head = num_head
        self.w_q = nn.Linear(in_features=D, out_features=D, bias=False)
        self.w_k = nn.Linear(in_features=D, out_features=D, bias=False)
        self.w_v = nn.Linear(in_features=D, out_features=D, bias=False)
        self.w_o = nn.Linear(in_features=D, out_features=D, bias=False)
        self.att = DotAttention(dropout=dropout)  
    
    def forward(self, q, mask=None):
        """
        in: Q.shape=(B, N, D), mask.shape=(B, N)
        return:
            out.shape=(B, N, D)
        """
        B, N, D = q.shape
        assert D % self.num_head == 0, f'{D=} {self.num_head=}'
        D2 = D // self.num_head
        if mask:
            mask = None
        Q = self.w_q(q).view(B, N, -1, D2) # (B, N, head#, D2)
        K = self.w_k(q).view(B, N, -1, D2)
        V = self.w_v(q).view(B, N, -1, D2)
        attention, out = self.att(Q, K, V)
        out = out.view(B, N, -1) #(B, N, head#, D2) -> (B, N, D)
        out = self.w_o(out)
        return out

class FFN(nn.Module):
    def __init__(self, D, dropout):
        super().__init__()
        self.ln1 = nn.Linear(in_features=D, out_features=D, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.ln2 = nn.Linear(in_features=D, out_features=D, bias=False)
    
    def forward(self, q):
        q = self.ln1(q)
        q = self.relu(q)
        q = self.dropout(q)
        q = self.ln2(q)
        q = self.dropout(q)
        return q
    
class TransformerBlock(nn.Module):
    def __init__(self, num_head, dropout, D):
        super().__init__()
        self.norm0 = nn.LayerNorm(D)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.att = MultiHeadAttention(num_head=num_head, D=D, dropout=dropout)
        self.ffn = FFN(D=D, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, mask=None):
        """
        in: q.shape=(B, N, D), mask=(B, N)
        return out.shape=(B, N, D)
        """
        q0 = q
        q = self.norm0(q)
        out = self.dropout(self.att(q, mask)) + q
        

        out0 = out
        out = self.norm1(out)
        out = self.ffn(out)
        out = self.dropout(out) + out0
        return out

class Decoder(nn.Module):
    def __init__(self, num_head, depth, D, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(num_head=num_head, dropout=dropout, D=D)] * depth)
        self.fc = nn.Linear(in_features=D, out_features=D)

    def forward(self, q):
        """
        in: q.shape=(B, N, D)
        """
        out = q
        for layer in self.layers:
            # breakpoint()
            out = layer(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    torch.manual_seed(0)
    model = Decoder(num_head=8, depth=2, D=128, dropout=0.1)
    x = torch.randn(4, 16, 128)  # batch=4, seq_len=16, embed_dim=128
    out = model(x)
    print(out.shape)  # (4, 16, 128)