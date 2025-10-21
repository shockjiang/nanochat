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
            Q, K, V in (B, h, N, D)
            mask.shape=(B, N)
        returns:
            attention.shape=(B, h, N, N)
            out.shape=(B, h, N, D)
        """
        B, h, N, d = Q.shape
        score = Q @ K.transpose(-2, -1)  # (B, h, N, D) @(B, h, D, N) -> (B, h, N, N), Note: score is calculated on N dimension instead of h dimension
        score = score / math.sqrt(d)  # (B, h, N, N)
        if mask is not None:
            score = score.masked_fill(mask[:, None, None, :] == 0, float('-inf'))  # mask:(B, 1, 1, N), score.shape=(B, h, N, N)
        attention = score.softmax(dim=-1)  # (B, h, N, N)
        attention = self.dropout(attention)  # (B, h, N, N)

        out = attention @ V  # (B, h, N, N) @ (B, h, N, D) -> (B, h, N, D)
        return attention, out


class MultiHeadAttention(nn.Module):
    def __init__(self, h, D, dropout):
        super().__init__()
        self.num_head = h
        self.w_q = nn.Linear(in_features=D, out_features=D, bias=False)
        self.w_k = nn.Linear(in_features=D, out_features=D, bias=False)
        self.w_v = nn.Linear(in_features=D, out_features=D, bias=False)
        self.w_o = nn.Linear(in_features=D, out_features=D, bias=False)
        self.att = DotAttention(dropout=dropout)

    def forward(self, q, mask=None, k=None, v=None,):
        """
        in: Q.shape=(B, N, D), mask.shape=(B, N)
        return:
            out.shape=(B, N, D)
        """
        B, N, D = q.shape
        assert D % self.num_head == 0, f'{D=} {self.num_head=}'
        d2 = D // self.num_head
        if k is None and v is None:
            v = k = q
        elif k is None and v is not None:
            k = v
        elif k is not None and v is not None:
            pass
        else:
            raise NotImplementedError(f'{k=} {v=}')
        Q = self.w_q(q).view(B, N, -1, d2).transpose(-3, -2)  # (B, N, D) -> (B, N, h, d) -> (B, h, N, d)
        K = self.w_k(k).view(B, N, -1, d2).transpose(-3, -2)
        V = self.w_v(v).view(B, N, -1, d2).transpose(-3, -2)
        attention, out = self.att(Q, K, V, mask)  # attention.shape=(B, h, N, N), out.shape=(B, h, N, d)
        out = out.transpose(-3, -2).contiguous().view(B, N, -1)  # (B, h, N, d)-> (B, N, h, d)->(B, N, D)
        out = self.w_o(out)  # (B, N, D)
        return out


class FFN(nn.Module):
    def __init__(self, D, dropout, inner_dim=None):
        super().__init__()
        if inner_dim is None:
            inner_dim = 4 * D
        self.fc1 = nn.Linear(in_features=D, out_features=inner_dim, bias=True)
        self.GELU = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(in_features=inner_dim, out_features=D, bias=False)

    def forward(self, q):
        q = self.fc1(q)
        q = self.GELU(q)
        q = self.dropout(q)
        q = self.fc2(q)
        q = self.dropout(q)
        return q


class PositionalEncoding(nn.Module):
    def __init__(self, D, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, D, dtype=float)  # (L, D)
        pos = torch.arange(0, max_len, dtype=float)  # (L)
        # div = torch.exp(-torch.arange(0, D, 2, dtype=float) * 2 * math.log(10000) / D) #(D/2)
        div = torch.pow(10000, torch.arange(0, D, 2, dtype=float) * 2 / D)  # (D/2)
        tmp = pos[:, None] / div[None, :]  # (L, D/2)
        pe[:, 0::2] = torch.sin(tmp)  # (L, D/2)
        pe[:, 1::2] = torch.cos(tmp)  # (L, D/2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        in: x.shape=(B, N, D)
        return: (B,N,D)
        """
        B, N, D = x.shape
        tmp = self.pe[:N]  # (N, D)
        return x + tmp[None, ...]  # (B, N, D) + (1, N, D) -> (B, N, D)


class RotaryPositionEncoding(nn.Module):
    def __init__(self, D, max_len=10000, base=10000):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=float)  # (L)
        div = torch.pow(base, torch.arange(0, D, 2, dtype=float) * 2 / D)  # (D/2)
        tmp = pos[:, None] / div[None, :]  # (L, D/2)
        sin = torch.sin(tmp)  # (L, D/2)
        cos = torch.cos(tmp)  # (L, D/2)
        cos = cos.repeat_interleave(2, dim=0)  # (cos0, cos0, cos1, cos1, ...) #(L, D)
        sin = sin.repeat_interleave(2, dim=0)  # (sin0, sin0, sin1, sin1, ...) #(L, D)
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)

    def apply(self, cos, sin, qk):
        tmp = qk.clone()
        tmp[..., 0::2] = -qk[:, 1::2]
        tmp[..., 1::2] = qk[:, 0::2]
        return cos * qk + sin * tmp  # (N,D) * (B,N,D)

    def forward(self, q, k):
        """
        in: x.shape=(B, N, D)
        """
        assert q.shape == k.shape, f'{q.shape=} {k.shape=}'
        B, N, D = q.shape
        cos = self.cos[:N]  # (N, D)
        sin = self.sin[:N]  # (N, D)

        q = self.apply(cos, sin, q)
        k = self.apply(cos, sin, k)
        return q, k


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_head, dropout, D):
        super().__init__()
        self.norm0 = nn.LayerNorm(D)
        self.norm1 = nn.LayerNorm(D)
        # self.norm2 = nn.LayerNorm(D)
        self.att = MultiHeadAttention(h=num_head, D=D, dropout=dropout)
        self.ffn = FFN(D=D, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, mask=None):
        """
        in: q.shape=(B, N, D), mask=(B, N)
        return out.shape=(B, N, D)
        """
        q0 = q
        q = self.norm0(q)
        out = self.dropout(self.att(q, mask)) + q0

        out0 = out
        out = self.norm1(out)
        out = self.ffn(out)
        out = self.dropout(out) + out0
        return out


class Encoder(nn.Module):
    def __init__(self, num_head, depth, D, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(num_head=num_head, dropout=dropout, D=D)
                                    for _ in range(depth)])  # do not use list multiple problem, which is shared reference
        self.fc = nn.Linear(in_features=D, out_features=D)
        self.pe = PositionalEncoding(D=D)

    def forward(self, q, mask=None):
        """
        in: q.shape=(B, N, D)
        """
        out = q
        out = self.pe.forward(out)
        for layer in self.layers:
            # breakpoint()
            out = layer(out, mask=mask)
        out = self.fc(out)
        return out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_head, dropout, D):
        super().__init__()
        self.norm0 = nn.LayerNorm(D)
        self.self_attn = MultiHeadAttention(h=num_head, D=D, dropout=dropout)
        self.cross_attn = MultiHeadAttention(h=num_head, D=D, dropout=dropout)
        self.ffn = FFN(D=D, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, q, self_mask, k, v, cross_mask):
        """
        in: q.shape=(B, N, D)
        """
        q0 = q
        q = self.norm0(q)
        q = self.self_attn(q, self_mask)
        q = self.dropout(q) + q0

        # q = self.dropout(self.self_attn(self.norm0(q), self_mask)) + q

        q0 = q
        q = self.norm1(q)
        q = self.cross_attn(q, cross_mask, k, v)
        q = self.dropout(q) + q0

        q0 = q
        q = self.norm2(q)
        q = self.ffn(q)
        q = self.dropout(q) + q0
        return q


class Decoder(nn.Module):
    def __init__(self, num_head, depth, D, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderBlock(num_head=num_head, dropout=dropout, D=D)
                                    for _ in range(depth)])  # do not use list multiple problem, which is shared reference
        self.fc = nn.Linear(in_features=D, out_features=D)
        self.pe = PositionalEncoding(D=D)

    def forward(self, q, q_mask, v, v_mask):
        """
        in: q.shape=(B, N, D)
        """
        out = q
        out = self.pe.forward(out)
        for layer in self.layers:
            out = layer(q=out, self_mask=q_mask, k=None, v=v, cross_mask=v_mask)
        out = self.fc(out)
        return out


# class Transformer(nn.Module):
#     def __init__(self, num_head, depth, D, dropout):
#         self.enc = Encoder(num_head=num_head, depth=depth, D=D, dropout=dropout)
#         self.dec = Encoder(num_head=num_head, depth=depth, D=D, dropout=dropout)

#     def forward(self, q):
#         v = self.enc(q)
#         self.dec(q, v)

if __name__ == '__main__':
    torch.manual_seed(0)
    model = Encoder(num_head=8, depth=2, D=128, dropout=0.1)
    x = torch.randn(4, 16, 128)  # batch=4, seq_len=16, embed_dim=128
    out = model(x)
    print(out.shape)  # (4, 16, 128)
