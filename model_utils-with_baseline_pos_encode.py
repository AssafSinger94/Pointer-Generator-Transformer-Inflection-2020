import math
import torch
import torch.nn as nn

""" MASKS UTILS """
def _generate_subsequent_mask(src_sz, tgt_sz):
    mask = (torch.triu(torch.ones(src_sz, tgt_sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _generate_square_subsequent_mask(sz):
    return _generate_subsequent_mask(sz, sz)


""" EMBEDDING UTILS """
def Embedding(num_embeddings, embedding_dim, padding_idx):
    """ Generates embeddings for tokens in vocabulary
        Weights initialized with mean=0 and std=sqrt(embedding_dim)"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


""" POSITIONAL ENCODING UTILS """
# class PositionalEncoding(nn.Module):
#     """ Adds positional encoding to sequences """
#     def __init__(self, embedding_dim, dropout=0.1, max_seq_len=100):
#         """ Initializes a seq_len x 1 x embedding_dim positional encoding matrix"""
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_seq_len, embedding_dim)
#         position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         """ Adds positional encoding to the input.
#             Input of dimensions (seq_len x batch_sz x embedding_dim).
#             Adds positional encoding matrix (seq_len x 1 x embedding_dim) to every individual example in batch """
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """
    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings,
                           dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        mask = input.ne(self.padding_idx).long()
        positions = torch.cumsum(mask, dim=0) * mask + self.padding_idx
        return self.weights.index_select(0, positions.view(-1)).view(
            bsz, seq_len, -1).detach()