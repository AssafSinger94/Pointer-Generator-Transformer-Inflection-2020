import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model_utils import PositionalEncoding, _generate_square_subsequent_mask, Embedding

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, fcn_hidden_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding_dim = embedding_dim
        # Source and Encoder layers
        self.src_embed = Embedding(src_vocab_size, embedding_dim, padding_idx=2)
        self.src_pos_encoder = PositionalEncoding(embedding_dim)
        self.encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, fcn_hidden_dim, dropout)
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers, encoder_norm)

        # Target and Decoder layers
        self.tgt_embed = Embedding(tgt_vocab_size, embedding_dim, padding_idx=2)
        self.tgt_pos_encoder = PositionalEncoding(embedding_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(embedding_dim, num_heads, fcn_hidden_dim, dropout)
        decoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers, decoder_norm)
        # Final linear layer
        self.out = nn.Linear(embedding_dim, tgt_vocab_size)

        # Initialize masks
        self.src_mask = None
        self.tgt_mask = None
        self.mem_mask = None
        # Initialize weights of model
        self._reset_parameters()

    def _reset_parameters(self):
        """ Initiate parameters in the transformer model. """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, src, tgt, has_mask=True, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Take in and process masked source/target sequences.

		Args:
			src: the sequence to the encoder (required).
			tgt: the sequence to the decoder (required).
			src_mask: the additive mask for the src sequence (optional).
			tgt_mask: the additive mask for the tgt sequence (optional).
			memory_mask: the additive mask for the encoder output (optional).
			src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

		Shape:
			- src: :math:`(S, N, E)`. Starts as (N, S) and changed after embedding
			- tgt: :math:`(T, N, E)`. Starts as (N, T) and changed after embedding
			- src_mask: :math:`(S, S)`.
			- tgt_mask: :math:`(T, T)`.
			- memory_mask: :math:`(T, S)`.
			- src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

			Note: [src/tgt/memory]_mask should be filled with
			float('-inf') for the masked positions and float(0.0) else. These masks
			ensure that predictions for position i depend only on the unmasked positions
			j and are applied identically for each sequence in a batch.
			[src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
			that should be masked with float('-inf') and False values will be unchanged.
			This mask ensures that no information will be taken from position i if
			it is masked, and has a separate mask for each sequence in a batch.

			- output: :math:`(T, N, E)`.

			Note: Due to the multi-head attention architecture in the transformer model,
			the output sequence length of a transformer is same as the input sequence
			(i.e. target) length of the decode.

			where S is the source sequence length, T is the target sequence length, N is the
			batch size, E is the feature number

		Examples:
			output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
		"""
        # Create target mask for transformer if no appropriate one was created yet, created of size (T, T)
        if has_mask:
            if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.size(1):
                self.tgt_mask = _generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        else:
            self.tgt_mask = None

        # Source embedding and positional encoding, changes dimension (N, S) -> (N, S, E) -> (S, N, E)
        src_embed = self.src_embed(src).transpose(0, 1)
        src_embed = self.src_pos_encoder(src_embed)
        # Pass the source to the encoder
        memory = self.encoder(src_embed, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)
        # Target embedding and positional encoding, changes dimension (N, T) -> (N, T, E) -> (T, N, E)
        tgt_embed = self.tgt_embed(tgt).transpose(0, 1)
        tgt_embed = self.tgt_pos_encoder(tgt_embed)
        # Get output of decoder. Dimensions stay the same
        decoder_output = self.decoder(tgt_embed, memory, tgt_mask=self.tgt_mask, memory_mask=self.mem_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        # Add linear layer (log softmax is added in Cross Entropy Loss), (T, N, E) -> (T, N, tgt_vocab_size)
        output = self.out(decoder_output)
        # Change back batch and sequence dimensions, from (T, N, tgt_vocab_size) -> (N, T, tgt_vocab_size)
        return output.transpose(0, 1)


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

    # def forward(self, x):
    #     """ Adds positional encoding to the input.
    #         Input of dimensions (seq_len x batch_sz x embedding_dim).
    #         Adds positional encoding matrix (seq_len x 1 x embedding_dim) to every individual example in batch """
    #     x = x + self.pe[:x.size(0), :]
    #     return self.dropout(x)

# def Embedding(num_embeddings, embedding_dim, padding_idx):
#     m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
#     nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
#     nn.init.constant_(m.weight[padding_idx], 0)
#     return m


# def _generate_subsequent_mask(self, src_sz, tgt_sz):
    #     mask = (torch.triu(torch.ones(src_sz, tgt_sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask
    #
    # def _generate_square_subsequent_mask(self, sz):
    #     return self._generate_subsequent_mask(sz, sz)
