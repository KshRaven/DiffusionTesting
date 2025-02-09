
from build.model.base.sub import SelfAttention
from build.model.base.debugging import get_tensor_info

from torch import nn
from torch import Tensor

import torch


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, token_num: int, device='cpu', dtype=torch.float32):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_size, device=device, dtype=dtype)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((token_num, embed_size), device=device, dtype=dtype))
    
    def forward(self, tokens: Tensor):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding[:tokens.shape[-2]]
        
        return x


class CLIPLayer(nn.Module):
    def __init__(self, embed_size: int, heads: int, bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        
        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(embed_size, bias=bias, device=device, dtype=dtype)
        # Self attention
        self.attention = SelfAttention(embed_size, heads, bias, device, dtype)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(embed_size, bias=bias, device=device, dtype=dtype)
        # Feedforward layer
        self.linear_1 = nn.Linear(embed_size, 4 * embed_size, bias, device, dtype)
        self.linear_2 = nn.Linear(4 * embed_size, embed_size, bias, device, dtype)

    def forward(self, x, debug=False):
        # (Batch_Size, Seq_Len, Dim)
        residue = x
        
        ### SELF ATTENTION ###

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True, debug=debug)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self, vocab_size: int = 49408, embed_size: int = 768, max_token_num: int = 77, layers: int = 12,
                 heads: int = 12, bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        self.embedding = CLIPEmbedding(vocab_size, embed_size, max_token_num, device, dtype)

        self.layers = nn.ModuleList([
            CLIPLayer(embed_size, heads, bias, device, dtype) for _ in range(layers)
        ])

        self.layernorm = nn.LayerNorm(embed_size)
    
    def forward(self, tokens: Tensor, debug=False) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)
        if debug:
            print(get_tensor_info(state, "CLIP ~ Embedding"))

        # Apply encoder layers similar to the Transformer's encoder.
        for i, layer in enumerate(self.layers):
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        if debug:
            print(get_tensor_info(state, "CLIP ~ Transformation"))
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output


if __name__ == '__main__':
    test_enc = CLIP()
    test_enc.eval()

    with torch.no_grad():
        print(test_enc)
        test_tensor = torch.randint(0, 10, (1, 16))
        print(get_tensor_info(test_tensor, "Input"))
        test_res = test_enc(test_tensor, debug=True)
        print(get_tensor_info(test_res, "Output"))
