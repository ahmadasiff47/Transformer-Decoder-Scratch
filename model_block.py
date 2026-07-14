import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Linear layers for projecting input to Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project inputs to Q, K, V spaces
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Calculate scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(embed_dim)
        
        # Apply mask to block future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        output = torch.matmul(attn_weights, V)
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.attention = SimpleAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )
        
    def forward(self, x):
        seq_len = x.shape[1]
        
        # Create lower triangular causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        
        # Attention sublayer with residual connection
        attn_out = self.attention(x, mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-Forward sublayer with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


if __name__ == "__main__":
    # Test with standard dummy inputs
    batch_size = 2
    seq_len = 5
    d_model = 64
    
    # Initialize input tensor
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize and run the decoder block
    decoder = TransformerDecoderBlock(d_model=d_model, ffn_dim=256)
    output = decoder(dummy_input)
    
    print("\n" + "="*40)
    print("     DECODER BLOCK INITIALIZED      ")
    print("="*40)
    print(f"Input Shape  : {list(dummy_input.shape)}")
    print(f"Output Shape : {list(output.shape)}")
    print("Status       : Execution verified successfully.")
    print("="*40 + "\n")