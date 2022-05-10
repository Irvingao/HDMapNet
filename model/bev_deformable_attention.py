import copy
import torch
import torch.nn as nn
import torch.functional as F

from .deformable_attention import DeformableAttention
# import sys
# import os
# sys.path.append(os.path.abspath(__file__))
# from deformable_attention import DeformableAttention

class BEVDeformableTransformerEncoder(nn.Module):
    def __init__(self, num_layers=6,d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",n_heads=8):
        super().__init__()
        decoder_layer = BEVDeformableTransformerLayer(d_model,d_ffn,dropout,activation,n_heads)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, bev_feat):
        output = bev_feat
        for _, layer in enumerate(self.layers):
            output = layer(bev_feat)
        return output


class BEVDeformableTransformerLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()

        # Deformable Attention
        self.cross_attn = DeformableAttention(
                        dim = d_model,               # feature dimensions
                        heads = n_heads,                   # attention heads
                        dropout = dropout,                # dropout
                        downsample_factor = 4,       # downsample factor (r in paper)
                        offset_scale = 4,            # scale of offset, maximum offset
                        offset_groups = None,        # number of offset groups, should be multiple of heads
                        offset_kernel_size = 6,      # offset kernel size
                        )
        '''
        # feed forward network
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        print(d_model)
        '''
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        # ([1, 256, 64, 64]) -> ([1, d_ffn])
        tgt = self.flatten(tgt)
        print(f"linear0: {tgt.shape}")
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, bev_feat):
        """Forward Function of BEVCrossAttention.

        Args:
            bev_query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims). 

        Returns:
             Tensor: forwarded bev results with shape [num_query, bs, embed_dims].
        """
        
        # cross attention
        tgt = self.cross_attn(bev_feat)
        '''
        # feed forward network
        tgt = self.forward_ffn(tgt)
        '''
        return tgt

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == '__main__':
    model = BEVDeformableTransformerDecoder(num_layers=6)
    model.cuda()
    # print(model)
    x = torch.randn(1, 256, 64, 64).cuda()
    print(model(x).shape) # (1, 256, 64, 64)
    