from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn import build_norm_layer
import torch.nn as nn
import numpy as np
import torch


def get_sinusoid_encoding_table(tra_length, embed_dims):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (dim // 2) / embed_dims) for dim in range(embed_dims)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(tra_length)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dims, tra_length):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', get_sinusoid_encoding_table(tra_length, embed_dims))
        # (1, tra_length, embed_dims)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoderLayer(BaseModule):
    """
    Args:
        drop_rate: A Dropout layer after nn.MultiheadAttention.
                   & Probability of an element to be zeroed in FFN.
        attn_drop_rate: A Dropout layer in nn.MultiheadAttention.
        drop_path_rate: The dropout_layer used when adding the shortcut
                        in MultiheadAttention & FFN.
    """
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate, attn_drop_rate,
                 drop_path_rate, num_fcs, qkv_bias, act_cfg, norm_cfg, batch_first):
        super(TransformerEncoderLayer, self).__init__()

        if act_cfg is None:
            act_cfg = dict(type='GELU')
        if norm_cfg is None:
            norm_cfg = dict(type='LN')

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(embed_dims=embed_dims, num_heads=num_heads,
                                       attn_drop=attn_drop_rate, proj_drop=drop_rate,
                                       dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                                       batch_first=batch_first, bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=num_fcs,
                       ffn_drop=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                       act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        x = self.attn(self.norm1(x), identity=x)
        x = self.ffn(self.norm2(x), identity=x)
        return x


class IKTransformer(BaseModule):
    def __init__(self, tra_length, num_layers, embed_dims, num_heads, feedforward_channels, drop_rate, attn_drop_rate,
                 drop_path_rate, num_fcs=2, qkv_bias=True, act_cfg=None, norm_cfg=None, batch_first=True):
        super(IKTransformer, self).__init__()

        self.eatv_linear = nn.Linear(6, embed_dims)

        self.pe = PositionalEncoding(embed_dims, tra_length)

        self.transformer_layers = ModuleList()
        for i in range(num_layers):
            self.transformer_layers.append(
                TransformerEncoderLayer(embed_dims=embed_dims, num_heads=num_heads,
                                        feedforward_channels=feedforward_channels,
                                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=drop_path_rate, num_fcs=num_fcs, qkv_bias=qkv_bias,
                                        act_cfg=act_cfg, norm_cfg=norm_cfg, batch_first=batch_first)
            )

        self.js_linear = nn.Linear(embed_dims, 6)

    def forward(self, eatv):
        eatv_embedding = self.eatv_linear(eatv)

        eatv_embedding = self.pe(eatv_embedding)

        for index, tr_layer in enumerate(self.transformer_layers):
            eatv_embedding = tr_layer(eatv_embedding)
            if index == 0:
                shallow_embedding_sequence = eatv_embedding
            if index == len(self.transformer_layers) - 1:
                deep_embedding_sequence = eatv_embedding

        js = self.js_linear(eatv_embedding)

        return js, shallow_embedding_sequence, deep_embedding_sequence


# if __name__ == '__main__':
#     import torch
#
#     input_data = torch.rand(12, 32, 6)
#
#     ik_transformer = IKTransformer(32, 4, 64, 2, 128, 0, 0, 0)
#
#     output_data = ik_transformer(input_data)
#
#     print(output_data.shape)  # torch.Size([12, 32, 6])
