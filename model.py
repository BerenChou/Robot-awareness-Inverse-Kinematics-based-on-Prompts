from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn import build_norm_layer
import torch.nn as nn


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
    def __init__(self, num_layers, embed_dims, num_heads, feedforward_channels, drop_rate, attn_drop_rate,
                 drop_path_rate, num_fcs=2, qkv_bias=True, act_cfg=None, norm_cfg=None, batch_first=True):
        super(IKTransformer, self).__init__()

        self.eatv_linear = nn.Linear(6, embed_dims)

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

        for tr_layer in self.transformer_layers:
            eatv_embedding = tr_layer(eatv_embedding)

        js = self.js_linear(eatv_embedding)

        return js


if __name__ == '__main__':
    import torch

    input_data = torch.rand(12, 32, 6)

    ik_transformer = IKTransformer(4, 64, 2, 128, 0, 0, 0)

    output_data = ik_transformer(input_data)

    print(output_data.shape)  # torch.Size([12, 32, 6])
