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
    def __init__(self, tra_length, embed_dims, num_heads, feedforward_channels, drop_rate, attn_drop_rate,
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

        # # 考虑到解IK问题原则上并不需要序列信息, 且随机生成的轨迹也不含有什么有规律的序列信息.
        # # 所以这里对attention_matrix进行mask, 但貌似也没什么效果.
        # # 1. Transformer decoder-like Masked Multi-Head Attention
        # mask = (torch.triu(torch.ones(tra_length, tra_length)) == 1).transpose(0, 1)
        # self.attn_mask = mask.float().masked_fill(mask == 0, float('-inf'))\
        #     .masked_fill(mask == 1, float(0.0)).requires_grad_(False).clone().detach()
        #
        # # 2. Local Attention where one query only attends to neighboring keys instead of all of them
        # mask = abs(
        #     torch.arange(1, tra_length + 1).reshape(1, tra_length).repeat(tra_length, 1) -
        #     torch.arange(1, tra_length + 1).reshape(tra_length, 1).repeat(1, tra_length)
        # ) < 11  # 1 + 2 * (i - 1)个位置将会被attend, 即当i = 11时, 21个位置将会被attend
        # self.attn_mask = mask.float().masked_fill(mask == 0, float('-inf'))\
        #     .masked_fill(mask == 1, float(0.0)).requires_grad_(False).clone().detach()
        #
        # # 3. Combine the above two methods
        # mask1 = (torch.triu(torch.ones(tra_length, tra_length)) == 1).transpose(0, 1)
        # mask1 = mask1.float().masked_fill(mask1 == 0, float('-inf')).masked_fill(mask1 == 1, float(0.0))
        # mask2 = abs(
        #     torch.arange(1, tra_length + 1).reshape(1, tra_length).repeat(tra_length, 1) -
        #     torch.arange(1, tra_length + 1).reshape(tra_length, 1).repeat(1, tra_length)
        # ) < 20  # i个位置将会被attend, 即当i = 20时, 20个位置将会被attend
        # mask2 = mask2.float().masked_fill(mask2 == 0, float('-inf')).masked_fill(mask2 == 1, float(0.0))
        # self.attn_mask = (mask1 + mask2).requires_grad_(False).clone().detach()

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

        # self.attn_mask = self.attn_mask.to(x.device)
        # x = self.attn(self.norm1(x), identity=x, attn_mask=self.attn_mask)
        x = self.attn(self.norm1(x), identity=x)

        x = self.ffn(self.norm2(x), identity=x)
        return x


class IKTransformer(BaseModule):
    def __init__(self, tra_length, num_layers, embed_dims, num_heads, feedforward_channels, drop_rate, attn_drop_rate,
                 drop_path_rate, num_fcs=2, qkv_bias=True, act_cfg=None, norm_cfg=None, batch_first=True):
        super(IKTransformer, self).__init__()

        self.eatv2embedding_linear = nn.Linear(6, embed_dims)

        # # 多几个线性层
        # self.eatv_linear_1 = nn.Linear(6, 64)
        # self.relu1_between_eatv_linear = nn.ReLU()
        # self.eatv_linear_2 = nn.Linear(64, 128)
        # self.relu2_between_eatv_linear = nn.ReLU()
        # self.eatv_linear_3 = nn.Linear(128, embed_dims)

        # # ea与tv, 一个在joint-space一个在Cartesian-space, 可以将其视为多模态数据, 故采用两个不同的线性层
        # self.ea2embedding_linear = nn.Linear(3, embed_dims//2)
        # self.tv2embedding_linear = nn.Linear(3, embed_dims//2)

        self.pe = PositionalEncoding(embed_dims, tra_length)
        # self.learned_pe = nn.Parameter(torch.zeros(1, tra_length, embed_dims))  # 可学习的位置编码

        self.ik_transformer_layers = ModuleList()
        for _ in range(num_layers):
            self.ik_transformer_layers.append(
                TransformerEncoderLayer(tra_length=tra_length, embed_dims=embed_dims, num_heads=num_heads,
                                        feedforward_channels=feedforward_channels,
                                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=drop_path_rate, num_fcs=num_fcs, qkv_bias=qkv_bias,
                                        act_cfg=act_cfg, norm_cfg=norm_cfg, batch_first=batch_first)
            )

        self.embedding2ja_linear = nn.Linear(embed_dims, 6)

    def forward(self, eatv):

        embedding = self.eatv2embedding_linear(eatv)

        # embedding = self.eatv_linear_3(
        #     self.relu2_between_eatv_linear(
        #         self.eatv_linear_2(
        #             self.relu1_between_eatv_linear(
        #                 self.eatv_linear_1(eatv)
        #             )
        #         )
        #     )
        # )

        # ea = eatv[:, :, :3]
        # tv = eatv[:, :, 3:]
        # ea_embedding = self.ea2embedding_linear(ea)
        # tv_embedding = self.tv2embedding_linear(tv)
        # embedding = torch.concat([ea_embedding, tv_embedding], dim=2)

        embedding = self.pe(embedding)
        # embedding = embedding + self.learned_pe  # 加入可学习的位置编码

        for index, transformer_layer in enumerate(self.ik_transformer_layers):
            embedding = transformer_layer(embedding)
            # if index == 0:
            #     shallow_embedding = embedding
            # if index == len(self.ik_transformer_layers) - 1:
            #     deep_embedding = embedding

        ja = self.embedding2ja_linear(embedding)

        # return ja, shallow_embedding, deep_embedding
        return ja


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(MLP, self).__init__()

        self.eatv2ja_MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, output_dim),
        )

    def forward(self, eatv):
        return self.eatv2ja_MLP(eatv)


# class IKFKTransformer(BaseModule):
#     def __init__(self, tra_length, num_layers, embed_dims, num_heads, feedforward_channels, drop_rate, attn_drop_rate,
#                  drop_path_rate, num_fcs=2, qkv_bias=True, act_cfg=None, norm_cfg=None, batch_first=True):
#         super(IKFKTransformer, self).__init__()
#
#         self.eatv2embedding_linear = nn.Linear(6, embed_dims)
#
#         self.pe = PositionalEncoding(embed_dims, tra_length)
#
#         self.ik_transformer_layers = ModuleList()
#         for _ in range(num_layers):
#             self.ik_transformer_layers.append(
#                 TransformerEncoderLayer(embed_dims=embed_dims, num_heads=num_heads,
#                                         feedforward_channels=feedforward_channels,
#                                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
#                                         drop_path_rate=drop_path_rate, num_fcs=num_fcs, qkv_bias=qkv_bias,
#                                         act_cfg=act_cfg, norm_cfg=norm_cfg, batch_first=batch_first)
#             )
#
#         self.embedding2ja_linear = nn.Linear(embed_dims, 6)
#
#         self.ja2embedding_linear = nn.Linear(6, embed_dims)
#         self.leaky_relu = nn.LeakyReLU()
#
#         self.fk_transformer_layers = ModuleList()
#         for _ in range(num_layers):
#             self.fk_transformer_layers.append(
#                 TransformerEncoderLayer(embed_dims=embed_dims, num_heads=num_heads,
#                                         feedforward_channels=feedforward_channels,
#                                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
#                                         drop_path_rate=drop_path_rate, num_fcs=num_fcs, qkv_bias=qkv_bias,
#                                         act_cfg=act_cfg, norm_cfg=norm_cfg, batch_first=batch_first)
#             )
#         self.embedding2eatv_linear = nn.Linear(embed_dims, 6)
#
#     def forward(self, eatv):
#         embedding = self.eatv2embedding_linear(eatv)
#         embedding = self.pe(embedding)
#         for transformer_layer in self.ik_transformer_layers:
#             embedding = transformer_layer(embedding)
#         ja = self.embedding2ja_linear(embedding)
#
#         # # ja->embedding
#         # embedding = self.leaky_relu(self.ja2embedding_linear(ja))
#
#         for transformer_layer in self.fk_transformer_layers:
#             embedding = transformer_layer(embedding)
#         eatv = self.embedding2eatv_linear(embedding)
#
#         return ja, eatv
