import torch.nn as nn
import torch


class ContrastiveLikeLoss(nn.Module):
    def __init__(self, batch_size, temperature, tra_length, keep_num, device):
        super(ContrastiveLikeLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.random_indices = torch.randperm(tra_length)[:keep_num].to(device)
        self.diagonal_matrix = torch.eye(keep_num).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        self.anti_diagonal_matrix = (1 - torch.eye(keep_num)).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    def forward(self, shallow_embedding_sequence, deep_embedding_sequence):  # (B, 100, 128)
        shorter_shallow_embedding_sequence = torch.index_select(shallow_embedding_sequence, 1, self.random_indices)  # (B, 30, 128), 以30为例
        shorter_deep_embedding_sequence = torch.index_select(deep_embedding_sequence, 1, self.random_indices)  # (B, 30, 128), 以30为例

        dot_product_tempered = shorter_shallow_embedding_sequence @ shorter_deep_embedding_sequence.permute(0, 2, 1) / self.temperature  # (B, 30, 30)

        # exp_dot_product_tempered = torch.exp(dot_product_tempered) + 1e-5  # (B, 30, 30)
        exp_dot_product_tempered = torch.exp(
            dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]
        ) + 1e-5  # (B, 30, 30)

        numerator = torch.sum(exp_dot_product_tempered * self.diagonal_matrix)
        denominator = torch.sum(exp_dot_product_tempered * self.anti_diagonal_matrix)
        loss = - torch.log(numerator / denominator) / self.batch_size
        return loss


# if __name__ == '__main__':
#     shallow = torch.rand(3, 100, 128)
#     deep = torch.rand(3, 100, 128)
#
#     cl_loss = ContrastiveLikeLoss(batch_size=3, temperature=1.0, tra_length=100,
#                                   keep_num=30, device=torch.device('cuda:0'))
#
#     output = cl_loss(shallow, deep)
#
#     print(output)
#     print(output.shape)
