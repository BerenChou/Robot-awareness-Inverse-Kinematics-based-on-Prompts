import torch.nn as nn
import torch


class ContrastiveLikeLoss(nn.Module):
    def __init__(self, temperature, keep_num):
        super(ContrastiveLikeLoss, self).__init__()
        self.temperature = temperature
        self.keep_num = keep_num

    def forward(self, shallow_embedding_sequence, deep_embedding_sequence):  # (B, 100, 128)

        batch_size = shallow_embedding_sequence.shape[0]

        random_indices = torch.randperm(shallow_embedding_sequence.shape[1])[:self.keep_num]
        shorter_shallow_embedding_sequence = torch.index_select(shallow_embedding_sequence, 1, random_indices)  # (B, 30, 128), 以30为例
        shorter_deep_embedding_sequence = torch.index_select(deep_embedding_sequence, 1, random_indices)  # (B, 30, 128), 以30为例

        dot_product_tempered = shorter_shallow_embedding_sequence @ shorter_deep_embedding_sequence.permute(0, 2, 1) / self.temperature  # (B, 30, 30)

        exp_dot_product_tempered = torch.exp(dot_product_tempered) + torch.tensor(1e-5)  # (B, 30, 30)

        numerator = torch.sum(
            exp_dot_product_tempered * torch.eye(self.keep_num).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        denominator = torch.sum(
            exp_dot_product_tempered * (1 - torch.eye(self.keep_num)).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        loss = - torch.log(numerator / denominator) / batch_size

        return loss


if __name__ == '__main__':
    shallow = torch.rand(3, 100, 128)
    deep = torch.rand(3, 100, 128)

    cl_loss = ContrastiveLikeLoss(temperature=0.2, keep_num=30)

    output = cl_loss(shallow, deep)

    print(output)
    print(output.shape)
