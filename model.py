import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d0: int, d1: int, d_ffn: int, dropout=0.):
        super(MLP, self).__init__()
        self.d0 = d0
        self.d1 = d1

        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(d0 * d1, d_ffn)
        self.act = nn.GELU()
        self.dropout0 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_ffn, d0 * d1)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        x = self.flatten(x)
        x = self.fc0(x)
        x = self.act(x)
        x = self.dropout0(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        return x.reshape(-1, self.d0, self.d1)


class Block(nn.Module):
    def __init__(self, dim: tuple[int, int, int], d_ffn: int, dropout=0.):
        super(Block, self).__init__()
        self.dim = dim
        # 同一時間
        self.mlp1 = nn.ModuleList([MLP(dim[1], dim[2], d_ffn, dropout) for _ in range(dim[0])])
        # 同一関節
        self.mlp2 = nn.ModuleList([MLP(dim[0], dim[2], d_ffn, dropout) for _ in range(dim[1])])

    def forward(self, x, **kwargs):
        u = torch.stack(
            [self.mlp1[i](x[:, i, :, :]) for i in range(self.dim[0])],
            dim=1
        )

        v = torch.stack(
            [self.mlp2[j](x[:, :, j, :]) for j in range(self.dim[1])],
            dim=2
        )

        return x + (u + v) / 2


class ManyMLP(nn.Module):
    def __init__(self, dim: tuple[int, int, int], d_ffn, depth: int, dropout=0.):
        super(ManyMLP, self).__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([Block(dim, d_ffn, dropout) for _ in range(depth)])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim[0] * dim[1] * dim[2], dim[0] * dim[1] * dim[2])
        self.act = nn.Identity()

    def forward(self, x, **kwargs):
        for i in range(self.depth):
            x = self.blocks[i](x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.act(x)

        return x.reshape((-1, *self.dim))


if __name__ == '__main__':
    from torchsummary import summary

    JOINT_NUM = 21
    JOINT_DIM = 3

    FRAMES = 45
    D_FFN = 32
    DEPTH = 16

    model = ManyMLP((FRAMES+1, JOINT_NUM, JOINT_DIM), d_ffn=D_FFN, depth=DEPTH)
    input_vec = torch.randn(10, FRAMES+1, JOINT_NUM, JOINT_DIM, dtype=torch.float32).to('cuda')

    summary(model, input_vec, depth=100)
