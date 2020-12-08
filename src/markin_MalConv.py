import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConv_markin(nn.Module):
    '''
    Implementation of the MalConv Model
    [Link to Paper] : https://arxiv.org/pdf/1710.09435.pdf
    '''

    def __init__(self, config):

        super(MalConv_markin, self).__init__()

        self.byte_embedding = nn.Embedding(257, config.embedding_dim, padding_idx=0)

        self.conv_1 = nn.Conv1d(
            in_channels=config.embedding_dim,
            out_channels=config.conv_out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            bias=True
        )

        self.conv_2 = nn.Conv1d(
            in_channels=config.embedding_dim,
            out_channels=config.conv_out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            bias=True
        )

        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.fc_1 = nn.Linear(config.conv_out_channels, config.conv_out_channels)
        self.fc_2 = nn.Linear(config.conv_out_channels, config.output_dim)

    def forward(self, X):

        #X = self.byte_embedding(X)
        X = torch.transpose(X, -1, -2)

        conv_out = self.conv_1(X)
        g_weight = torch.sigmoid(self.conv_2(X))

        X = conv_out * g_weight
        X = self.pooling(X)

        X = X.view(X.size(0), -1)
        X = F.relu(self.fc_1(X))
        X = self.fc_2(X)

        return X
