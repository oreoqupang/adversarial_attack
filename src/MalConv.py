import os
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MalConv1(nn.Module):
    # trained to minimize cross-entropy loss
    # criterion = nn.CrossEntropyLoss()
    def __init__(self, out_size=2, channels=128, window_size=512, embd_size=8):
        super(MalConv1, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        
        self.window_size = window_size
    
        self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        
        self.pooling = nn.AdaptiveMaxPool1d(1)
        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
    
    def get_embedding(self):
        return self.embd

    def calculate_loss(self, outputs):
        labels = torch.tensor([0]*len(outputs), dtype=torch.long).to(device)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def get_result(self, output, threshold):
        results = F.softmax(output, dim=1)[:, 1]
        return results > threshold

    def forward(self, x):
        
        #x = self.embd(x.long())
        x = torch.transpose(x,-1,-2)
        
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))
        
        x = cnn_value * gating_weight
        
        x = self.pooling(x)
        
        #Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        
        return x

class MalConv2(nn.Module):
    def __init__(self, pretrained_path=None, embedding_size=8, max_input_size=2**20):
        super(MalConv2, self).__init__()
        self.embedding_1 = nn.Embedding(num_embeddings=257, embedding_dim=embedding_size) #special padding value = 256
        self.conv1d_1 = nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=(500,), stride=(500,), groups=1, bias=True)
        self.conv1d_2 = nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=(500,), stride=(500,), groups=1, bias=True)
        self.dense_1 = nn.Linear(in_features=128, out_features=128, bias=True)
        self.dense_2 = nn.Linear(in_features=128, out_features=1, bias=True)
		
        if pretrained_path is not None:
            self.load_simplified_model(pretrained_path)
        if use_cuda:
            self.cuda()
            
    def embed(self, input_x, transpose=True):
        """
        It embeds an input vector into MalConv embedded representation.
        """
        if isinstance(input_x, torch.Tensor):
            x = input_x.type(torch.LongTensor)
        else:
            x = torch.autograd.variable(torch.from_numpy(input_x).type(torch.LongTensor))
        x = x.squeeze(dim=1)
        if use_cuda:
            x = x.cuda()
        emb_x = self.embedding_1(x)
        if transpose:
            emb_x = torch.transpose(emb_x, 1, 2)
        return emb_x
    
    def get_embedding(self):
        return self.embedding_1

    def get_result(self, output, threshold):
        result = output.squeeze() > threshold
        return result

    def calculate_loss(self, output):
        labels = torch.zeros_like(output).to(device)
        return nn.BCEWithLogitsLoss()(output, labels)
        
    def forward(self, x):
        #x = self.embedding_1(x)
        x = torch.transpose(x, 1, 2)
        
        conv1d_1 = self.conv1d_1(x)
        conv1d_2 = self.conv1d_2(x)
        conv1d_1_activation = torch.relu(conv1d_1)
        conv1d_2_activation = torch.sigmoid(conv1d_2)
        multiply_1 = conv1d_1_activation * conv1d_2_activation
        global_max_pooling1d_1 = F.max_pool1d(input=multiply_1, kernel_size=multiply_1.size()[2:])
        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(global_max_pooling1d_1.size(0), -1)
        dense_1 = self.dense_1(global_max_pooling1d_1_flatten)
        dense_1_activation = torch.relu(dense_1)
        dense_2 = self.dense_2(dense_1_activation)
        dense_2_activation = torch.sigmoid(dense_2)
        return dense_2_activation

class MalConv3(nn.Module):
    '''
    Implementation of the MalConv Model
    [Link to Paper] : https://arxiv.org/pdf/1710.09435.pdf
    '''

    def __init__(self, config):

        super(MalConv3, self).__init__()

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

    def get_embedding(self):
        return self.byte_embedding

    def get_result(self, output, threshold):
        result = torch.sigmoid(output).squeeze() > threshold
        return result

    def calculate_loss(self, output):
        labels = torch.zeros_like(output).to(device)
        return nn.BCEWithLogitsLoss()(output, labels)

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
