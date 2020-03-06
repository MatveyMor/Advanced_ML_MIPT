import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_conv1_1 = nn.Conv1d(in_channels=hid_size, out_channels=2*hid_size, kernel_size=2)
        self.title_pool_overtime = nn.AdaptiveMaxPool1d(output_size=1)   
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_conv1_1 = nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=3)
        self.full_conv1_2 = nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=5, padding=1)
        self.full_conv1_3 = nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=7, padding=2)
        self.full_pool_overtime1 = nn.AdaptiveMaxPool1d(output_size=1)
        
        self.category_out = nn.Linear(in_features=n_cat_features, out_features=128)

        self.last_fc = nn.Linear(in_features=448, out_features=1)
        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input

        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_conv1_1(title_beg)
        title = nn.ReLU()(title)
        title = self.title_pool_overtime(title)

        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full_1_1 = self.full_conv1_1(full_beg)
        full_1_2 = self.full_conv1_2(full_beg)
        full_1_3 = self.full_conv1_3(full_beg)
        full_1 = torch.cat((full_1_1, full_1_2, full_1_3), dim=1)
        full_1 = nn.ReLU()(full_1)
        full_1 = self.full_pool_overtime1(full_1)

        
        category = self.category_out(input3)
        category = nn.ReLU()(category)
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full_1.view(full_1.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.last_fc(concatenated)     
        
        return out