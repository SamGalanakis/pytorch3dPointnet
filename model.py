import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import batch_norm


class FeatureExtractor(nn.Module):
    def __init__(self,in_channels,feature_size,layer_dims,global_feat=True,activation= F.relu,batchnorm = True, transposed_input= False):
        super().__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.layer_dims = layer_dims
        self.global_feat = global_feat
        self.activaton = activation
        self.batchnorm = batchnorm
        self.transposed_input = transposed_input

        self.layer_dims.insert(0,self.in_channels)
        self.layer_dims.append(feature_size)

        self.conv_layers = nn.ModuleList()

        if self.batchnorm:
            self.bn_layers = nn.ModuleList()

        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx],
                                                layer_dims[idx + 1], 1))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(layer_dims[idx + 1]))
    def forward(self,x):
        # Put points on "side" so sliding conv1d horizontally over each channel
        if not self.transposed_input:
            x = x.transpose(1,2)
        n_points = x.shape[2]

        local_features = None

        if self.batchnorm:
            x = self.activation(self.bn_layers[0](self.conv_layers[0](x)))
        else:
            x = self.activation(self.conv_layers[0](x))
        
        if self.global_feat is False:
            local_features = x
        
        for idx in range(1, len(self.conv_layers) - 1):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                self.conv_layers[idx](x)))
            else:
                x = self.activation(self.conv_layers[idx](x))

        if self.batchnorm:
            x = self.bn_layers[-1](self.conv_layers[-1](x))
        else:
            x = self.conv_layers[-1](x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feat_size)

    
        if self.global_feat:
            return x

        # x = x.view(-1, self.feat_size, 1).repeat(1, 1, num_points)
        # return torch.cat((x, local_features), dim=1)



