import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import batch_norm
from torch.nn.modules import module


#Original paper :

  #  @article{qi2016pointnet,
            #  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
            #  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
            #  journal={arXiv preprint arXiv:1612.00593},
            #  year={2016}
      #      }

            #

 # Code below heavily based on Kaolin implementation

class FeatureExtractor(nn.Module):
    def __init__(self,in_channels,feature_size,layer_dims,global_feat=True,activation= F.relu,batchnorm = True, transposed_input= False):
        super().__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.layer_dims = layer_dims
        self.global_feat = global_feat
        self.activation = activation
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
        x = x.view(-1, self.feature_size)

    
        if self.global_feat:
            return x

        x = x.view(-1, self.feat_size, 1).repeat(1, 1, n_points)
        return torch.cat((x, local_features), dim=1)

class Classifier(nn.Module):
    #Classify the global features extracted by the FeatureExtractor
    def __init__(self,
                 in_channels: int = 3,
                 feature_size: int = 1024,
                 num_classes: int =2,
                 dropout: float = 0.,
                 classifier_layer_dims = [512, 256],
                 feat_layer_dims = [64, 128],
                 activation=F.relu,
                 batchnorm: bool = True,
                 transposed_input: bool = False):
        super().__init__()

        
        if not isinstance(classifier_layer_dims, list):
            classifier_layer_dims = list(classifier_layer_dims)
        classifier_layer_dims.insert(0, feature_size)



        self.feature_extractor = FeatureExtractor(
            in_channels=in_channels, feature_size=feature_size,
            layer_dims=feat_layer_dims, global_feat=True,
            activation=activation, batchnorm=batchnorm,
            transposed_input=transposed_input
        )

        self.linear_layers = nn.ModuleList()
        if batchnorm:
            self.bn_layers = nn.ModuleList()
        for idx in range(len(classifier_layer_dims) - 1):
            self.linear_layers.append(nn.Linear(classifier_layer_dims[idx],
                                                classifier_layer_dims[idx + 1]))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(
                    classifier_layer_dims[idx + 1]))

        self.last_linear_layer = nn.Linear(classifier_layer_dims[-1],
                                           num_classes)

        self.final_dropout = nn.Dropout(p=0.7)
        # Store activation as a class attribute
        self.activation = activation

        # Dropout layer (if dropout ratio is in the interval (0, 1]).
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

        else:
            self.dropout = None

        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm

        self.transposed_input = transposed_input

    def forward(self, x):
        r"""Forward pass through the PointNet classifier.
        Args:
            x (torch.Tensor): Tensor representing a pointcloud
                (shape: :math:`B \times N \times D`, where :math:`B`
                is the batchsize, :math:`N` is the number of points
                in the pointcloud, and :math:`D` is the dimensionality
                of each point in the pointcloud).
                If self.transposed_input is True, then the shape is
                :math:`B \times D \times N`.
        """
        x = self.feature_extractor(x)
        for idx in range(len(self.linear_layers) - 1):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                    self.linear_layers[idx](x)))
            else:
                x = self.activation(self.linear_layers[idx](x))
        # For penultimate linear layer, apply dropout before batchnorm
        if self.dropout:
            if self.batchnorm:
                x = self.activation(self.bn_layers[-1](self.dropout(
                    self.linear_layers[-1](x))))
            else:
                x = self.activation(self.dropout(self.linear_layers[-1](x)))
        else:
            if self.batchnorm:
                x = self.activation(self.bn_layers[-1](
                    self.linear_layers[-1](x)))
            else:
                x = self.activation(self.linear_layers[-1](x))
        # TODO: Use dropout before batchnorm of penultimate linear layer
        x = self.last_linear_layer(x)
        # return F.log_softmax(x, dim=1)
        return x








