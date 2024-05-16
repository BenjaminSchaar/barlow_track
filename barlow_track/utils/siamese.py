import torch.nn as nn
from pytorch3dunet.unet3d.buildingblocks import ExtResNetBlock, create_encoders
from pytorch3dunet.unet3d.utils import number_of_features_per_level


class Siamese(nn.Module):

    def __init__(self, embedding_dim=4096):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm3d(1),
            nn.Conv3d(1, 64, 4, padding=3),  # 64@8*64*64
            # nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),  # 64@4*32*32
            nn.Conv3d(64, 128, 4),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),  # 128@2*16*16
            nn.Conv3d(128, 128, (1, 2, 2)),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU(),    # 128@1*14*14
        )
        self.projection = nn.Sequential(nn.Linear(128 * 1 * 14 * 14, embedding_dim), nn.Sigmoid())
        self.out = nn.Linear(embedding_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.projection(x)
        return x


class SiameseResNet(nn.Module):

    def __init__(self, embedding_dim=4096):
        super(SiameseResNet, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm3d(1),
            nn.Conv3d(1, 64, 4, padding=3),  # 64@8*64*64
            # nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),  # 64@4*32*32
            nn.Conv3d(64, 128, 4),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),  # 128@2*16*16
            nn.Conv3d(128, 128, (1, 2, 2)),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU(),    # 128@1*14*14
        )
        self.projection = nn.Sequential(nn.Linear(128 * 1 * 14 * 14, embedding_dim), nn.Sigmoid())
        self.out = nn.Linear(embedding_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.projection(x)  # Brings 3d output to 1d
        return x


class Abstract3DEncoder(nn.Module):
    """
    From: https://github.com/wolny/pytorch-3dunet

    Base class for standard and residual UNet BUT no decoders

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, basic_module, crop_sz, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, embedding_dim=2048, **kwargs):
        super(Abstract3DEncoder, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[-1], 1, 1)
        # Final conv should have
        self.projection = nn.Sequential(nn.Linear(int(crop_sz.prod()/8), embedding_dim), nn.Sigmoid())

        self.embedding_dim = embedding_dim
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    # def get_output_shape(self):
    #     linear_layer = next(self.backbone.projection.children())
    #     return linear_layer.weight.shape[-1]

    def forward(self, x):
        for encoder in self.encoders:
            # print(x.shape)
            x = encoder(x)
        # print(x.shape)
        x = self.final_conv(x)
        # print(x.shape)
        x = x.view(x.size()[0], -1)  # Compress everything except the batch
        # print(x.shape)
        x = self.projection(x)
        return x


class ResidualEncoder3D(Abstract3DEncoder):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, crop_sz, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, conv_padding=1, embedding_dim=2048, **kwargs):
        super(ResidualEncoder3D, self).__init__(in_channels=in_channels,
                                                basic_module=ExtResNetBlock,
                                                crop_sz=crop_sz,
                                                f_maps=f_maps,
                                                layer_order=layer_order,
                                                num_groups=num_groups,
                                                num_levels=num_levels,
                                                conv_padding=conv_padding,
                                                embedding_dim=embedding_dim,
                                                **kwargs)


class ResidualClassifier3D(Abstract3DEncoder):
    """
    Same as ResidualEncoder3D, but adds a classification layer
    """

    def __init__(self, num_categories, in_channels, crop_sz, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, conv_padding=1, embedding_dim=2048, **kwargs):
        super(ResidualClassifier3D, self).__init__(in_channels=in_channels,
                                                   basic_module=ExtResNetBlock,
                                                   crop_sz=crop_sz,
                                                   f_maps=f_maps,
                                                   layer_order=layer_order,
                                                   num_groups=num_groups,
                                                   num_levels=num_levels,
                                                   conv_padding=conv_padding,
                                                   embedding_dim=embedding_dim,
                                                   **kwargs)

        # Only classify one at a time, in principle
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_categories),
            nn.Softmax()
        )

    def forward(self, x):
        x = super().forward(x)
        return self.classifier(x)

