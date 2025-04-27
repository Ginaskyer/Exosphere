import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Double convolution block for U-Net.

        Parameters:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TimeSeriesUNet(nn.Module):
    def __init__(self, input_features, base_filters=64, depth=3):
        """
        U-Net architecture for time-series semantic segmentation.

        Parameters:
        input_features (int): Number of features per time step
        base_filters (int): Number of base filters (will be doubled in each layer)
        depth (int): Depth of the U-Net (number of down/up-sampling operations)
        """
        super(TimeSeriesUNet, self).__init__()

        # Save parameters
        self.depth = depth

        # Create encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()

        # First encoder block (input to base_filters)
        self.encoder_blocks.append(ConvBlock(input_features, base_filters))

        # Remaining encoder blocks
        for i in range(1, depth):
            in_channels = base_filters * (2 ** (i - 1))
            out_channels = base_filters * (2**i)
            self.encoder_blocks.append(ConvBlock(in_channels, out_channels))
            self.pool_blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Bridge
        bridge_channels = base_filters * (2 ** (depth - 1))
        self.bridge = ConvBlock(bridge_channels, bridge_channels * 2)

        # Create decoder blocks
        self.upconv_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Decoder blocks
        for i in range(depth - 1, -1, -1):
            in_channels = base_filters * (2 ** (i + 1))
            out_channels = base_filters * (2**i)

            self.upconv_blocks.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(ConvBlock(in_channels, out_channels))

        # Output layer
        self.output = nn.Conv1d(base_filters, 5, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Parameters:
        x (torch.Tensor): Input tensor of shape [batch_size, input_features, sequence_length]

        Returns:
        torch.Tensor: Binary classification output for each time step
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []

        # Encoder path
        for i in range(self.depth):
            if i == 0:
                enc_features = self.encoder_blocks[i](x)
            else:
                pooled = self.pool_blocks[i - 1](enc_features)
                enc_features = self.encoder_blocks[i](pooled)

            encoder_outputs.append(enc_features)

        # Bridge
        bridge_output = self.bridge(self.pool_blocks[-1](encoder_outputs[-1]))

        # Decoder path with skip connections
        decoder_output = bridge_output

        for i in range(self.depth):
            upconv = self.upconv_blocks[i](decoder_output)
            # Use encoder output from the opposite side of the U
            enc_features = encoder_outputs[self.depth - i - 1]
            concat = torch.cat([upconv, enc_features], dim=1)
            decoder_output = self.decoder_blocks[i](concat)

        # Output
        output = self.output(decoder_output)
        output = output.permute(0, 2, 1)
        return torch.softmax(output, dim=-1)
