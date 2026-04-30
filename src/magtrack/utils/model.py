import torch
import torch.nn as nn
import torch.nn.functional as F


class ColocationCNN(nn.Module):
    def __init__(
            self,
            input_channels=2,
            signal_length=600,
            embedding_dim=256,
            num_conv_layers=3,
            conv_channels=None,
            kernel_sizes=None,
            strides=None,
            paddings=None,
            pool_kernel_sizes=None,
            pool_strides=None,
            fc_hidden_dims=None,
            fc_dropout=0.0,
    ):
        super(ColocationCNN, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.target_signal_length = signal_length

        # Defaults matching previous hard-coded values
        if conv_channels is None:
            conv_channels = [16, 32, 64][:num_conv_layers]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3][:num_conv_layers]
        if strides is None:
            strides = [2] * len(conv_channels)
        if paddings is None:
            paddings = [k // 2 for k in kernel_sizes]
        if pool_kernel_sizes is None:
            pool_kernel_sizes = [2] * len(conv_channels)
        if pool_strides is None:
            pool_strides = [2] * len(conv_channels)
        if fc_hidden_dims is None:
            fc_hidden_dims = [256]

        if not (len(conv_channels) == len(kernel_sizes) == len(strides) == len(paddings) == len(
                pool_kernel_sizes) == len(pool_strides)):
            raise ValueError("Conv/pool parameter lists must be the same length.")

        if self.num_conv_layers > 0:
            conv_blocks = []
            in_channels = input_channels
            for out_channels, kernel_size, stride, padding, pool_k, pool_s in zip(
                    conv_channels, kernel_sizes, strides, paddings, pool_kernel_sizes, pool_strides
            ):
                conv_blocks.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=pool_k, stride=pool_s),
                ])
                in_channels = out_channels

            self.cnn = nn.Sequential(*conv_blocks)

            # Calculate the size after convolutions and pooling
            self.conv_output_shape = self._get_conv_output_shape(2, signal_length)

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        if self.num_conv_layers == 0:
            prev_dim = input_channels * signal_length
        else:
            prev_dim = self.conv_output_shape

        for hidden_dim in fc_hidden_dims:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(fc_dropout) if fc_dropout > 0 else nn.Identity(),
                )
            )
            prev_dim = hidden_dim

        # Final embedding layer
        self.embedding_layer = nn.Linear(prev_dim, embedding_dim)

        # Classifier head
        self.classifier = nn.Linear(embedding_dim, 1)

        # Initialize weights using Kaiming (He) initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Kaiming initialization to Conv1d and Linear layers and
        sensible defaults for BatchNorm layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_conv_output_shape(self, input_channels, signal_length):
        """Calculate output shape after conv and pooling layers"""
        was_training = self.cnn.training
        self.cnn.eval()
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, signal_length)
            print(f"Sample input shape for conv output calculation: {sample_input.shape}")
            sample_input = self.cnn(sample_input)
            output_dim = sample_input.view(1, -1).size(1)
        if was_training:
            self.cnn.train()
        return output_dim

    def forward(self, x1, x2):
        if x1.size(1) != self.target_signal_length:
            x1 = F.interpolate(x1.unsqueeze(1), size=self.target_signal_length, mode="nearest-exact",
                               align_corners=None).squeeze(1)
        if x2.size(1) != self.target_signal_length:
            x2 = F.interpolate(x2.unsqueeze(1), size=self.target_signal_length, mode="nearest-exact",
                               align_corners=None).squeeze(1)

        x = torch.stack((x1, x2), dim=1)  # Shape: (batch_size, 2, L)

        if self.num_conv_layers > 0:
            x = self.cnn(x)

        x = x.view(x.size(0), -1)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # No activation after embedding layer
        x = self.embedding_layer(x)

        x = self.classifier(x)
        return x
