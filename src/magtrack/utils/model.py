import torch
import torch.nn as nn
import torch.nn.functional as F


class ColocationCNN(nn.Module):
    """Siamese network that compares the two signals over time before pooling.

    A shared conv encoder maps each signal to a feature sequence (B, C, L'). Because both signals
    pass through the same layers, the two sequences stay time-aligned and are compared per time
    step via [|f1 - f2|, f1 * f2]. A conv head over this comparison sequence is then reduced by
    global average + max pooling and classified by the FC layers. `max_head_channels` caps the
    width of the head. `input_channels` is kept for compatibility; the encoder always processes
    one signal. The output is symmetric in (x1, x2): one logit per pair, 1 = not colocated.
    """

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
            max_head_channels=128,
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

        # Shared encoder: one signal (B, 1, L) -> feature sequence (B, C, L')
        in_channels = 1
        if self.num_conv_layers > 0:
            conv_blocks = []
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

        # Head over the per-time-step comparison [|f1 - f2|, f1 * f2] (B, 2C, L')
        head_channels = min(max(2 * in_channels, 16), max_head_channels)
        self.compare = nn.Sequential(
            nn.Conv1d(2 * in_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(head_channels),
            nn.ReLU(),
        )

        # Build fully connected layers on the globally average + max pooled head output and comparison
        self.fc_layers = nn.ModuleList()
        prev_dim = 2 * head_channels + 2 * (2 * in_channels)

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

    def forward(self, x1, x2):
        if x1.size(1) != self.target_signal_length:
            x1 = F.interpolate(x1.unsqueeze(1), size=self.target_signal_length, mode="nearest-exact",
                               align_corners=None).squeeze(1)
        if x2.size(1) != self.target_signal_length:
            x2 = F.interpolate(x2.unsqueeze(1), size=self.target_signal_length, mode="nearest-exact",
                               align_corners=None).squeeze(1)

        # Encode both signals in one pass so BatchNorm sees the same batch statistics for both sides.
        x = torch.cat((x1, x2), dim=0).unsqueeze(1)  # Shape: (2 * batch_size, 1, L)

        if self.num_conv_layers > 0:
            x = self.cnn(x)

        f1, f2 = x[:x1.size(0)], x[x1.size(0):]
        c = torch.cat((torch.abs(f1 - f2), f1 * f2), dim=1)  # Shape: (batch_size, 2C, L')
        h = self.compare(c)
        x = torch.cat((h.mean(dim=2), h.amax(dim=2), c.mean(dim=2), c.amax(dim=2)), dim=1)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # No activation after embedding layer
        x = self.embedding_layer(x)

        x = self.classifier(x)
        return x
