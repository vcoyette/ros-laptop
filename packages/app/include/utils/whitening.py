import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Whitening(nn.Module):
    def __init__(
        self, num_features, group_size, momentum=0.1, eps=1e-6,
    ):
        super(_Whitening, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.group_size = min(self.num_features, group_size)
        self.num_groups = self.num_features // self.group_size

        self.register_buffer(
            "running_mean", torch.zeros(self.num_features),
        )
        self.register_buffer(
            "running_variance",
            torch.ones(self.num_groups, self.group_size, self.group_size),
        )

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _check_group_size(self):
        raise NotImplementedError

    def forward(self, x):
        self._check_input_dim(x)
        self._check_group_size()

        # Register shape
        m, d, h, w = x.shape

        # (m, d, h, w) -> (d, m, h, w)
        x = x.permute(1, 0, 2, 3).contiguous()

        if self.training:
            # Mean over (m, h, w) to obtain mean by channel
            mu = x.mean([1, 2, 3])
        else:
            # Use running_mean for inference
            mu = self.running_mean

        # Center x
        x_centered = x - mu[:, None, None, None]

        # Reshape X by groups and flatten pixels: (d, m, h, w) -> (n, g, m * h * w)
        x_centered = x_centered.view(self.num_groups, self.group_size, -1)

        # Identity matrix repeated for each group
        E = torch.eye(self.group_size).repeat(self.num_groups, 1, 1).to(device)

        if self.training:
            # Compute covariance matrix inside each group
            sigma_b_hat = torch.bmm(x_centered, x_centered.transpose(-1, -2)) / (
                m * h * w - 1
            )

            # Add epsilon for numerical stability
            sigma_b = (1 - self.eps) * sigma_b_hat + self.eps * E
        else:
            # Use running_variance for inference, and add epsilon for stability
            sigma_b = (1 - self.eps) * self.running_variance + self.eps * E

        # Get lower triangular T such that T.T^t = sigma_b
        T = torch.cholesky(sigma_b)

        # Inverse T to obtain W_b
        W_b = torch.inverse(T)

        # Compute decorrelated x
        x_hat = torch.bmm(W_b, x_centered)

        # (n, g, m * h * w) -> (d, m, h, w)
        x_hat = x_hat.view(d, m, h, w)

        # (d, m, h, w) -> (m, d, h, w)
        x_hat = x_hat.permute(1, 0, 2, 3).contiguous()

        # Update Running stats
        if self.training:
            self.running_mean = (
                self.momentum * mu.detach() + (1 - self.momentum) * self.running_mean
            )
            self.running_variance = (
                self.momentum * sigma_b_hat.detach()
                + (1 - self.momentum) * self.running_variance
            )

        return x_hat


class WTransform2d(_Whitening):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def _check_group_size(self):
        if self.num_features % self.group_size != 0:
            raise ValueError(
                "expected number of channels divisible by group_size (got {} group_size\
				for {} number of features".format(
                    self.group_size, self.num_features
                )
            )
