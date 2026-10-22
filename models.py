"""Network definitions for Tasks 2, 3 and 4."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# >>> REPORT: smallcnn
class SmallCNN(nn.Module):
    """Task 2(c): small CNN, no batch norm (so the optimiser matters more)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),    # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.net(x)


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallBNCNN(nn.Module):
    """Task 3(a): small CNN with batch norm and optional dropout in the head."""
    def __init__(self, dropout=0.0, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(3, 32), conv_bn_relu(32, 32), nn.MaxPool2d(2),
            conv_bn_relu(32, 64), conv_bn_relu(64, 64), nn.MaxPool2d(2),
            conv_bn_relu(64, 128), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))
# <<< REPORT


# >>> REPORT: resnet
class ShortcutA(nn.Module):
    """Parameter-free shortcut (option A of He et al. 2016): subsample by the
    stride and zero-pad the extra channels. Using it means the plain and the
    residual networks have exactly the same parameters."""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.stride, self.extra = stride, cout - cin

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.extra > 0:
            x = F.pad(x, (0, 0, 0, 0, self.extra // 2, self.extra - self.extra // 2))
        return x


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, residual):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.residual = residual
        self.shortcut = ShortcutA(cin, cout, stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:                 # <-- the ONLY difference plain/residual
            out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    """Depth 6n+2: conv -> 3 stages of n blocks (16/32/64 ch) -> GAP -> FC."""
    def __init__(self, depth, residual=True, widths=(16, 32, 64), num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        self.conv = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(widths[0])
        blocks, cin = [], widths[0]
        for s, w in enumerate(widths):
            for j in range(n):
                stride = 2 if (s > 0 and j == 0) else 1
                blocks.append(BasicBlock(cin, w, stride, residual))
                cin = w
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(widths[-1], num_classes)
        for m in self.modules():          # Kaiming (He) initialisation
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
# <<< REPORT


# >>> REPORT: task4net
class Residual(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.f = nn.Sequential(conv_bn_relu(c, c), conv_bn_relu(c, c))
        self.shortcut = shortcut

    def forward(self, x):
        return x + self.f(x) if self.shortcut else self.f(x)


class Scale(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x * self.s


class Task4Net(nn.Module):
    """ResNet-9-style network scaled down to < 500k parameters."""
    def __init__(self, c=(32, 64, 96, 192), shortcut=True, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu(3, c[0]),
            conv_bn_relu(c[0], c[1]), nn.MaxPool2d(2), Residual(c[1], shortcut),   # 16x16
            conv_bn_relu(c[1], c[2]), nn.MaxPool2d(2), Residual(c[2], shortcut),   # 8x8
            conv_bn_relu(c[2], c[3]), nn.MaxPool2d(2),                              # 4x4
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(c[3], num_classes, bias=False), Scale(0.125))

    def forward(self, x):
        return self.net(x)
# <<< REPORT
