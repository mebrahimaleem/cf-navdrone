import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLinearNet(nn.Module):
	def __init__(self):
		super(SimpleLinearNet, self).__init__()

		self.fc1 = nn.Linear(8, 8)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		return x

	def example_inputs(self):
		return torch.empty(8)

	def gen_calib_data(self, n=100):
		return torch.randn(n, 8)

class SimpleConv(nn.Module):
	def __init__(self):
		super(SimpleConv, self).__init__()

		self.net = nn.Sequential(
			# 1x16x80x80

			nn.Conv2d(in_channels=16, out_channels=32, stride=4, kernel_size=3, padding=1),
			# 1x32x20x20
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=4),
			# 1x32x5x5

			nn.Flatten(),
			# 1x800

			nn.Linear(800, 3),
			# 1x3
		)

	def forward(self, x):
		return self.net(x)

	def example_inputs(self):
		return torch.empty(1, 16, 80, 80)

	def gen_calib_data(self, n=100):
		return torch.randn(n, 1, 16, 80, 80)

net_table = {
	"SimpleLinearNet": SimpleLinearNet,
	"SimpleConv": SimpleConv
}
