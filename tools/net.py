import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLinearNet(nn.Module):
	def __init__(self):
		super(SimpleLinearNet, self).__init__()

		self.fc1 = nn.Linear(8, 2048)
		self.fc2 = nn.Linear(2048, 2048)
		self.fc3 = nn.Linear(2048, 8)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def example_inputs(self):
		return torch.empty(8)

	def gen_calib_data(self, n=100):
		return torch.randn(n, 8)

class SimpleConvNet(nn.Module):
	def __init__(self):
		super(SimpleConvNet, self).__init__()

		self.net = nn.Sequential(
			# 1x1x320x320

			nn.Conv2d(in_channels=1, out_channels=16, stride=2, kernel_size=3, padding=1),
			# 1x16x160x160
			nn.ReLU(),
			nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=3, padding=1),
			# 1x32x80x80
			nn.ReLU(),
			nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=3, padding=1),
			# 1x64x40x40
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
			# 1x64x1x1
			nn.Flatten(),
			# 1x64
			nn.Linear(64, 3),
			# 1x3
		)

	def forward(self, x):
		return self.net(x)

	def example_inputs(self):
		return torch.empty(1, 1, 320, 320)

	def gen_calib_data(self, n=100):
		return torch.randn(n, 1, 1, 320, 320)

class SimpleContinuousNet(nn.Module):
	def __init__(self):
		super(SimpleContinuousNet, self).__init__()

		self.proj = nn.Linear(6, 16)
		self.fc = nn.Linear(16, 16)
		self.head = nn.Linear(16, 3)
		self.relu = nn.ReLU()

		self.register_buffer("mem", torch.zeros((1, 16)))

	def forward(self, x):
		# x: 1x6

		x_proj = self.proj(x)
		comb = x_proj + self.mem

		mem = self.fc(comb)
		self.mem = mem.detach()

		return self.head(mem)

	def example_inputs(self):
		return torch.empty(1, 6)

	def gen_calib_data(self, n=100):
		first = torch.randn(n, 1, 4)
		second = torch.randint(low=0, high=320, size=(n, 1, 2)).float()
		return torch.cat([first, second], dim=-1)

net_table = {
	"SimpleLinear": SimpleLinearNet,
	"SimpleConv": SimpleConvNet,
	"SimpleContinuous": SimpleContinuousNet,
}
