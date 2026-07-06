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

class Simple8DNet(nn.Module):
	def __init__(self):
		super(Simple8DNet, self).__init__()

		class DimLinear(nn.Module):
			def __init__(self, i, o, d, post=F.relu):
				super(DimLinear, self).__init__()
				self.fc = nn.Linear(i, o)
				self.post = post
				self.d = d

			def forward(self, x):
				nd = x.dim()
				dims = list(range(x.dim()))
				dims[self.d], dims[-1] = dims[-1], dims[self.d]

				x = x.permute(*dims)
				x = self.fc(x)
				if self.post:
					x = self.post(x)
				x = x.permute(*dims)
				return x

		self.net = nn.Sequential(
			# 1x1x1x8x6

			DimLinear(1, 10, 1),
			# 1x10x1x8x6

			DimLinear(1, 10, 2),
			# 1x10x10x8x6

			nn.Flatten(),
			# 1x4800

			nn.Linear(4800, 3)
		)

	def forward(self, x):
		return self.net(x)

	def example_inputs(self):
		return torch.empty(1, 1, 1, 8, 6)

	def gen_calib_data(self, n=100):
		return torch.randn(n, 1, 1, 1, 8, 6)

class SimpleLSTMNet(nn.Module):
	def __init__(self):
		super(SimpleLSTMNet, self).__init__()

		self.lstm0 = nn.LSTMCell(16, 32)
		self.lstm1 = nn.LSTMCell(16, 32)
		self.lstm2 = nn.LSTMCell(16, 32)
		self.lstm3 = nn.LSTMCell(16, 32)
		self.fc = nn.Linear(32, 2)

		self.register_buffer("h", torch.zeros(1, 32))
		self.register_buffer("c", torch.zeros(1, 32))

	def forward(self, x):
		self.h, self.c = self.lstm0(x, (self.h, self.c))
		self.h, self.c = self.lstm1(x, (self.h, self.c))
		self.h, self.c = self.lstm2(x, (self.h, self.c))
		self.h, self.c = self.lstm3(x, (self.h, self.c))

		return self.fc(self.h)

	def example_inputs(self):
		return torch.empty(1, 16)

	def gen_calib_data(self, n=100):
		return torch.randn(n, 1, 16)


class BigConvNet(nn.Module):
	def __init__(self):
		super(BigConvNet, self).__init__()

		self.net = nn.Sequential(
			# 1x1x320x320

			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
			# 1x32x320x320
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# 1x32x160x160

			nn.AdaptiveAvgPool2d((1, 1)),
			# 1x128x1x1
			nn.Flatten(),
			# 1x128

			nn.Linear(16, 3),
			# 1x3
		)

	def forward(self, x):
		return self.net(x)

	def example_inputs(self):
		return torch.empty(1, 1, 320, 320)

	def gen_calib_data(self, n=100):
		return torch.randn(n, 1, 1, 320, 320)

net_table = {
	"SimpleLinear": SimpleLinearNet,
	"SimpleConv": SimpleConvNet,
	"SimpleContinuous": SimpleContinuousNet,
	"Simple8D": Simple8DNet,
	"SimpleLSTM": SimpleLSTMNet,
	"BigConv": BigConvNet,
}
