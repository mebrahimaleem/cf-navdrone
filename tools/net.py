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
		return (torch.empty(8),)

	def gen_calib_data(self, n=100):
		return [{"inputs": torch.randn(8)} for i in range(n)]

	def input_names(self):
		return ["inputs"]

	def output_names(self):
		return ["outputs"]

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
		return (torch.empty(1, 1, 320, 320),)

	def gen_calib_data(self, n=100):
		return [{"inputs": torch.randn(1, 1, 320, 320)} for i in range(n)]

	def input_names(self):
		return ["inputs"]

	def output_names(self):
		return ["outputs"]

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
		return (torch.empty(1, 6),)

	def gen_calib_data(self, n=100):
		data = list()
		for i in range(n):
			first = torch.randn(n, 1, 4)
			second = torch.randint(low=0, high=320, size=(n, 1, 2)).float()
			data.append({"inputs": torch.cat([first, second], dim=-1)})
		return data

	def input_names(self):
		return ["inputs"]

	def output_names(self):
		return ["outputs"]

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
		return (torch.empty(1, 1, 1, 8, 6),)

	def gen_calib_data(self, n=100):
		return [{"inputs": torch.randn(1, 1, 1, 8, 6)} for i in range(n)]

	def input_names(self):
		return ["inputs"]

	def output_names(self):
		return ["outputs"]

class SimpleLSTMNet(nn.Module):
	def __init__(self):
		super(SimpleLSTMNet, self).__init__()

		self.lstm_cell = nn.LSTMCell(16, 32)
		self.fc = nn.Linear(32, 2)

		self.register_buffer("h", torch.zeros(1, 32))
		self.register_buffer("c", torch.zeros(1, 32))

	def forward(self, x):
		self.h, self.c = self.lstm_cell(x, (self.h, self.c))

		return self.fc(self.h)

	def example_inputs(self):
		return (torch.empty(1, 16),)

	def gen_calib_data(self, n=100):
		return [{"inputs": torch.randn(1, 16)} for i in range(n)]

	def input_names(self):
		return ["inputs"]

	def output_names(self):
		return ["outputs"]


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
		return (torch.empty(1, 1, 320, 320),)

	def gen_calib_data(self, n=100):
		return [{"inputs": torch.randn(1, 1, 320, 320)} for i in range(n)]

	def input_names(self):
		return ["inputs"]

	def output_names(self):
		return ["outputs"]

class CompleteNet(nn.Module):
	def __init__(self):
		super(CompleteNet, self).__init__()

		self.cnn = nn.Sequential(
			# 1x1x320x320
			nn.Conv2d(in_channels=1, out_channels=32, stride=4, kernel_size=3, padding=1),
			# 1x32x80x80
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# 1x32x40x40

			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			# 1x64x40x40
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# 1x64x20x20

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
			# 1x64x20x20
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# 1x128x10x10

			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
			# 1x128x10x10
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# 1x256x5x5

			nn.AdaptiveAvgPool2d((1, 1)),
			# 1x256x1x1
			nn.Flatten(),
			# 1x256
		)

		self.lstm_layer_1 = nn.LSTMCell(256, 256)
		self.lstm_layer_2 = nn.LSTMCell(256, 256)
		self.lstm_layer_3 = nn.LSTMCell(256, 256)
		self.fc = nn.Linear(256, 3)

	def forward(self, x):
		h0 = self.cnn(x)

		h1, _ = self.lstm_layer_1(F.relu(h0, inplace=True))
		h2, _ = self.lstm_layer_2(F.relu(h1, inplace=True))
		h3, _ = self.lstm_layer_3(F.relu(h2, inplace=True))

		return self.fc(F.relu(h3, inplace=True))

	def example_inputs(self):
		return (torch.empty(1, 1, 320, 320),)

	def gen_calib_data(self, n=100):
		return [{"inputs": torch.randn(1, 1, 320, 320)} for i in range(n)]

	def input_names(self):
		return ["inputs"]

	def output_names(self):
		return ["outputs"]

class UNet(nn.Module):
	class ConvLSTM(nn.Module):
		class ConvLSTMCell(nn.Module):
			def __init__(self, input_size, hidden_size, kernel_size, bias):
				super(UNet.ConvLSTM.ConvLSTMCell, self).__init__()

				self.hidden_size = hidden_size

				padding = kernel_size // 2

				self.rescale = nn.Conv2d(input_size, hidden_size, kernel_size=1)

				self.conv_i = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=bias)
				self.conv_f = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=bias)
				self.conv_o = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=bias)
				self.conv_g = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=bias)

			def forward(self, x, h_prev, s_prev):
				rescaled = self.rescale(x)
				comb_input = rescaled + h_prev

				i = torch.sigmoid(self.conv_i(comb_input))
				f = torch.sigmoid(self.conv_f(comb_input))
				o = torch.sigmoid(self.conv_o(comb_input))
				g = torch.relu(self.conv_g(comb_input))

				s_next = f * s_prev + i * g
				h_next = o * torch.relu(s_next)

				return h_next, s_next

		def __init__(self, num_layers, input_sizes, kernel_sizes, bias, output_size):
			super(UNet.ConvLSTM, self).__init__()

			if num_layers != len(input_sizes):
				raise ValueError(f"Number of input sizes ({len(input_sizes)}) must match number of layers ({num_layers})")

			if num_layers != len(kernel_sizes):
				raise ValueError(f"Number of kernel sizes ({len(kernel_sizes)}) must match number of layers ({num_layers})")

			hidden_sizes = input_sizes + (output_size,)

			cells = list()

			for i in range(num_layers):
				cells.append(UNet.ConvLSTM.ConvLSTMCell(hidden_sizes[i], hidden_sizes[i+1], kernel_sizes[i], bias))

			self.cells = nn.ModuleList(cells)

		def forward(self, x, *states):
			new_states = list()

			for i, cell in enumerate(self.cells):
				h_next, s_next = cell(x, states[2 * i], states[2 * i + 1])
				new_states.append(h_next)
				new_states.append(s_next)
				x = h_next

			return (x, *new_states,)


	def __init__(self):
		super(UNet, self).__init__()

		self.enc1 = nn.Sequential(
				nn.Conv2d(1, 8, kernel_size=3, stride=4, padding=1),
				nn.ReLU(True)
		)
		self.enc2 = nn.Sequential(
				nn.Conv2d(8, 16, kernel_size=3, padding=1),
				nn.ReLU(True)
		)

		self.convlstm = UNet.ConvLSTM(1,
															 (16,),
															 (3,),
															 True,
															 16)

		self.dec1 = nn.Sequential(
				nn.Conv2d(16, 8, kernel_size=3, padding=1),
				nn.ReLU(True)
		)
		self.dec2 = nn.Sequential(
				nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
				nn.ReLU(True)
		)

	def forward(self, x, *states):
		x = self.enc1(x)
		skip1 = x
		x = self.enc2(x)
		skip2 = x

		res = self.convlstm(x, *states)
		x = res[0]
		states = res[1:]

		x = x + skip2
		x = self.dec1(x)

		x = x + skip1
		x = self.dec2(x)
		return (x, *states,)

	def example_inputs(self):
		return (torch.empty(1, 1, 320, 320),
						torch.empty(1, 16, 80, 80),
						torch.empty(1, 16, 80, 80),
		)

	def gen_calib_data(self, n=100):
		data = list()
		for i in range(n):
			x = torch.randn(1, 1, 320, 320)
			h0 = torch.randn(1, 16, 80, 80)
			s0 = torch.randn(1, 16, 80, 80)
			data.append({
				"inputs": x,
				"h0": h0, 
				"s0": s0,
			})
		return data

	def input_names(self):
		return ["inputs", "h0", "s0"]

	def output_names(self):
		return ["outputs", "h0_prev", "s0_prev"]

net_table = {
	"SimpleLinear": SimpleLinearNet,
	"SimpleConv": SimpleConvNet,
	"SimpleContinuous": SimpleContinuousNet,
	"Simple8D": Simple8DNet,
	"SimpleLSTM": SimpleLSTMNet,
	"BigConv": BigConvNet,
	"Complete": CompleteNet,
	"UNet": UNet,
}
