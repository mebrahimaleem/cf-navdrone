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


	class ResBlock(nn.Module):
		def __init__(self, in_channels, out_channels):
			super(UNet.ResBlock, self).__init__()

			self.net = nn.Sequential(
					nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),

					nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
					nn.ReLU(inplace=True),
					nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
					nn.ReLU(inplace=True),
					nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
					nn.ReLU(inplace=True)
			)

		def forward(self, x):
			return self.net(x)

	class ResBlockTranspose(nn.Module):
		def __init__(self, in_channels, out_channels):
			super(UNet.ResBlockTranspose, self).__init__()

			self.net = nn.Sequential(
					nn.ReLU(inplace=True),
					nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),

					nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
					nn.ReLU(inplace=True),
					nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
					nn.ReLU(inplace=True),
					nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
			)

		def forward(self, x):
			return self.net(x)


	def __init__(self):
		super(UNet, self).__init__()

		self.enc1_conv = UNet.ResBlock(1, 16)
		self.enc2_conv = UNet.ResBlock(16, 32)
		self.enc3_conv = UNet.ResBlock(32, 64)
		self.enc4_conv = UNet.ResBlock(64, 128)

		self.convlstm = UNet.ConvLSTM(4, (128,) * 4, (3,) * 4, True, 128)

		self.dec4_conv_trans = UNet.ResBlockTranspose(128, 64)
		self.dec3_conv_trans = UNet.ResBlockTranspose(64, 32)
		self.dec2_conv_trans = UNet.ResBlockTranspose(32, 16)
		self.dec1_conv_trans = UNet.ResBlockTranspose(16, 1)

		self.head = nn.Sequential(
				nn.Conv2d(1, 16, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(16, 16, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(16, 16, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(16, 32, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(32, 32, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(32, 32, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.Conv2d(32, 64, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(64, 64, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(64, 64, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2),

				nn.AdaptiveAvgPool2d((1, 1)),
				nn.Flatten(),
				nn.Linear(64, 8),
				nn.ReLU(inplace=True),
				nn.Linear(8, 4),
		)

	def forward(self, x, *states):
		x = skip1 = self.enc1_conv(x)
		x = skip2 = self.enc2_conv(x)
		x = skip3 = self.enc3_conv(x)
		x = skip4 = self.enc4_conv(x)
		x, *states = self.convlstm(x, *states)
		x = self.dec4_conv_trans(x + skip4)
		x = self.dec3_conv_trans(x + skip3)
		x = self.dec2_conv_trans(x + skip2)
		x = depth = self.dec1_conv_trans(x + skip1)
		x = v = self.head(x)

		return (v, depth, *states,)

	def example_inputs(self):
		return (torch.empty(1, 1, 160, 160),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
		)

	def gen_calib_data(self, n=100):
		data = list()
		for i in range(n):
			x = torch.randn(1, 1, 160, 160)
			h0 = torch.randn(1, 128, 10, 10)
			s0 = torch.randn(1, 128, 10, 10)
			h1 = torch.randn(1, 128, 10, 10)
			s1 = torch.randn(1, 128, 10, 10)
			h2 = torch.randn(1, 128, 10, 10)
			s2 = torch.randn(1, 128, 10, 10)
			h3 = torch.randn(1, 128, 10, 10)
			s3 = torch.randn(1, 128, 10, 10)
			data.append({
				"bem": x,
				"h0": h0, 
				"s0": s0,
				"h1": h1, 
				"s1": s1,
				"h2": h2, 
				"s2": s2,
				"h3": h3, 
				"s3": s3,
			})
		return data

	def input_names(self):
		return ["bem", "h0", "s0", "h1", "s1", "h2", "s2", "h3", "s3"]

	def output_names(self):
		return ["v", "depth", "h0_prev", "s0_prev", "h1_prev", "s1_prev", "h2_prev", "s2_prev", "h3_prev", "s3_prev"]

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
