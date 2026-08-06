import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, kernel_size, bias):
		super().__init__()

		padding = kernel_size // 2

		self.rescale = nn.Conv2d(input_size, hidden_size, kernel_size=1)

		self.conv_i = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding, bias=bias)
		self.conv_f = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding, bias=bias)
		self.conv_o = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding, bias=bias)
		self.conv_g = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding, bias=bias)

	def forward(self, x, h_prev, s_prev):
		rescaled = self.rescale(x)
		comb_input = torch.cat([rescaled, h_prev], 1)

		i = torch.sigmoid(self.conv_i(comb_input))
		f = torch.sigmoid(self.conv_f(comb_input))
		o = torch.sigmoid(self.conv_o(comb_input))
		g = torch.relu(self.conv_g(comb_input))

		s_next = f * s_prev + i * g
		h_next = o * torch.relu(s_next)

		return h_next, s_next

class ConvLSTM(nn.Module):
	def __init__(self, num_layers, input_sizes, kernel_sizes, bias, output_size):
		super().__init__()

		if num_layers != len(input_sizes):
			raise ValueError(f"Number of input sizes ({len(input_sizes)}) must match number of layers ({num_layers})")

		if num_layers != len(kernel_sizes):
			raise ValueError(f"Number of kernel sizes ({len(kernel_sizes)}) must match number of layers ({num_layers})")

		hidden_sizes = input_sizes + (output_size,)

		cells = list()

		for i in range(num_layers):
			cells.append(ConvLSTMCell(hidden_sizes[i], hidden_sizes[i+1], kernel_sizes[i], bias))

		self.cells = nn.ModuleList(cells)

	def forward(self, x, *states):
		new_states = list()

		for i, cell in enumerate(self.cells):
			h_next, s_next = cell(x, states[2 * i], states[2 * i + 1])
			new_states.append(h_next)
			new_states.append(s_next)
			x = h_next

		return (x, *new_states,)

class FastConvTranspose2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super().__init__()

		self.in_channels = in_channels
		self.stride = stride

		self.spatial_mix = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
		self.upscale = nn.Conv2d(in_channels, in_channels * stride ** 2, kernel_size=1)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

		self.permute_upscale()

	@torch.no_grad()
	def permute_upscale(self):
		w = self.upscale.weight.data
		w = w.reshape(self.in_channels, self.stride, self.stride, -1)
		w = w.permute(0, 2, 1, 3)
		w = w.reshape(self.in_channels * self.stride ** 2, -1, 1, 1)
		self.upscale.weight.data = w

		if self.upscale.bias is not None:
			b = self.upscale.bias.data
			b = b.reshape(self.in_channels, self.stride, self.stride)
			b = b.permute(0, 2, 1)
			b = b.reshape(-1)
			self.upscale.bias.data = b

	def forward(self, x):
		b, _, h, w = x.shape

		x = self.spatial_mix(x)
		x = self.upscale(x)
		x = x.reshape(b, self.in_channels, h * self.stride, w * self.stride)
		x = self.conv(x)

		return x

class Net(nn.Module):
	def __init__(self):
		super().__init__()

		# Nx1x320x320
		self.channel_split = nn.Conv2d(1, 2, kernel_size=1, padding=0, bias=True)
		self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

		# Nx1x160x160
		self.enc11 = nn.Conv2d(2, 16, kernel_size=3, padding=1, stride=2)
		# Nx16x80x80
		self.enc12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
		# Nx16x80x80
		self.enc13 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Nx16x40x40
		self.enc21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		# Nx32x40x40
		self.enc22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
		# Nx32x40x40
		self.enc23 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Nx32x20x20
		self.enc31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		# Nx64x20x20
		self.enc32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		# Nx64x20x20
		self.enc33 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Nx64x10x10
		self.enc41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		# Nx128x10x10
		self.enc42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

		# Nx128x10x10
		self.bottleneck = ConvLSTM(4, (128,) * 4, (3,) * 4, True, 128)

		# Nx128x10x10
		self.dec41 = FastConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
		# Nx64x20x20
		self.dec42 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
		# Nx64x20x20
		self.dec43 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

		# Nx64x20x20
		self.dec31 = FastConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
		# Nx32x40x40
		self.dec32 = nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1)
		# Nx32x40x40
		self.dec33 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

		# Nx32x40x40
		self.dec21 = FastConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
		# Nx16x80x80
		self.dec22 = nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1)
		# Nx16x80x80
		self.dec23 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

		# Nx16x80x80
		self.dec11 = nn.Conv2d(16, 8, kernel_size=1, padding=0)
		# Nx8x80x80
		self.dec12 = FastConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1)
		# Nx1x160x160

		# TODO: maybe share latent from encoder
		# Nx1x160x160
		self.vel_head = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),
			# Nx16x80x80
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 16, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 16, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			# Nx16x40x40
			nn.Conv2d(16, 64, kernel_size=3, padding=1),
			# Nx64x40x40
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			# Nx64x20x20
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			# Nx128x20x20
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),

			# Nx128x1x1
			nn.Flatten(),
			nn.Linear(128, 16),
			nn.ReLU(inplace=True),
			nn.Linear(16, 4)
		)

	def forward(self, x, *states):
		x = self.channel_split(x)
		x = self.downsample(x)

		x = self.enc11(x)
		x = F.relu(x, inplace=True)
		x = self.enc12(x)
		x = skip2 = F.relu(x, inplace=True)
		x = self.enc13(x)

		x = self.enc21(x)
		x = F.relu(x, inplace=True)
		x = self.enc22(x)
		x = skip3 = F.relu(x, inplace=True)
		x = self.enc23(x)

		x = self.enc31(x)
		x = F.relu(x, inplace=True)
		x = self.enc32(x)
		x = skip4 = F.relu(x, inplace=True)
		x = self.enc33(x)

		x = self.enc41(x)
		x = F.relu(x, inplace=True)
		x = self.enc42(x)
		x = F.relu(x, inplace=True)

		x, *states = self.bottleneck(x, *states)

		x = self.dec41(x)
		x = F.relu(x, inplace=True)
		x = torch.cat([x, skip4], dim=1)
		x = self.dec42(x)
		x = F.relu(x, inplace=True)
		x = self.dec43(x)
		x = F.relu(x, inplace=True)

		x = self.dec31(x)
		x = F.relu(x, inplace=True)
		x = torch.cat([x, skip3], dim=1)
		x = self.dec32(x)
		x = F.relu(x, inplace=True)
		x = self.dec33(x)
		x = F.relu(x, inplace=True)

		x = self.dec21(x)
		x = F.relu(x, inplace=True)
		x = torch.cat([x, skip2], dim=1)
		x = self.dec22(x)
		x = F.relu(x, inplace=True)
		x = self.dec23(x)
		x = F.relu(x, inplace=True)

		x = self.dec11(x)
		x = F.relu(x, inplace=True)
		x = depth = self.dec12(x)

		x = vel = self.vel_head(x)

		return (vel, depth, *states,)

	def example_inputs(self):
		return (torch.empty(1, 1, 320, 320),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10),
						torch.empty(1, 128, 10, 10)
		)

	def gen_calib_data(self, n=100):
		data = list()
		for i in range(n):
			x = torch.randn(1, 1, 320, 320)
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
		return ["vel", "depth", "h0_next", "s0_next", "h1_next", "s1_next", "h2_next", "s2_next", "h3_next", "s3_next"]

