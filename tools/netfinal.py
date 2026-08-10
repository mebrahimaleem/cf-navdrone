import copy
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

		i = F.sigmoid(self.conv_i(comb_input))
		f = F.sigmoid(self.conv_f(comb_input))
		o = F.sigmoid(self.conv_o(comb_input))
		g = F.relu(self.conv_g(comb_input), inplace=True)

		s_next = f * s_prev + i * g
		h_next = o * F.relu(s_next, inplace=True)

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

class VelPred(nn.Module):
	def __init__(self, in_channels, skip_decoder=False):
		super().__init__()

		if not skip_decoder:
			# 1x1x320x320
			self.cnn = nn.Sequential(
				nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=4),
				# 1x4x80x80
				nn.ReLU(inplace=True),
				nn.Conv2d(4, 16, kernel_size=3, padding=1),
				# 1x16x80x80
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2),
				# 1x16x40x40

				nn.Conv2d(16, 32, kernel_size=3, padding=1),
				# 1x32x40x40
				nn.ReLU(inplace=True),
				nn.Conv2d(32, 32, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2),
				# 1x32x20x20

				nn.Conv2d(32, 64, kernel_size=3, padding=1),
				# 1x64x20x20
				nn.ReLU(inplace=True),
				nn.Conv2d(64, 64, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2),
				# 1x64x10x10

				nn.Conv2d(64, 128, kernel_size=3, padding=1),
				# 1x128x10x10
				nn.ReLU(inplace=True),
				nn.Conv2d(128, 128, kernel_size=3, padding=1),
				nn.ReLU(inplace=True)
			)

		# NxCxHxW
		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),

			# NxCx1x1
			nn.Flatten(),
			# NxC
			nn.Linear(in_channels, 16),
			# Nx16
			nn.ReLU(inplace=True),
			nn.Linear(16, 4)
			# Nx4
		)

		self.skip_decoder = skip_decoder


	def forward(self, x):
		if not self.skip_decoder:
			x = self.cnn(x)

		x = self.head(x)
		return x

class FastPixelShuffle(nn.Module):
	def __init__(self, upscale_factor):
		super().__init__()

		self.upscale_factor = upscale_factor

	def forward(self, x):
		N, C, H, W = x.shape

		out_channels = C // (self.upscale_factor ** 2)

		x = x.reshape(N, out_channels, self.upscale_factor, self.upscale_factor, H, W)
		x = x.permute(0, 1, 4, 2, 5, 3)
		x = x.reshape(N, C // (self.upscale_factor ** 2), H * self.upscale_factor, W * self.upscale_factor)

		return x

def _test_FastPixelShuffle():
	x = torch.rand((1, 16, 10, 10))

	act = FastPixelShuffle(4)
	exp = nn.PixelShuffle(4)

	assert torch.allclose(act(x), exp(x)), "Bad FastPixelShuffle implementation"

_test_FastPixelShuffle()

class ConvUpsample(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding, upscale_factor):
		super().__init__()

		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=kernel_size, padding=padding),
			FastPixelShuffle(upscale_factor)
		)

	def forward(self, x):
		return self.net(x)

class Net(nn.Module):
	def __init__(self, generate_vel=True, generate_depthmap=True, train_unet=False, train_velpred=False, use_convtrans=False, skip_decoder=True):
		super().__init__()

		# FastPixelShuffle decoder has a throughput of ~28.8fps and uses 2 meta epochs for each decode layer
		# ConvTrans decoder has a throughput of ~27.1fps and uses over 2x as much npuRAM and uses 3 meta epochs for each decode layer

		# ConvTrans uses hybrid dma reads for strided mem access for depth to space and channel concat
		# FastPixelShuffle uses hybrid dma reads for strided mem reads for transpose

		# Throughputs are measured with gc collection every 5 inferences and no vision pipeline
		# All inference outputs are post processed copy by reference and without dequantization (see n6/inference.py)
		# Deep copy is ~2x slower for EC, and ~1.5x slower for hybrid
		# Throughputs measured with genereate_depthmap=True and skip_decoder=True

		# generate_depthmap_True and skip_decoder=False, FastPixelShuffle: ~27.0fps and 2 meta epochs for each decode layer
		# generate_depthmap=False and skip_decoder=True: Throughput of ~34.4fps and only uses one meta epoch (fully EC)

		# Nx1x320x320
		self.enc11 = nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=4)
		# Nx4x80x80
		self.enc12 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
		# Nx16x80x80
		self.enc13 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Nx16x40x40
		self.enc21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		# Nx32x40x40
		self.enc22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
		self.enc23 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Nx32x20x20
		self.enc31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		# Nx64x20x20
		self.enc32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.enc33 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Nx64x10x10
		self.enc41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		# Nx128x10x10
		self.enc42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

		# Nx128x10x10
		self.bottleneck = ConvLSTM(4, (128,) * 4, (3,) * 4, True, 128)

		if generate_depthmap or not skip_decoder:
			# Nx128x10x10
			if use_convtrans:
				self.dec11 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
			else:
				self.dec11 = ConvUpsample(128, 64, kernel_size=3, padding=1, upscale_factor=2)
			# Nx64x20x20
			# skip cat Nx128x20x20
			self.dec12 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
			self.dec13 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

			# Nx64x20x20
			if use_convtrans:
				self.dec21 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
			else:
				self.dec21 = ConvUpsample(64, 32, kernel_size=3, padding=1, upscale_factor=2)
			# Nx32x40x40
			# skip cat Nx64x40x40
			self.dec22 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
			self.dec23 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

			# Nx32x40x40
			if use_convtrans:
				self.dec31 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
			else:
				self.dec31 = ConvUpsample(32, 16, kernel_size=3, padding=1, upscale_factor=2)
			# Nx16x80x80
			# skip cat Nx32x80x80
			self.dec32 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
			self.dec33 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

			# Nx16x80x80
			self.dec41 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
			# Nx4x80x80
			if use_convtrans:
				self.dec42 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=4)
			else:
				self.dec42 = ConvUpsample(4, 1, kernel_size=3, padding=1, upscale_factor=4)
			# Nx1x320x320

		if generate_vel:
			# skip from latent
			# Nx128x10x10
			self.vel_head = VelPred(128, skip_decoder=skip_decoder)

		self.generate_vel = generate_vel
		self.generate_depthmap = generate_depthmap
		self.skip_decoder = skip_decoder

		if not train_unet:
			for module in (self.enc11, self.enc12, self.enc13,
								 		self.enc21, self.enc22, self.enc23,
								 		self.enc31, self.enc32, self.enc33,
								 		self.enc41, self.enc42,
								 		self.bottleneck):
				for param in module.parameters():
					param.requires_grad = False
			
			if generate_depthmap or not skip_decoder:
				for module in (self.dec11, self.dec12, self.dec13,
											self.dec21, self.dec22, self.dec23,
											self.dec31, self.dec32, self.dec33,
											self.dec41, self.dec42,
											self.bottleneck):
					for param in module.parameters():
						param.requires_grad = False

		if not train_velpred:
			if generate_vel:
				for param in self.vel_head.parameters():
					param.requires_grad = False

	def forward(self, x, *states):
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
		latent = x

		if self.generate_depthmap or not self.skip_decoder:
			x = self.dec11(x)
			x = F.relu(x, inplace=True)
			x = torch.cat([x, skip4], dim=1)
			x = self.dec12(x)
			x = F.relu(x, inplace=True)
			x = self.dec13(x)
			x = F.relu(x, inplace=True)

			x = self.dec21(x)
			x = F.relu(x, inplace=True)
			x = torch.cat([x, skip3], dim=1)
			x = self.dec22(x)
			x = F.relu(x, inplace=True)
			x = self.dec23(x)
			x = F.relu(x, inplace=True)

			x = self.dec31(x)
			x = F.relu(x, inplace=True)
			x = torch.cat([x, skip2], dim=1)
			x = self.dec32(x)
			x = F.relu(x, inplace=True)
			x = self.dec33(x)
			x = F.relu(x, inplace=True)

			x = self.dec41(x)
			x = F.relu(x, inplace=True)
			x = depth = self.dec42(x)
		
		if self.generate_vel:
			if self.skip_decoder:
				x = latent
			else:
				x = depth

			x = vel = self.vel_head(x)

		return ((vel,) if self.generate_vel else ()) + \
					 ((depth,) if self.generate_depthmap else ()) + \
					 ((*states,))

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
		return (["vel"] if self.generate_vel else []) + \
					 (["depth"] if self.generate_depthmap else []) + \
					 (["h0_next", "s0_next", "h1_next", "s1_next", "h2_next", "s2_next", "h3_next", "s3_next"])

