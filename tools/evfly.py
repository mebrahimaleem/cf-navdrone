# This code was copied from the evfly project. Their license is below:

# MIT License with Commercial Use Restriction
#
# Copyright (c) 2024 Anish Bhattacharya
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# Commercial use of this software is prohibited without the prior written permission of the copyright holder. For commercial use inquiries, please contact Anish Bhattacharya, email address anishb1010@gmail.com.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

            # by Anish Bhattacharya Feb 2024

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            # raise NotImplementedError()
            hidden_state = hidden_state
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class OrigUNet(nn.Module):
    def __init__(self, num_in_channels=2, num_out_channels=1, num_recurrent=0, enc_params=None, dec_params=None, input_shape=[1, 2, 320, 320], device=None, logger=None, velpred=0, fc_params=None, form_BEV=0, is_deployment=False, is_large=False, evs_min_cutoff=1e-3, skip_type='crop'):
        super().__init__()

        if logger is not None:
            mylogger = logger
        else:
            mylogger = print
        
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_recurrent = [num_recurrent]
        self.input_shape = input_shape
        self.input_h, self.input_w = input_shape[-2], input_shape[-1]
        self.velpred = velpred
        self.fc_params = fc_params
        self.enc_params = enc_params
        self.device = device
        self.form_BEV = form_BEV
        self.evs_min_cutoff = evs_min_cutoff
        self.skip_type = skip_type

        self.decoder_numch_scalar = 1 if self.skip_type == 'none' else 2

        if self.form_BEV == 1 or self.form_BEV == 2:
            self.num_in_channels = 1
        elif self.form_BEV != 0:
            raise ValueError(f'form_BEV should be 0/1/2, but is {self.form_BEV}')
        self.is_deployment = is_deployment

        mylogger(f'[OrigUNet] Initializing OrigUNet with num_in_channels={self.num_in_channels}, num_out_channels={self.num_out_channels}, num_recurrent={self.num_recurrent}, form_BEV={self.form_BEV}, is_deployment={self.is_deployment}, evs_min_cutoff={self.evs_min_cutoff}, skip_type={self.skip_type}')

        # current model is 5 layers deep (7.76M params)
        #Input: (N, 1, 260, 346)
        self.unet_e11 = nn.Conv2d(self.num_in_channels, 32, kernel_size=3, padding=0) # (N, 16, 258, 344)
        self.unet_e12 = nn.Conv2d(32, 32, kernel_size=3, padding=0) # (N, 32, 256, 342)
        self.unet_pool1 = nn.MaxPool2d(kernel_size=2, stride=2,) # (N, 32, 128, 171)

        self.unet_e21 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.unet_e22 = nn.Conv2d(64, 64, kernel_size=3, padding=0) # (N, 64, 124, 167)
        self.unet_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 64, 62, 83)

        self.unet_e31 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.unet_e32 = nn.Conv2d(128, 128, kernel_size=3, padding=0) # (N, 128, 58, 79)
        self.unet_pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 128, 29, 39)

        self.unet_e41 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.unet_e42 = nn.Conv2d(256, 256, kernel_size=3, padding=0) # (N, 256, 25, 35)
        self.unet_pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 256, 12, 17)

        self.unet_e51 = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        self.unet_e52 = nn.Conv2d(512, 512, kernel_size=3, padding=0) # (N, 512, 8, 13)
        self.unet_upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2,) # (N, 256, 16, 26)

        # size before first upconv
        self.middle_shape = (1, 512, 8, 13)
        
        # concatenate here with cropped tensor from unet_e42
        self.unet_d11 = nn.Conv2d(self.decoder_numch_scalar*256, 256, kernel_size=3, padding=0) # (N, 256, 14, 24)
        self.unet_d12 = nn.Conv2d(256, 256, kernel_size=3, padding=0) # (N, 256, 12, 22)
        self.unet_upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2,) # (N, 128, 16, 26)

        # concatenate here with cropped tensor from unet_e32
        self.unet_d21 = nn.Conv2d(self.decoder_numch_scalar*128, 128, kernel_size=3, padding=0) # (N, 128, 46, 68)
        self.unet_d22 = nn.Conv2d(128, 128, kernel_size=3, padding=0) # (N, 128, 44, 66)
        self.unet_upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2,) # (N, 64, 88, 132)

        # concatenate here with cropped tensor from unet_e22
        self.unet_d31 = nn.Conv2d(self.decoder_numch_scalar*64, 64, kernel_size=3, padding=0) # (N, 64, 86, 130)
        self.unet_d32 = nn.Conv2d(64, 64, kernel_size=3, padding=0) # (N, 64, 84, 128)
        self.unet_upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2,) # (N, 32, 168, 256)

        # concatenate here with cropped tensor from unet_e12
        self.unet_d41 = nn.Conv2d(self.decoder_numch_scalar*32, 32, kernel_size=3, padding=0) # (N, 32, 166, 254)
        self.unet_d42 = nn.Conv2d(32, 32, kernel_size=3, padding=0) # (N, 32, 164, 252)
        self.unet_out = nn.Conv2d(32, self.num_out_channels, kernel_size=1) # (N, 1, 164, 252)

        self.nonlin = nn.ReLU()

        # decoded size
        self.decoded_shape = (1, 1, 68, 148)

        if self.num_recurrent[0] > 0:
            mylogger(f'[OrigUNet] Using {self.num_recurrent[0]} recurrent layers')
            # NOTE that while PyTorch LSTM can take (L, H_in) unbatched input, ConvLSTM requires a batch dimension that can be first when batch_first=True
            self.lstm = ConvLSTM(input_dim=self.middle_shape[1], hidden_dim=[self.middle_shape[1]]*self.num_recurrent[0], num_layers=self.num_recurrent[0], kernel_size=(1, 1), bias=False, batch_first=True, return_all_layers=False)

        ## Velocity Predictor

        if self.velpred > 0:

            if self.velpred == 1:

                mylogger(f'[OrigUNet] self.velpred == 1; Using velocity predictor with a ConvNet encoder and FC head.')
                # :\nenc_params {enc_params}\nfc_params {fc_params}; input_shape_enc = {input_shape} ')

                self.convnet_velpred = DynamicConvNet(in_channels=1, num_layers=enc_params['num_layers'], kernel_sizes=enc_params['kernel_sizes'], kernel_strides=enc_params['kernel_strides'], out_channels=enc_params['out_channels'], activations=enc_params['activations'], pool_type=enc_params['pool_type'], pool_kernels=enc_params['pool_kernels'], pool_strides=enc_params['pool_strides'], conv_function=enc_params['conv_function'], invert_pool_input=enc_params['invert_pool_inputs'], logger=mylogger, device=device)
                
                mylogger(f'[OrigUNet] Input size to velpred: {[1, 1, input_shape[-2], input_shape[-1]]}')
                input_shape_enc = torch.Size([1, 1, input_shape[-2], input_shape[-1]])

            elif self.velpred == 11:
                    
                mylogger(f'[OrigUNet] self.velpred == 11; Using velocity predictor with a ConvNet encoder and FC head.')
                # :\nenc_params {enc_params}\nfc_params {fc_params}; input_shape_enc = {self.decoded_shape}')

                self.convnet_velpred = DynamicConvNet(in_channels=self.decoded_shape[1], num_layers=enc_params['num_layers'], kernel_sizes=enc_params['kernel_sizes'], kernel_strides=enc_params['kernel_strides'], out_channels=enc_params['out_channels'], activations=enc_params['activations'], pool_type=enc_params['pool_type'], pool_kernels=enc_params['pool_kernels'], pool_strides=enc_params['pool_strides'], conv_function=enc_params['conv_function'], invert_pool_input=enc_params['invert_pool_inputs'], logger=mylogger, device=device)
                
                mylogger(f'[OrigUNet] Input size to velpred: {[1, self.decoded_shape[1], self.decoded_shape[2], self.decoded_shape[3]]}')
                input_shape_enc = torch.Size([1, self.decoded_shape[1], self.decoded_shape[2], self.decoded_shape[3]])

            elif self.velpred == 2:

                mylogger(f'[OrigUNet] self.velpred == 2; Using velocity predictor with a ConvNet encoder and ConvNet head.')
                # :\nenc_params {enc_params}\nfc_params {fc_params}; input_shape_enc = {self.middle_shape}')

                self.convnet_velpred = DynamicConvNet(in_channels=self.middle_shape[1], num_layers=enc_params['num_layers'], kernel_sizes=enc_params['kernel_sizes'], kernel_strides=enc_params['kernel_strides'], out_channels=enc_params['out_channels'], activations=enc_params['activations'], pool_type=enc_params['pool_type'], pool_kernels=enc_params['pool_kernels'], pool_strides=enc_params['pool_strides'], conv_function=enc_params['conv_function'], invert_pool_input=enc_params['invert_pool_inputs'], logger=mylogger, device=device)
                
                mylogger(f'[OrigUNet] Input size to velpred: {[1, self.middle_shape[1], self.middle_shape[2], self.middle_shape[3]]}')
                input_shape_enc = torch.Size([1, self.middle_shape[1], self.middle_shape[2], self.middle_shape[3]])

            self.convnet_velpred_outsize = find_output_size(self.convnet_velpred, input_shape_enc)

            mylogger(f'[OrigUNet] Calculated self.convnet_velpred_outsize = {self.convnet_velpred_outsize}')

            if self.num_recurrent[1] > 0:
                self.lstm_velpred = LSTM(input_size=self.convnet_velpred_outsize[1]*self.convnet_velpred_outsize[2]*self.convnet_velpred_outsize[3], hidden_size=self.convnet_velpred_outsize[1]*self.convnet_velpred_outsize[2]*self.convnet_velpred_outsize[3], num_layers=self.num_recurrent[1], dropout=0.1)
                mylogger(f'[OrigUNet] LSTM for velocity prediction has {sum(p.numel() for p in self.lstm_velpred.parameters() if p.requires_grad):,} parameters.')

            self.velpred_head = VelPredictor(fc_params=fc_params, input_size=self.convnet_velpred_outsize[1]*self.convnet_velpred_outsize[2]*self.convnet_velpred_outsize[3], num_out=1, device=device, logger=mylogger)

            # print number of parameters for velpred parts
            mylogger(f'[OrigUNet] ConvNet for velocity prediction has {sum(p.numel() for p in self.convnet_velpred.parameters() if p.requires_grad):,} parameters.')
            mylogger(f'[OrigUNet] FCNet for velocity prediction has {sum(p.numel() for p in self.velpred_head.fcnet.parameters() if p.requires_grad):,} parameters.')

    # given evframe, form 2-channel desired input
    # first channel negative values, second channel positive values
    def form_input(self, x):
        x[x.abs()<self.evs_min_cutoff] = 0.0
        if self.form_BEV == 0:
            des_input = torch.zeros_like(x).expand(-1, 2, -1, -1)
            des_input[:, 0, :, :] = torch.where(x < 0, torch.abs(x), torch.tensor(0.0).float().to(self.device))[:, 0, :, :]
            des_input[:, 1, :, :] = torch.where(x > 0, x, torch.tensor(0.0).float().to(self.device))[:, 0, :, :]
        
        # single-channel absolute value of evframe
        elif self.form_BEV == 1:
            des_input = torch.abs(x)
        
        # single-channel binary event mask
        elif self.form_BEV == 2:
            des_input = torch.zeros_like(x)
            des_input[x!=0.0] = 1.0

        else:
            raise ValueError(f'form_BEV should be 0/1/2, but is {self.form_BEV}')
        return des_input

    def form_output(self, x):
        upsampled_tensor = F.interpolate(x, size=(self.input_h, self.input_w), mode='bilinear', align_corners=False)
        upconv_tensor = x

        # if outputting the evframe in 2 channels, form back into a single-channel evframe before outputting
        if self.num_out_channels == 2:
            upsampled_tensor = upsampled_tensor[:, 1, :, :] - upsampled_tensor[:, 0, :, :]
            upsampled_tensor.unsqueeze_(1)

            upconv_tensor = x[:, 1, :, :] - x[:, 0, :, :]
            upconv_tensor.unsqueeze_(1)

        return upsampled_tensor, upconv_tensor

    def skip(self, y, big, small):
        if self.skip_type == 'crop':
            skip_out = y[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
        elif self.skip_type == 'interp':
            skip_out = F.interpolate(y, size=(small[0], small[1]), mode='bilinear', align_corners=False)
        elif self.skip_type == 'none':
            skip_out = None
        else:
            raise ValueError(f'[LEARNER_MODELS/ORIGUNET] skip_type should be crop/interp/none, but is {self.skip_type}.')
        return skip_out

    def forward(self, x):
        im = x[0]
        if self.num_in_channels == 2 or self.form_BEV > 0:
            im = self.form_input(im)
        
        if x[2] is None:
            x[2] = (None, None)

        # st_unet = time.time()

        # encoder

        y_e1 = self.nonlin(self.unet_e12(self.nonlin(self.unet_e11(im))))
        unet_enc1 = self.unet_pool1(y_e1)
        y_e2 = self.nonlin(self.unet_e22(self.nonlin(self.unet_e21(unet_enc1))))
        unet_enc2 = self.unet_pool2(y_e2)
        y_e3 = self.nonlin(self.unet_e32(self.nonlin(self.unet_e31(unet_enc2))))
        unet_enc3 = self.unet_pool3(y_e3)
        y_e4 = self.nonlin(self.unet_e42(self.nonlin(self.unet_e41(unet_enc3))))
        unet_enc4 = self.unet_pool4(y_e4)
        y_e5 = self.nonlin(self.unet_e52(self.nonlin(self.unet_e51(unet_enc4))))

        h_unet = None
        if self.num_recurrent[0] > 0:
            y_e5_lstm, h_unet = self.lstm(y_e5.unsqueeze(0), x[2][0])
            y_e5 = y_e5_lstm[0].squeeze(0)

        y_upconv = None # torch.zeros((), device=self.device)
        y_interp = None # torch.zeros((), device=self.device)

        # decoder

        if not self.is_deployment or (self.is_deployment and (self.velpred == 1 or self.velpred == 11)):

            # big = (25, 35)
            # small = (16, 26)
            # cropped_enc = y_e4[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
            skipped_enc = self.skip(y_e4, (25, 35), (16, 26))
            concat_input = torch.cat((skipped_enc, self.unet_upconv1(y_e5)), 1) if skipped_enc is not None else self.unet_upconv1(y_e5)
            y_d1 = self.nonlin(self.unet_d12(self.nonlin(self.unet_d11( concat_input ))))

            # big = (58, 79)
            # small = (24, 44)
            # cropped_enc = y_e3[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
            skipped_enc = self.skip(y_e3, (58, 79), (24, 44))
            concat_input = torch.cat((skipped_enc, self.unet_upconv2(y_d1)), 1) if skipped_enc is not None else self.unet_upconv2(y_d1)
            y_d2 = self.nonlin(self.unet_d22(self.nonlin(self.unet_d21( concat_input ))))

            # big = (124, 167)
            # small = (40, 80)
            # cropped_enc = y_e2[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
            skipped_enc = self.skip(y_e2, (124, 167), (40, 80))
            concat_input = torch.cat((skipped_enc, self.unet_upconv3(y_d2)), 1) if skipped_enc is not None else self.unet_upconv3(y_d2)
            y_d3 = self.nonlin(self.unet_d32(self.nonlin(self.unet_d31( concat_input ))))

            # big = (256, 342)
            # small = (72, 152)
            # cropped_enc = y_e1[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
            skipped_enc = self.skip(y_e1, (256, 342), (72, 152))
            concat_input = torch.cat((skipped_enc, self.unet_upconv4(y_d3)), 1) if skipped_enc is not None else self.unet_upconv4(y_d3)
            y_d4 = self.nonlin(self.unet_d42(self.nonlin(self.unet_d41( concat_input ))))
            
            y_upconv = self.unet_out(y_d4)

            y_interp, y_upconv = self.form_output(y_upconv)

        # velocity prediction

        # make tensor [1, 0, 0] repeat to first dim length of batch
        y_vel = torch.Tensor([1., 0., 0.]) # default value is forward full speed
        y_vel = y_vel.repeat(x[0].shape[0], 1)
        
        h_velpred = None
        if self.velpred > 0:

            if self.velpred == 1:

                y_postconvnet_velpred = self.convnet_velpred(y_interp)

            elif self.velpred == 11:

                y_postconvnet_velpred = self.convnet_velpred(y_upconv)
            
            elif self.velpred == 2:
            
                y_postconvnet_velpred = self.convnet_velpred(y_e5)
            
            y_postconvnet_velpred = torch.flatten(y_postconvnet_velpred, 1)
            
            if self.num_recurrent[1] > 0:
            
                y_postconvnet_velpred, h_velpred = self.lstm_velpred(y_postconvnet_velpred, x[2][1])
            
            y_vel, _ = self.velpred_head([y_postconvnet_velpred])

        return y_vel, (y_interp, y_upconv, (h_unet, h_velpred))
    def example_inputs(self):
        return torch.empty(1, 2, 320, 320)

    def gen_calib_data(self, n=100):
        return torch.randn(n, 1, 2, 320, 320)


