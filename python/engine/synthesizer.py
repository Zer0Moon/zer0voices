import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

class SourceModuleHnNSF(nn.Module):
    """Harmonic + Noise source module for NSF"""
    def __init__(self, sample_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.harmonic_num = harmonic_num
        self.sample_rate = sample_rate
        self.voiced_threshold = voiced_threshold
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).float()
        return uv

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sample_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, :, 0] = rad_values[:, :, 0] + rand_ini
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx.float()
        sine_waves = torch.sin(
            torch.cumsum(rad_values - cumsum_shift, dim=1) * 2 * np.pi
        )
        sine_waves = sine_waves * self.sine_amp
        return sine_waves

    def forward(self, f0):
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.harmonic_num + 1, device=f0.device)
        f0_buf[:, :, 0] = f0[:, :, 0]
        for i in range(self.harmonic_num):
            f0_buf[:, :, i + 1] = f0_buf[:, :, 0] * (i + 2)

        uv = self._f02uv(f0)
        sine_waves = self._f02sine(f0_buf)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise

        sine_merge = self.l_tanh(self.l_linear(sine_waves))
        return sine_merge, None, None

class GeneratorNSF(nn.Module):
    def __init__(self, initial_channel, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes,
                 gin_channels, sr):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        self.m_source = SourceModuleHnNSF(sr, harmonic_num=0)

        stride_f0s = [
            np.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2**(i+1)),
                    k, u, padding=(k-u)//2
                )
            ))
            self.noise_convs.append(
                nn.Conv1d(1, upsample_initial_channel // (2**(i+1)),
                    kernel_size=stride_f0s[i] * 2 if stride_f0s[i] != 1 else 1,
                    stride=stride_f0s[i] if stride_f0s[i] != 1 else 1,
                    padding=stride_f0s[i] // 2 if stride_f0s[i] != 1 else 0
                )
            )

        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        self.upp = np.prod(upsample_rates)

    def forward(self, x, f0, g=None):
        # Generate harmonic source
        f0 = f0[:, :, None]
        f0 = F.interpolate(
            f0.transpose(1, 2),
            size=x.shape[-1] * self.upp,
            mode="linear"
        ).transpose(1, 2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

class TextEncoder(nn.Module):
    """Encodes HuBERT features"""
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths=None):
        x = self.pre(x) * (x.shape[1] ** 0.5)
        stats = self.proj(x)
        m, logs = stats.split(stats.shape[1] // 2, dim=1)
        return m, logs, x

class Synthesizer(nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels,
                 hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, spk_embed_dim, gin_channels,
                 sr, is_half=False, version="v1"):
        super().__init__()
        self.version = version
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim

        if isinstance(sr, str):
            sr = int(sr.replace("k", "000"))

        in_channels = 256 if version == "v1" else 768

        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        self.emb_f0 = nn.Embedding(256, hidden_channels)
        self.enc_p = nn.Conv1d(in_channels, hidden_channels, 1)

        self.dec = GeneratorNSF(
            hidden_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr
        )

    def infer(self, phone, phone_lengths, pitch, pitchf, sid, nosplit=False):
        g = self.emb_g(sid).unsqueeze(-1)
        phone = self.enc_p(phone.transpose(1, 2))
        pitch_emb = self.emb_f0(pitch).transpose(1, 2)
        phone = phone + pitch_emb
        o = self.dec(phone, pitchf, g=g)
        return o, None, None