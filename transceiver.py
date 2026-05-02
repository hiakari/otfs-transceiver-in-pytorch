from sionna.phy import Block
import torch

class OTFSModulator(Block):
    "Generate an Time-Domain OTFS frame of size MN"
    def __init__(self, cyclic_prefix_length, **kwargs):
        super().__init__(**kwargs)
        self.L = cyclic_prefix_length
    def call(self, grid):
        grid_TF = self.isfft(grid)
        grid_T = self.heisenberg(grid_TF)
        td_signal = self.flatten(grid_T)
        cp = td_signal[..., -self.L:]
        td_signal = torch.concatenate([cp, td_signal], dim=-1)
        return td_signal
    def isfft(self, grid):
        grid_FD = torch.fft.fft(grid, dim=-1, norm="ortho") # FFT Delay = Frequency
        grid_FT = torch.fft.ifft(grid_FD, dim=-2, norm="ortho") # IFFT Doppler = Time
        return grid_FT
    def heisenberg(self, grid, pulse_shaping_matrix = None):
        G_tx = torch.ones_like(grid[-1]) if pulse_shaping_matrix is None else pulse_shaping_matrix

        grid_TD = torch.fft.ifft(grid, dim=-1, norm="ortho")
        pshaped_grid = grid_TD * G_tx
        return pshaped_grid
    def flatten(self, grid):
        td_signal = grid.flatten(start_dim=-2)
        return td_signal

class OTFSDemodulator(Block):
    "Recover DD-domain symbols from a Time-Domain OTFS signal"
    def __init__(self, M, N, L, **kwargs):
        super().__init__(**kwargs)
        self.M = M # Delay bins
        self.N = N # Doppler bins
        self.L = L # CP length

    def call(self, signal):
        # 1. CP and Trailing samples removal
        # Discard CP (first L samples) and take MN samples
        signal = signal[..., self.L : self.L + self.M * self.N]

        # 2. Reshape to Time-Time grid [..., N, M]
        grid_T = signal.reshape(*signal.shape[:-1], self.N, self.M)

        # 3. Transform back to DD domain
        grid_TF = self.wigner(grid_T)
        grid_DD = self.sfft(grid_TF)
        return grid_DD

    def wigner(self, grid):
        # Inverse Heisenberg: FFT over Frequency (Delay) dimension
        return torch.fft.fft(grid, dim=-1, norm="ortho")

    def sfft(self, grid_TF):
        # Inverse ISFFT: IFFT over Frequency, FFT over Time
        grid_FD = torch.fft.ifft(grid_TF, dim=-1, norm="ortho")
        grid_DD = torch.fft.fft(grid_FD, dim=-2, norm="ortho")
        return grid_DD
