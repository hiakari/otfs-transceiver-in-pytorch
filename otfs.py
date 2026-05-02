import numpy as np
import torch
import sionna.phy

from sionna.phy import Block
from sionna.phy.channel import (
    ApplyTimeChannel,
    cir_to_time_channel,
    time_lag_discrete_time_channel,
)
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.mapping import BinarySource, Demapper, Mapper
from sionna.phy.utils import ebnodb2no

from OTFS.ddgrid import DDGrid
from OTFS.equalizer import lmmse_equalizer_cg
from OTFS.transceiver import OTFSDemodulator, OTFSModulator
from OTFS.utils import generate_hdd_from_ht, release


sionna.phy.config.seed = 42
sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")


class OTFSModel(Block):
    def __init__(
        self,
        subcarrier_spacing,
        cp_length,
        carrier_frequency,
        model,
        delay_spread,
        speed,
        num_bits_per_symbol,
        coderate,
        delay_bins,
        doppler_bins,
        cg_max_iter=30,
        cg_tol=1e-5,
        num_hutchinson_probes=2,
    ):
        super().__init__()
        self.subcarrier_spacing = float(subcarrier_spacing)
        self.cp_length = int(cp_length)
        self.carrier_frequency = float(carrier_frequency)
        self.model = model
        self.delay_spread = delay_spread
        self.speed = float(speed)
        self.num_bits_per_symbol = int(num_bits_per_symbol)
        self.coderate = float(coderate)
        self.delay_bins = int(delay_bins)
        self.doppler_bins = int(doppler_bins)
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tol = float(cg_tol)
        self.num_hutchinson_probes = int(num_hutchinson_probes)

        self.bandwidth = self.delay_bins * self.subcarrier_spacing
        self.frame_size = self.delay_bins * self.doppler_bins
        self.n = self.frame_size * self.num_bits_per_symbol

        self.l_min, self.l_max = time_lag_discrete_time_channel(
            bandwidth=self.bandwidth
        )
        self.l_tot = self.l_max - self.l_min + 1

        self.bit_source = BinarySource()
        self.mapper = Mapper(
            constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol
        )
        self.demapper = Demapper(
            demapping_method="app",
            constellation_type="qam",
            num_bits_per_symbol=self.num_bits_per_symbol,
            hard_out=True
        )

        self.dd_grid = DDGrid(
            delay_bins=self.delay_bins, doppler_bins=self.doppler_bins
        )
        self.modulator = OTFSModulator(cyclic_prefix_length=self.cp_length)
        self.demodulator = OTFSDemodulator(
            self.delay_bins, self.doppler_bins, self.cp_length
        )
        self.tdl = TDL(
            model=self.model,
            delay_spread=self.delay_spread,
            min_speed=self.speed,
            carrier_frequency=self.carrier_frequency
        )
        self.apply_channel = ApplyTimeChannel(
            num_time_samples=self.frame_size + self.cp_length,
            l_tot=self.l_tot,
        )

    def call(self, batch_size, ebno_db):
        bits = self.bit_source([batch_size, self.n])
        symbols = self.mapper(bits)
        grid = self.dd_grid.collect(symbols)
        tx_signal = self.modulator(grid)

        a, tau = self.tdl(
            batch_size=batch_size,
            num_time_steps=tx_signal.shape[-1] + self.l_tot - 1,
            sampling_frequency=self.bandwidth,
        )
        h_t = cir_to_time_channel(
            bandwidth=self.bandwidth,
            a=a,
            tau=tau,
            l_min=self.l_min,
            l_max=self.l_max,
            normalize=True,
        )

        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=self.coderate,
        )
        rx_signal = self.apply_channel(tx_signal.unsqueeze(1).unsqueeze(1), h_t, no)
        rx_signal = rx_signal.squeeze()
        rx_grid = self.demodulator(rx_signal)
        rx_symbols = release(rx_grid)

        h_t = h_t.squeeze()
        h_dd = generate_hdd_from_ht(
            h_t,
            M=self.delay_bins,
            N=self.doppler_bins,
            cp_len=self.cp_length,
        )
        x_hat, no_eff = lmmse_equalizer_cg(
            rx_symbols,
            h_dd,
            no,
            max_iter=self.cg_max_iter,
            tol=self.cg_tol,
            num_probes=self.num_hutchinson_probes,
        )

        bit_estimates = self.demapper(x_hat, no_eff)
        return bits, bit_estimates


SIM_PARAMS = {
    "ebno_db": list(np.arange(0, 30, 2.0)),
    "tdl_model": "C",
    "delay_spread": 300e-9,
    "subcarrier_spacing": 15e3,
    "carrier_frequency": 4e9,
    "num_bits_per_symbol": 2,
    "coderate": 1.0,
    "speeds_ms": [0.0, 100.0],
    "cp_length": 18,
    "delay_bins": 128,
    "doppler_bins": 14,
}


if __name__ == "__main__":
    model = OTFSModel(
        subcarrier_spacing=SIM_PARAMS["subcarrier_spacing"],
        cp_length=SIM_PARAMS["cp_length"],
        carrier_frequency=SIM_PARAMS["carrier_frequency"],
        model=SIM_PARAMS["tdl_model"],
        delay_spread=SIM_PARAMS["delay_spread"],
        speed=SIM_PARAMS["speeds_ms"][0],
        num_bits_per_symbol=SIM_PARAMS["num_bits_per_symbol"],
        coderate=SIM_PARAMS["coderate"],
        delay_bins=SIM_PARAMS["delay_bins"],
        doppler_bins=SIM_PARAMS["doppler_bins"],
    )
    b, b_hat = model(batch_size=2, ebno_db=torch.tensor(10.0))
    print("bits:", b.shape)
    print("estimated bits:", b_hat.shape)
