# Config
import matplotlib.pyplot as plt
import numpy as np
import torch


import time
import sionna.phy
from sionna.phy.fec import Deinterleaver, RowColumnInterleaver

from sionna.phy.ofdm import RemoveNulledSubcarriers

sionna.phy.config.seed = 42
torch.set_float32_matmul_precision('high')
sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
from sionna.phy import Block
from sionna.phy.mimo import StreamManagement

# Components
from sionna.phy.mapping import BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator, LMMSEEqualizer
# Channel Components
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import cir_to_time_channel, time_lag_discrete_time_channel, ApplyTimeChannel, \
    time_to_ofdm_channel, ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import ebnodb2no, PlotBER

class Model(Block):
    def __init__(self,
                 subcarrier_spacing,
                 cp_length,
                 pilot_indices,
                 carrier_frequency,
                 model,
                 delay_spread,
                 speed,
                 domain,
                 num_bits_per_symbol,
                 coderate):
        super().__init__()
        # Parameters
        # Channel Coding + Modulation
        self.coderate = coderate
        self.num_bits_per_symbol = int(num_bits_per_symbol)
        self.domain = domain
        # OFDM
        self.subcarrier_spacing = subcarrier_spacing
        self.cp_length = int(cp_length)
        self.pilot_indices = pilot_indices
        # Antenna + Channel
        self.carrier_frequency = carrier_frequency
        self.model = model
        self.delay_spread = delay_spread
        self.speed = speed
        # Const

        self.num_tx = 1
        self.num_rx = 1
        self.num_streams_per_tx = 1
        self.num_tx_ant = 1
        self.num_rx_ant = 1
        # Config
        sionna.phy.config.seed = 42
        self.stream_management = StreamManagement(np.array([[1]]),
                                                  num_streams_per_tx=self.num_streams_per_tx)
        # RG allocation
        self.rg = ResourceGrid(num_ofdm_symbols=14,
                               fft_size=128,
                               subcarrier_spacing=self.subcarrier_spacing,
                               num_tx=self.num_tx,
                               num_streams_per_tx=self.num_streams_per_tx,
                               cyclic_prefix_length=self.cp_length,
                               num_guard_carriers=[0, 0],
                               dc_null=True,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=self.pilot_indices)
        # Calculate min tap and max tap delay
        self.l_min, self.l_max = time_lag_discrete_time_channel(bandwidth=self.rg.bandwidth)
        self.l_tot = self.l_max - self.l_min + 1
        # Channel Coding parameters
        self.n = int(self.rg.num_data_symbols * self.num_bits_per_symbol)
        self.k = int(self.n * self.coderate)

        # Bit generator
        self.bit_source = BinarySource()

        # LDPC Encoder and Decoder
        # self.encoder = LDPC5GEncoder(self.k, self.n)
        # self.decoder = LDPC5GDecoder(encoder=self.encoder, hard_out=True)
        # Modulation
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol)
        self.demapper = Demapper('app', constellation_type="qam", num_bits_per_symbol=self.num_bits_per_symbol)
        # Interleaver
        # self.interleaver = RowColumnInterleaver(row_depth=int(self.n // self.num_bits_per_symbol)
        # self.deinterleaver = Deinterleaver(self.interleaver)
        # RG Mapper
        self.rg_mapper = ResourceGridMapper(resource_grid=self.rg)
        # OFDM Modulator
        self.ofdm_modulator = OFDMModulator(cyclic_prefix_length=self.rg.cyclic_prefix_length)
        self.ofdm_demodulator = OFDMDemodulator(fft_size=self.rg.fft_size, l_min=self.l_min, cyclic_prefix_length=self.rg.cyclic_prefix_length)
        # Channel
        self.tdl = TDL(model=self.model,
                  delay_spread=self.delay_spread,
                  min_speed=float(self.speed),
                  max_speed=float(self.speed),
                  carrier_frequency=self.carrier_frequency,
                  num_tx_ant=self.num_tx_ant,
                  num_rx_ant=self.num_rx_ant)
        if self.domain == "time":
            self.apply_channel = ApplyTimeChannel(num_time_samples=self.rg.num_time_samples, l_tot=self.l_tot)
        if self.domain == "freq":
            self.apply_channel = ApplyOFDMChannel()
        # Equalizer
        self.equalizer = LMMSEEqualizer(resource_grid=self.rg, stream_management=self.stream_management)
        # Remove DC/ Guard from RG
        self.remover = RemoveNulledSubcarriers(resource_grid=self.rg)
        
    def call(self, batch_size, ebno_db):
        # Bits
        b = self.bit_source([batch_size, self.num_tx, self.num_streams_per_tx, self.n])

        # Coded bits
        # coded_bits = self.encoder(b)
        coded_bits = b
        # Interleaved bits
        # interleaved_bits = self.interleaver(coded_bits)
        interleaved_bits = coded_bits
        # Data symbols
        data_symbols = self.mapper(interleaved_bits)

        # RG mapper
        resource_grid = self.rg_mapper(data_symbols)

        # Channel
        if self.domain == "time":
            a, tau = self.tdl(batch_size=batch_size,
                         num_time_steps=self.rg.num_time_samples + self.l_tot - 1,
                         sampling_frequency=self.rg.bandwidth)
            # Sinc pulse shaping to convert pseudo-baseband to baseband channel impulse response,
            # i.e., interpolating values of `a` at sampling instants
            h = cir_to_time_channel(bandwidth=self.rg.bandwidth, a=a, tau=tau, l_min=self.l_min, l_max=self.l_max, normalize=True)
            h_f = time_to_ofdm_channel(h_t=h, rg=self.rg, l_min=self.l_min)
        if self.domain == "freq":
            a, tau = self.tdl(batch_size=batch_size,
                              num_time_steps=self.rg.num_ofdm_symbols, # Sample at the beginning of each OFDM symbol
                              sampling_frequency=1/self.rg.ofdm_symbol_duration)
            frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
            h_f = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
            
        precoded_rg, h_eff = resource_grid, h_f
        
        # Apply Channel
        no = ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate, resource_grid=self.rg)
        if self.domain == "time":
            td_signal = self.ofdm_modulator(precoded_rg)
            received_td_signal = self.apply_channel(td_signal, h, no)
            received_rg = self.ofdm_demodulator(received_td_signal)
        else:
            received_rg = self.apply_channel(precoded_rg, h_f, no)

        # Perfect CSI
        # If h_eff already has nulled subcarriers removed (e.g., from Precoder),
        # don't call remover again.
        if h_eff.shape[-1] == self.rg.fft_size:
            h_hat = self.remover(h_eff)
        else:
            h_hat = h_eff
        err_var = 0.0
        
        # Equalization
        estimated_symbols, no_eff = self.equalizer(received_rg, h_hat, err_var, no)

        # Demapper
        llr = self.demapper(estimated_symbols, no_eff)
        # Deinterleaver
        # llr = self.deinterleaver(llr)
        # b_hat = self.decoder(llr)
        b_hat = (llr >= 0)
        return b, b_hat


SIM_PARAMS = {
  "ebno_db" : list(np.arange(0, 30, 2.0)),
  "tdl_model" : "C",
  "delay_spread" : 300e-9,
  "subcarrier_spacing" : 15e3,
  "carrier_frequency" : 4e9,
  "domain": "time",
  "num_bits_per_symbol" : 2,
  "coderate" : 1.0,
  "speeds_ms" : [0.0, 100.0],
  "cp_length" : 6,
  "pilot_indices" : [0],
  "ber" : [],
  "bler" : [],
  "duration": None
}
BATCH_SIZE = 30


if __name__ == "__main__":
    ber_plot = PlotBER("Effects of Mobility")
    for speed in SIM_PARAMS["speeds_ms"]:
        legend = f"{speed} m/s, Perf. CSI"
        print("Running: " + legend)
        model = Model(
          subcarrier_spacing=SIM_PARAMS["subcarrier_spacing"],
          cp_length=SIM_PARAMS["cp_length"],
          pilot_indices=SIM_PARAMS["pilot_indices"],
          carrier_frequency=SIM_PARAMS["carrier_frequency"],
          model=f"{SIM_PARAMS['tdl_model']}",
          delay_spread=SIM_PARAMS["delay_spread"],
          speed=speed,
          domain=SIM_PARAMS["domain"],
          num_bits_per_symbol=SIM_PARAMS["num_bits_per_symbol"],
          coderate=SIM_PARAMS["coderate"],
        )
        ber_plot.simulate(model,
                          ebno_dbs=SIM_PARAMS["ebno_db"],
                          batch_size=BATCH_SIZE,
                          max_mc_iter=100,
                          num_target_block_errors=1000,
                          soft_estimates=False,
                          early_stop=True,
                          show_fig=False,
                          add_bler=True,
                          forward_keyboard_interrupt=True,
                          legend=legend)
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    ber_plot(show_bler=False, show_ber=True)
    plt.show()
