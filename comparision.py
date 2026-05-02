import matplotlib.pyplot as plt
import numpy as np
import torch
import sionna.phy

from sionna.phy.utils import sim_ber

from OTFS.SISOOFDMTDL import Model as OFDMModel
from OTFS.otfs import OTFSModel


sionna.phy.config.seed = 42
sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")


COMPARISION_PARAMS = {
    "ebno_db": np.arange(0, 30, 2.0),
    "tdl_model": "C",
    "delay_spread": 300e-9,
    "subcarrier_spacing": 15e3,
    "carrier_frequency": 4e9,
    "domain": "time",
    "num_bits_per_symbol": 2,
    "coderate": 1.0,
    "speeds_ms": [0.0, 50.0],
    "cp_length": 20,
    "pilot_indices": [0],
    "otfs_delay_bins": 128,
    "otfs_doppler_bins": 14,
    "batch_size": 20,
    "max_mc_iter": 100,
    "num_target_block_errors": 1000,
    "target_ber": 1e-4,
}


def build_ofdm_model(speed):
    return OFDMModel(
        subcarrier_spacing=COMPARISION_PARAMS["subcarrier_spacing"],
        cp_length=COMPARISION_PARAMS["cp_length"],
        pilot_indices=COMPARISION_PARAMS["pilot_indices"],
        carrier_frequency=COMPARISION_PARAMS["carrier_frequency"],
        model=COMPARISION_PARAMS["tdl_model"],
        delay_spread=COMPARISION_PARAMS["delay_spread"],
        speed=speed,
        domain=COMPARISION_PARAMS["domain"],
        num_bits_per_symbol=COMPARISION_PARAMS["num_bits_per_symbol"],
        coderate=COMPARISION_PARAMS["coderate"],
    )


def build_otfs_model(speed):
    return OTFSModel(
        subcarrier_spacing=COMPARISION_PARAMS["subcarrier_spacing"],
        cp_length=COMPARISION_PARAMS["cp_length"],
        carrier_frequency=COMPARISION_PARAMS["carrier_frequency"],
        model=COMPARISION_PARAMS["tdl_model"],
        delay_spread=COMPARISION_PARAMS["delay_spread"],
        speed=speed,
        num_bits_per_symbol=COMPARISION_PARAMS["num_bits_per_symbol"],
        coderate=COMPARISION_PARAMS["coderate"],
        delay_bins=COMPARISION_PARAMS["otfs_delay_bins"],
        doppler_bins=COMPARISION_PARAMS["otfs_doppler_bins"],
    )


def run_simulation(label, model):
    print(f"Running {label}")
    ber, bler = sim_ber(
        mc_fun=model,
        ebno_dbs=COMPARISION_PARAMS["ebno_db"],
        batch_size=COMPARISION_PARAMS["batch_size"],
        max_mc_iter=COMPARISION_PARAMS["max_mc_iter"],
        num_target_block_errors=COMPARISION_PARAMS["num_target_block_errors"],
        target_ber=COMPARISION_PARAMS["target_ber"],
        soft_estimates=False,
        early_stop=True,
        forward_keyboard_interrupt=True,
        compile_mode=None,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch._dynamo.reset()
    return ber.cpu().numpy(), bler.cpu().numpy()


def main():
    results = {}

    for speed in COMPARISION_PARAMS["speeds_ms"]:
        case_name = "static" if speed == 0.0 else "mobility"
        ofdm_label = f"OFDM, {case_name}, {speed} m/s"
        otfs_label = f"OTFS, {case_name}, {speed} m/s"

        results[ofdm_label] = run_simulation(ofdm_label, build_ofdm_model(speed))
        results[otfs_label] = run_simulation(otfs_label, build_otfs_model(speed))

    plt.figure(figsize=(10, 6))
    for label, (ber, _) in results.items():
        line_style = "--" if "static" in label else "-."
        marker = "o" if label.startswith("OFDM") else "s"
        plt.semilogy(
            COMPARISION_PARAMS["ebno_db"],
            ber,
            linestyle=line_style,
            marker=marker,
            label=label,
        )
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.ylim((1e-5, 1.0))
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_ber_results.png", dpi=200)
    plt.show()

    return results


if __name__ == "__main__":
    main()
