import torch
def release(grid):
    batch_dims = grid.shape[:-2]
    return grid.reshape(*batch_dims, -1).contiguous()

import torch


def generate_hdd_from_ht(
    h_t: torch.Tensor,
    M: int,
    N: int,
    cp_len: int,
    return_ht: bool = False,
):
    """
    Input
    ----
    h_t: [batch_size, num_time_samples, num_delay_taps]
    M: Number of Delay bins
    N: Number of Doppler bins
    cp_len: CP length
    return_ht: Return H_T if true

    Output
    ----
    H_DD: [batch_size, num_time_samples, num_time_samples]
    """
    B, T, L = h_t.shape
    NM = N * M
    tx_len = NM + cp_len


    device = h_t.device
    dtype = h_t.dtype

    # Build the CP insertion matrix C
    C = torch.zeros(tx_len, NM, dtype=dtype, device=device)

    if cp_len > 0:
        C[:cp_len, NM - cp_len:] = torch.eye(cp_len, dtype=dtype, device=device)

    C[cp_len:, :] = torch.eye(NM, dtype=dtype, device=device)

    # Demodulator keeps samples [cp_len : cp_len + NM]
    kept_start = cp_len
    kept_stop = cp_len + NM

    # Initialize matrices
    H_DD = torch.zeros(B, 1, 1, NM, NM, dtype=dtype, device=device)
    H_T_all = (
        torch.zeros(B, 1, 1, NM, NM, dtype=dtype, device=device)
        if return_ht else None
    )

    for b in range(B):
        # Build time-domain channel matrix H_full
        #    y[row] = sum_ell h[row, ell] * x[row-ell]
        H_full = torch.zeros(tx_len, tx_len, dtype=dtype, device=device)

        for ell in range(L):
            rows = torch.arange(ell, tx_len, device=device)
            cols = rows - ell
            H_full[rows, cols] = h_t[b, rows, ell]

        # Effective channel matrix after removing CP (R @ H_full @ C)
        H_T = H_full[kept_start:kept_stop, :] @ C   # [NM, NM]

        if return_ht:
            H_T_all[b, 0, 0] = H_T

        # 3) Convert time-domain operator -> DD operator.
        H = H_T.reshape(N, M, N, M)

        # Left multiply by F_N on output Doppler axis
        H = torch.fft.fft(H, dim=0, norm="ortho")

        # Right multiply by F_N^H on input Doppler axis
        H = torch.fft.ifft(H, dim=2, norm="ortho")

        # No permutation needed anymore
        H_DD[b, 0, 0] = H.reshape(NM, NM)

    if return_ht:
        return H_DD, H_T_all
    return H_DD.squeeze()







# def bad_generate_hdd_from_tdl(h_t, M, N, basis_batch=64):
#     "Build H_DD by transmitting basis vectors to probe the channel instead of building it from the physics"
#     device = h_t.device
#     dtype = h_t.dtype
#
#     B = h_t.shape[0]
#     MN = M * N
#     H_DD = torch.zeros((B, 1, 1, MN, MN), dtype=dtype, device=device)
#
#     # Canonical basis in symbol space
#     eye = torch.eye(MN, dtype=torch.complex64, device=device)
#
#     def release(grid):
#         # Flatten [..., N, M] -> [..., MN]
#         batch_dims = grid.shape[:-2]
#         return grid.transpose(-1, -2).reshape(*batch_dims, -1)
#
#     for b in range(B):
#         h_t_b = h_t[b:b+1]  # Process each batch separately (can be optimized here)
#
#         for start in range(0, MN, basis_batch): # Process basis_batch bases simultaneously
#             stop = min(start + basis_batch, MN)
#             K = stop - start
#
#             # Each row is one basis vector e_j
#             basis_symbols = eye[start:stop]  # [K, MN]
#
#             # Match existing DD shape: [..., N, M]
#             basis_grid = basis_symbols.reshape(K, M, N).transpose(-1, -2)
#             basis_grid = basis_grid.unsqueeze(1).unsqueeze(1).contiguous()  # [K,1,1,N,M]
#
#             # Pass thru the system
#             tx = otfs_modulator(basis_grid)  # [K,1,1,T]
#
#             # Repeat the same channel realization K times to process K bases simultaneously
#             h_t_rep = h_t_b.expand(K, *h_t_b.shape[1:]).contiguous()
#
#             # Noiseless channel probing
#             rx = apply_channel(tx, h_t_rep, None)
#             rx_grid = otfs_demodulator(rx)                       # [K,1,1,N,M]
#             rx_symbols = release(rx_grid).squeeze(1).squeeze(1) # [K, MN]
#
#             # Column j of H is response to e_j
#             H_DD[b, 0, 0, :, start:stop] = rx_symbols.transpose(0, 1)
#
#     return H_DD