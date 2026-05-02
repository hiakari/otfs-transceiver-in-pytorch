import torch

def lmmse_equalizer_cg(
    received_symbols,
    H_DD,
    no,
    max_iter=50,
    tol=1e-6,
    num_probes=4,
):
    """
    LMMSE equalizer using CG + per-symbol no_eff estimate.

    returns:
        equalized_symbols: [B, P] complex
        no_eff:            [B, P] real
    """
    B, P = received_symbols.shape

    Hh = H_DD.conj().transpose(-2, -1)
    b = torch.matmul(Hh, received_symbols.unsqueeze(-1)).squeeze(-1)
    x_hat = cg_solve_normal(H_DD, b, no, max_iter=max_iter, tol=tol)

    diag_Ainv = estimate_diag_Ainv_hutchinson(
        H_DD, no, num_probes=num_probes, max_iter=max_iter, tol=tol
    )  # [B, P]

    if not torch.is_tensor(no):
        no = torch.tensor(no, device=H_DD.device, dtype=H_DD.real.dtype)

    if no.ndim == 0:
        mse = no * diag_Ainv
    else:
        mse = no[:, None] * diag_Ainv

    # gamma = (1 - mse) / mse
    mse = mse.clamp(min=1e-9, max=1 - 1e-6)
    no_eff = mse / (1.0 - mse)

    return x_hat, no_eff


def estimate_diag_Ainv_hutchinson(H, no, num_probes=4, max_iter=50, tol=1e-5):
    """
    Estimate diag(A^{-1}) where A = H^H H + no I
    using Hutchinson probing.

    H:  [B, P, P] complex
    no: scalar or [B] real

    returns:
        diag_Ainv: [B, P] real
    """
    B, P, _ = H.shape
    device = H.device
    real_dtype = H.real.dtype

    diag_est = torch.zeros((B, P), device=device, dtype=real_dtype)

    for _ in range(num_probes):
        # Rademacher probes in real domain; cast to complex
        z = torch.randint(0, 2, (B, P), device=device, dtype=torch.int64)
        z = 2 * z - 1
        z = z.to(real_dtype).to(H.dtype)

        u = cg_solve_normal(H, z, no, max_iter=max_iter, tol=tol)  # A^{-1} z

        # diag(A^{-1}) ≈ mean(z ⊙ A^{-1}z)
        contrib = (z.conj() * u).real
        diag_est += contrib

    diag_est /= num_probes
    return diag_est.clamp_min(1e-12)


def cg_solve_normal(H, b, no, x0=None, max_iter=50, tol=1e-6):
    """
    Solve (H^H H + no I) x = b using batched complex CG.

    H:   [B, P, P] complex
    b:   [B, P] complex
    no:  scalar or [B] real
    x0:  optional initial guess [B, P] complex

    returns:
        x: [B, P] complex
    """
    B, P, _ = H.shape
    device = H.device
    dtype = H.dtype

    if x0 is None:
        x = torch.zeros((B, P), device=device, dtype=dtype)
    else:
        x = x0.clone()

    r = b - apply_normal_matrix(H, x, no)
    p = r.clone()

    # rs_old = r^H r
    rs_old = torch.sum(torch.conj(r) * r, dim=-1).real  # [B]

    eps = 1e-12

    for _ in range(max_iter):
        Ap = apply_normal_matrix(H, p, no)

        denom = torch.sum(torch.conj(p) * Ap, dim=-1).real.clamp_min(eps)  # [B]
        alpha = rs_old / denom                                              # [B]

        x = x + alpha[:, None] * p
        r = r - alpha[:, None] * Ap

        rs_new = torch.sum(torch.conj(r) * r, dim=-1).real  # [B]

        # early stopping per batch: if all converged, break
        if torch.sqrt(rs_new.max()).item() < tol:
            break

        beta = rs_new / rs_old.clamp_min(eps)
        p = r + beta[:, None] * p
        rs_old = rs_new

    return x

def apply_normal_matrix(H, v, no):
    """
    Compute A v = (H^H H + no I) v

    H:  [B, P, P] complex
    v:  [B, P] complex
    no: scalar or [B] real

    returns: [B, P] complex
    """
    Hv = torch.matmul(H, v.unsqueeze(-1)).squeeze(-1)                 # [B, P]
    HhHv = torch.matmul(H.conj().transpose(-2, -1), Hv.unsqueeze(-1)).squeeze(-1)

    if not torch.is_tensor(no):
        no = torch.tensor(no, device=H.device, dtype=H.real.dtype)

    if no.ndim == 0:
        no_term = no * v
    else:
        no_term = no[:, None] * v

    return HhHv + no_term