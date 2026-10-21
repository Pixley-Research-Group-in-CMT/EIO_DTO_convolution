"""Band-resolved Sec. 4.1 response for the uniform J_zpm=J_pmpm=0 Ansatz.

The transport contraction needs Sxx+Syy=(S+-+S-+)/2 on each link.
Consequently only diagonal entries of the spectral projectors survive,
although both bands and the full matrix Hamiltonian are retained. Condensate
matrices are per momentum and divided by the number of quadrature points;
their mean trace convention is the one used by gmft_field_general.
"""
import numpy as np

from tto_structure_factor import E_MU, reduced_k_grid, spinon_potential, lorentzian_delta


class UnstableSpinonSpectrum(ValueError):
    """A converged saddle has a negative squared frequency at DSSF momenta."""


def spectrum(k, mean_field):
    k = np.asarray(k)
    v = spinon_potential(k, mean_field["j_pm_ratio"], mean_field["s_minus"])
    hopping = -np.exp(2j * np.pi * k @ E_MU.T) @ (
        mean_field["b_plus"] * np.conj(mean_field["s_minus"]))
    kernel = np.zeros(k.shape[:-1] + (2, 2), dtype=complex)
    kernel[..., 0, 0] = kernel[..., 1, 1] = v
    kernel[..., 0, 1] = hopping
    kernel[..., 1, 0] = hopping.conj()
    bands, vectors = np.linalg.eigh(kernel)
    shifted = mean_field["lambda_ratio"] + bands
    if shifted.min() < -1e-8:
        raise UnstableSpinonSpectrum(
            "General gMFT saddle is unstable on DSSF momenta: "
            f"min((lambda+v)/Jzz)={float(shifted.min()):.12g}; "
            "check the momentum grid and band minimum")
    frequencies = np.sqrt(2 * np.maximum(shifted, 0))
    residues = np.zeros_like(frequencies)
    np.divide(1, frequencies, out=residues, where=shifted > 1e-10)
    # (..., sublattice, band), including the Jzz/omega factor.
    return frequencies, np.abs(vectors) ** 2 * residues[..., None, :]


def transverse_weight(a, b):
    """Half of the sum of the +- and -+ traces, without |s_mu|^2."""
    return (a[..., 1] * b[..., 0] + a[..., 0] * b[..., 1]) / 2


def dynamic_structure_factor(q_reduced, omega_ev, mean_field, *, j_zz_ev,
                             eta_ev, nk=8, interface_sublattices=(1, 2, 3),
                             chunk_size=128):
    """Eqs. (92), (97)-(99), with all occupied soft momenta included.

    Chunk transfers so memory scales as chunk_size*nk**3, not N_EIO**2*nk**3.
    Condensate occupations come from the winning solver result, not a fresh
    choice of eigenvector or of soft momentum. The elastic term is retained
    separately to preserve the existing spinon transport convention.
    """
    q = np.asarray(q_reduced, dtype=float)
    energy = np.asarray(omega_ev, dtype=float)
    if q.shape != energy.shape + (3,):
        raise ValueError("q_reduced must have shape omega_ev.shape + (3,)")
    if chunk_size < 1 or j_zz_ev <= 0 or eta_ev <= 0:
        raise ValueError("chunk_size, Jzz and eta must be positive")
    shape = energy.shape
    q, energy = q.reshape(-1, 3), energy.ravel()
    k = reduced_k_grid(nk)
    wk, rk = spectrum(k, mean_field)
    wk *= j_zz_ev
    ck = np.asarray(mean_field["condensate_k"])
    cd = np.asarray(mean_field["condensate_diagonal"])
    base = {key: np.zeros(len(q)) for key in
            ("continuum", "one_condensate", "two_condensate")}
    for start in range(0, len(q), chunk_size):
        sl = slice(start, start + chunk_size)
        qc, ec = q[sl], energy[sl]
        wq, rq = spectrum(k[None, :, :] + qc[:, None, :], mean_field)
        wq *= j_zz_ev
        for m in range(2):
            for n in range(2):
                weight = transverse_weight(rq[..., :, m], rk[None, ..., :, n])
                base["continuum"][sl] += np.mean(weight * lorentzian_delta(
                    ec[:, None] - wq[..., m] - wk[None, :, n], eta_ev), axis=1)
        for point, diagonal in zip(ck, cd):
            for sign in (1, -1):
                ws, rs = spectrum(point + sign * qc, mean_field)
                for m in range(2):
                    base["one_condensate"][sl] += transverse_weight(
                        rs[..., :, m], diagonal) * lorentzian_delta(
                            ec - j_zz_ev * ws[..., m], eta_ev)
            for other, other_diagonal in zip(ck, cd):
                difference = point + qc - other
                match = np.linalg.norm(difference - np.round(difference), axis=1) < 1e-10
                base["two_condensate"][sl] += (len(k) * match
                    * transverse_weight(other_diagonal, diagonal)
                    * lorentzian_delta(ec, eta_ev))
    weights = np.abs(mean_field["s_minus"][list(interface_sublattices)]) ** 2
    result = {key: (weights[:, None] * value).reshape((len(weights),) + shape)
              for key, value in base.items()}
    result["total"] = sum(result.values())
    return result


def load_saddle(path):
    """Read a lossless winning saddle saved by gmft_transport_job.py."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
