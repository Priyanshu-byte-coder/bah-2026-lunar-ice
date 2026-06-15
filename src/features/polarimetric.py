"""
Polarimetric Feature Extraction from Chandrayaan-2 DFSAR Data.

Extracts radar polarimetric decomposition features used as input
to LunarIceNet for subsurface ice detection.

Features:
    - CPR (Circular Polarization Ratio)
    - DOP (Degree of Polarization)
    - m-chi decomposition
    - Eigenvalue decomposition (H/A/alpha)
    - Shannon Entropy
    - Pedestal Height
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_covariance_matrix(
    hh: np.ndarray, hv: np.ndarray, vh: np.ndarray, vv: np.ndarray
) -> np.ndarray:
    """
    Compute 4x4 covariance matrix [C4] from full-pol SAR data.

    Args:
        hh, hv, vh, vv: Complex SAR channels (H x W)

    Returns:
        C4 matrix of shape (H, W, 4, 4) complex
    """
    h, w = hh.shape
    scattering_vector = np.stack([hh, hv, vh, vv], axis=-1)  # (H, W, 4)

    # C4 = k * k^H (outer product)
    C4 = np.einsum('...i,...j->...ij', scattering_vector, np.conj(scattering_vector))
    return C4


def compute_coherency_matrix(
    hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
) -> np.ndarray:
    """
    Compute 3x3 coherency matrix [T3] from Pauli basis.

    Args:
        hh, hv, vv: Complex SAR channels

    Returns:
        T3 matrix of shape (H, W, 3, 3) complex
    """
    k1 = (hh + vv) / np.sqrt(2)    # Surface scattering
    k2 = (hh - vv) / np.sqrt(2)    # Double-bounce
    k3 = 2 * hv / np.sqrt(2)       # Volume scattering

    pauli = np.stack([k1, k2, k3], axis=-1)
    T3 = np.einsum('...i,...j->...ij', pauli, np.conj(pauli))
    return T3


def compute_cpr(
    hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
) -> np.ndarray:
    """
    Circular Polarization Ratio (CPR).

    CPR > 1 indicates volumetric scattering — potential subsurface ice.
    Used by PRL scientists in Faustini crater study.

    CPR = |SC|^2 / |OC|^2
    where SC = same-sense circular, OC = opposite-sense circular

    For linear pol data:
        SC = (HH - VV + 2j*HV) / 2
        OC = (HH + VV) / 2
    """
    sc = (hh - vv + 2j * hv) / 2.0
    oc = (hh + vv) / 2.0

    sc_power = np.abs(sc) ** 2
    oc_power = np.abs(oc) ** 2

    # Avoid division by zero
    cpr = np.where(oc_power > 1e-10, sc_power / oc_power, 0.0)
    return cpr.astype(np.float32)


def compute_dop(
    hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
) -> np.ndarray:
    """
    Degree of Polarization (DOP).

    DOP < 0.13 with CPR > 1 indicates volumetric scattering
    associated with subsurface ice (PRL 2026 criteria).

    DOP = sqrt(1 - 4*|C|/(trace(C))^2)
    where C is the 2x2 covariance matrix
    """
    # 2x2 covariance matrix elements
    c11 = np.mean(np.abs(hh) ** 2, axis=(-1,) if hh.ndim > 2 else ())
    c22 = np.mean(np.abs(vv) ** 2, axis=(-1,) if vv.ndim > 2 else ())
    c12 = np.mean(hh * np.conj(vv), axis=(-1,) if hh.ndim > 2 else ())

    # For pixel-wise computation
    c11 = np.abs(hh) ** 2
    c22 = np.abs(vv) ** 2
    c12 = hh * np.conj(vv)

    trace = c11 + c22
    det = c11 * c22 - np.abs(c12) ** 2

    # DOP formula
    trace_sq = trace ** 2
    ratio = np.where(trace_sq > 1e-10, 4.0 * det / trace_sq, 1.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    dop = np.sqrt(1.0 - ratio)

    return dop.astype(np.float32)


def compute_mchi_decomposition(
    hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    m-chi decomposition for compact polarimetric SAR.

    Decomposes into:
        - Blue (double-bounce): Pd
        - Red (surface/odd-bounce): Ps
        - Green (volume): Pv

    Returns dict with keys: 'surface', 'double_bounce', 'volume', 'm', 'chi'
    """
    # Stokes parameters from full-pol
    s0 = np.abs(hh) ** 2 + np.abs(vv) ** 2
    s1 = np.abs(hh) ** 2 - np.abs(vv) ** 2
    s2 = 2.0 * np.real(hh * np.conj(vv))
    s3 = -2.0 * np.imag(hh * np.conj(vv))

    # Degree of polarization
    m = np.where(
        s0 > 1e-10,
        np.sqrt(s1**2 + s2**2 + s3**2) / s0,
        0.0
    )
    m = np.clip(m, 0.0, 1.0)

    # Chi angle (ellipticity)
    chi = np.where(
        m * s0 > 1e-10,
        0.5 * np.arcsin(np.clip(s3 / (m * s0 + 1e-10), -1, 1)),
        0.0
    )

    # Decomposition powers
    total_power = s0
    pv = total_power * m * (1 - np.sin(2 * chi)) / 2.0   # Volume
    pd = total_power * m * (1 + np.sin(2 * chi)) / 2.0   # Double-bounce
    ps = total_power * (1 - m)                              # Surface

    return {
        'surface': ps.astype(np.float32),
        'double_bounce': pd.astype(np.float32),
        'volume': pv.astype(np.float32),
        'm': m.astype(np.float32),
        'chi': chi.astype(np.float32),
    }


def compute_eigenvalue_decomposition(
    T3: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Cloude-Pottier eigenvalue decomposition of coherency matrix.

    Returns:
        H (Entropy): Randomness of scattering (0=single mechanism, 1=random)
        A (Anisotropy): Relative importance of 2nd and 3rd eigenvalues
        alpha: Mean scattering angle
    """
    h, w = T3.shape[:2]

    # Eigendecomposition per pixel
    eigenvalues = np.zeros((h, w, 3), dtype=np.float64)
    eigenvectors = np.zeros((h, w, 3, 3), dtype=np.complex128)

    for i in range(h):
        for j in range(w):
            vals, vecs = np.linalg.eigh(T3[i, j])
            # Sort descending
            idx = np.argsort(vals)[::-1]
            eigenvalues[i, j] = vals[idx]
            eigenvectors[i, j] = vecs[:, idx]

    # Ensure non-negative
    eigenvalues = np.maximum(eigenvalues, 0)

    # Probabilities
    total = np.sum(eigenvalues, axis=-1, keepdims=True)
    p = np.where(total > 1e-10, eigenvalues / total, 1.0 / 3.0)

    # Entropy H
    p_safe = np.clip(p, 1e-10, 1.0)
    log_p = np.where(p > 1e-10, np.log2(p_safe), 0.0)
    H = -np.sum(p * log_p, axis=-1) / np.log2(3)
    H = np.clip(H, 0.0, 1.0)

    # Anisotropy A
    denom = eigenvalues[..., 1] + eigenvalues[..., 2]
    safe_denom = np.where(denom > 1e-10, denom, 1.0)
    A = np.where(
        denom > 1e-10,
        (eigenvalues[..., 1] - eigenvalues[..., 2]) / safe_denom,
        0.0
    )
    A = np.clip(A, 0.0, 1.0)

    # Mean alpha angle
    alpha_i = np.arccos(np.clip(np.abs(eigenvectors[..., 0, :]), 0, 1))
    alpha = np.sum(p * alpha_i, axis=-1)

    return {
        'entropy': H.astype(np.float32),
        'anisotropy': A.astype(np.float32),
        'alpha': alpha.astype(np.float32),
        'eigenvalues': eigenvalues.astype(np.float32),
    }


def compute_shannon_entropy(T3: np.ndarray) -> np.ndarray:
    """
    Shannon Entropy — information content of polarimetric SAR.

    SE = SE_intensity + SE_polarization + SE_phase
    """
    h, w = T3.shape[:2]
    n = 3  # matrix dimension

    se = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            T = T3[i, j]
            trace = np.real(np.trace(T))
            if trace < 1e-10:
                continue

            # Intensity component
            se_i = n * np.log(np.pi * np.e * trace / n)

            # Polarimetric component
            det = np.real(np.linalg.det(T))
            if det > 0:
                se_p = np.log(det / (trace / n) ** n)
            else:
                se_p = 0.0

            se[i, j] = se_i + se_p

    return se


def compute_pedestal_height(
    hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
) -> np.ndarray:
    """
    Pedestal Height — minimum of co-pol signature.

    Low pedestal height = single scattering (surface).
    High pedestal height = multiple scattering (volume/subsurface).

    Important indicator for subsurface ice (volumetric scattering).
    """
    co_pol_power = np.abs(hh) ** 2 + np.abs(vv) ** 2
    cross_pol_power = np.abs(hv) ** 2

    total_power = co_pol_power + 2 * cross_pol_power

    pedestal = np.where(
        total_power > 1e-10,
        2 * cross_pol_power / total_power,
        0.0
    )

    return pedestal.astype(np.float32)


def extract_all_features(
    hh: np.ndarray,
    hv: np.ndarray,
    vh: np.ndarray,
    vv: np.ndarray,
    spatial_filter_size: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Extract all polarimetric features from full-pol DFSAR data.

    Args:
        hh, hv, vh, vv: Complex SAR channels (H x W)
        spatial_filter_size: Boxcar filter size for speckle reduction

    Returns:
        Dictionary of feature maps, each (H x W)
    """
    # Optional spatial averaging (speckle filter)
    if spatial_filter_size > 1:
        from scipy.ndimage import uniform_filter
        k = spatial_filter_size
        hh = uniform_filter(hh.real, k) + 1j * uniform_filter(hh.imag, k)
        hv = uniform_filter(hv.real, k) + 1j * uniform_filter(hv.imag, k)
        vh = uniform_filter(vh.real, k) + 1j * uniform_filter(vh.imag, k)
        vv = uniform_filter(vv.real, k) + 1j * uniform_filter(vv.imag, k)

    features = {}

    # Core ice indicators
    features['cpr'] = compute_cpr(hh, hv, vv)
    features['dop'] = compute_dop(hh, hv, vv)

    # m-chi decomposition
    mchi = compute_mchi_decomposition(hh, hv, vv)
    features.update({f'mchi_{k}': v for k, v in mchi.items()})

    # Pedestal height
    features['pedestal_height'] = compute_pedestal_height(hh, hv, vv)

    # Coherency matrix and eigenvalue decomposition
    T3 = compute_coherency_matrix(hh, hv, vv)
    eigen = compute_eigenvalue_decomposition(T3)
    features['entropy'] = eigen['entropy']
    features['anisotropy'] = eigen['anisotropy']
    features['alpha'] = eigen['alpha']

    # Shannon entropy
    features['shannon_entropy'] = compute_shannon_entropy(T3)

    # Backscatter intensities
    features['hh_intensity'] = (np.abs(hh) ** 2).astype(np.float32)
    features['hv_intensity'] = (np.abs(hv) ** 2).astype(np.float32)
    features['vv_intensity'] = (np.abs(vv) ** 2).astype(np.float32)

    # Ratios
    features['hh_vv_ratio'] = np.where(
        features['vv_intensity'] > 1e-10,
        features['hh_intensity'] / features['vv_intensity'],
        1.0
    ).astype(np.float32)

    return features


def create_feature_stack(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Stack all feature maps into single tensor for model input.

    Returns:
        Numpy array of shape (C, H, W) where C = number of features
    """
    feature_names = sorted(features.keys())
    stack = np.stack([features[name] for name in feature_names], axis=0)
    return stack, feature_names
