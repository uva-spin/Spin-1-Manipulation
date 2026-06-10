import numpy as np


def intensities_to_populations(Iplus, Iminus, coords: str = "theta"):
    """
    Recover per-bin population densities from absorption-line intensities.

    In the theta representation, each bin i carries the spin
    fraction mu(theta) = I+(theta) + I-(theta) and

        rho_0 = mu / 3
        rho_+ = rho_0 + I+(theta)
        rho_- = rho_0 - I-(theta)

    with I+(theta) = Iplus[i] and I-(theta) = Iminus[mirror(i)].
    """
    n = len(Iplus)
    Iplus = np.asarray(Iplus, dtype=float)
    Iminus = np.asarray(Iminus, dtype=float)
    tot_area = np.sum(Iplus + Iminus)

    rho_plus = np.zeros(n)
    rho_zero = np.zeros(n)
    rho_minus = np.zeros(n)

    if coords.lower() == "r":
        for i in range(n):
            mu = (Iplus[i] + Iminus[i]) / tot_area
            rho_zero[i] = mu / 3.0
            rho_plus[i] = rho_zero[i] + Iplus[i]
            rho_minus[i] = rho_zero[i] - Iminus[i]
    elif coords.lower() == "theta":
        for i in range(n):
            m = n - 1 - i
            mu = (Iplus[i] + Iminus[m]) / tot_area
            rho_zero[i] = mu / 3.0
            rho_plus[i] = rho_zero[i] + Iplus[i]
            rho_minus[i] = rho_zero[i] - Iminus[m]
    else:
        raise ValueError(f"coords must be 'theta' or 'r', got {coords!r}")

    return rho_plus, rho_zero, rho_minus


def populations_to_intensities(rho_plus, rho_zero, rho_minus, coords: str = "r"):
    """
    Convert population densities back to R- or theta-domain intensities.

    R-domain (default): I+(R_i) = rho_+(theta_i) - rho_0(theta_i),
    I-(R_m) = rho_0(theta_i) - rho_-(theta_i) with m = mirror(i).
    """
    n = len(rho_plus)
    Iplus = np.zeros(n)
    Iminus = np.zeros(n)

    if coords.lower() == "r":
        for i in range(n):
            m = n - 1 - i
            Iplus[i] = rho_plus[i] - rho_zero[i]
            Iminus[m] = rho_zero[i] - rho_minus[i]
    elif coords.lower() == "theta":
        for i in range(n):
            m = n - 1 - i
            Iplus[i] = rho_plus[i] - rho_zero[i]
            Iminus[m] = rho_zero[i] - rho_minus[i]
    else:
        raise ValueError(f"coords must be 'theta' or 'r', got {coords!r}")

    return Iplus, Iminus


def populations_R_to_Theta(rho_plus, rho_zero, rho_minus):
    """Sum R-representation populations into theta bins."""
    n = len(rho_plus)
    rho_plus_theta = np.zeros(n)
    rho_zero_theta = np.zeros(n)
    rho_minus_theta = np.zeros(n)

    for i in range(n):
        m = n - 1 - i
        rho_plus_theta[i] = rho_plus[i] + rho_plus[m]
        rho_zero_theta[i] = rho_zero[i] + rho_zero[m]
        rho_minus_theta[i] = rho_minus[i] + rho_minus[m]

    return rho_plus_theta, rho_zero_theta, rho_minus_theta


def populations_Theta_to_R(rho_plus_theta, rho_zero_theta, rho_minus_theta):
    """Split theta-bin populations into the two R bins per theta."""
    n = len(rho_plus_theta)
    rho_plus = np.zeros(n)
    rho_zero = np.zeros(n)
    rho_minus = np.zeros(n)

    for i in range(n):
        m = n - 1 - i
        rho_plus[i] = rho_plus_theta[i] - rho_plus_theta[m]
        rho_zero[i] = rho_zero_theta[i] - rho_zero_theta[m]
        rho_minus[i] = rho_minus_theta[i] - rho_minus_theta[m]

    return rho_plus, rho_zero, rho_minus


def _equalize_pair(rho_high, rho_low, source_population, rate):
    """Transfer ``rate * source_population`` from the over-populated level to the other."""
    transfer = rate * source_population
    return rho_high - transfer, rho_low + transfer, transfer


def apply_ssrf_transfer(rho_plus, rho_zero, rho_minus, burn_idx: int, xi: float, dt: float = 1.0):
    """
    Apply one ss-RF step as pure population equalization.

    At burn frequency R_burn (bin ``burn_idx``):

    * theta_2 bin (idx): equalize rho_+ and rho_0
    * theta_1 bin (mirror): equalize rho_0 and rho_-

    For positive vector polarization rho_+ > rho_0 > rho_- and the transfers
    match the paper (rho_+ -> rho_0, rho_0 -> rho_-).  For negative
    polarization the ordering reverses and population flows in the opposite
    direction (rho_0 -> rho_+, rho_- -> rho_0), so a burn always moves
    Ps toward spin-temperature equilibrium whether Ps > 0 or Ps < 0.

    Transfer magnitude is xi * rho_source * dt, where rho_source is the
    population of the level being depleted at that transition.
    """
    n = len(rho_plus)
    burn_idx = int(burn_idx)
    mirror_idx = n - 1 - burn_idx
    rate = xi * dt

    rp_b = rho_plus[burn_idx]
    rz_b = rho_zero[burn_idx]
    if rp_b >= rz_b:
        rho_plus[burn_idx], rho_zero[burn_idx], _ = _equalize_pair(
            rp_b, rz_b, rp_b, rate
        )
    else:
        rho_zero[burn_idx], rho_plus[burn_idx], _ = _equalize_pair(
            rz_b, rp_b, rz_b, rate
        )

    rz_m = rho_zero[mirror_idx]
    rm_m = rho_minus[mirror_idx]
    if rz_m >= rm_m:
        rho_zero[mirror_idx], rho_minus[mirror_idx], _ = _equalize_pair(
            rz_m, rm_m, rz_m, rate
        )
    else:
        rho_minus[mirror_idx], rho_zero[mirror_idx], _ = _equalize_pair(
            rm_m, rz_m, rm_m, rate
        )

    return rho_plus, rho_zero, rho_minus


def _ps_crosses_zero(ps_before: float, ps_after: float) -> bool:
    """True when Ps moves to the opposite side of zero (or hits zero from one side)."""
    if ps_before > 0:
        return ps_after <= 0
    if ps_before < 0:
        return ps_after >= 0
    return ps_after != 0.0


def burn_preserves_ps_sign(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_new: np.ndarray,
    iminus_new: np.ndarray,
    burn_idx: int,
) -> bool:
    """
    Return True when Ps = I+ + I- at the burn and mirror bins stays on its
    original side of zero (matching ssRFMapper's no-zero-crossing constraint).
    """
    n = len(iplus)
    burn_idx = int(burn_idx)
    mirror_idx = n - 1 - burn_idx
    for idx in (burn_idx, mirror_idx):
        ps_before = float(iplus[idx] + iminus[idx])
        ps_after = float(iplus_new[idx] + iminus_new[idx])
        if _ps_crosses_zero(ps_before, ps_after):
            return False
    return True


def solve_rate_equations(Iplus, Iminus, dt: float, xi: float, burn_idx: int):
    """
    One ss-RF integration step: intensities -> populations -> transfer -> intensities.
    """
    rho_plus, rho_zero, rho_minus = intensities_to_populations(Iplus, Iminus, "theta")
    rho_plus, rho_zero, rho_minus = apply_ssrf_transfer(
        rho_plus, rho_zero, rho_minus, burn_idx, xi, dt
    )
    Iplus_new, Iminus_new = populations_to_intensities(rho_plus, rho_zero, rho_minus, "r")
    return Iplus_new, Iminus_new, rho_plus, rho_zero, rho_minus


def verify_rates_response(Iplus, Iminus, burn_idx: int, xi: float, dt: float = 1.0, rtol: float = 1e-6):
    """
    Expected (magnitudes of changes):
        Amp_burn  = 2 * Amp_mirror
        dIplus_burn  = 2 * dIminus_mirror
        dIminus_burn = 2 * dIplus_mirror

    Also checks that |Ps| decreases at the burn bin (saturation toward TE)
    for both positive and negative vector polarization.
    """
    burn_idx = int(burn_idx)
    mirror_idx = len(Iplus) - 1 - burn_idx

    Iplus_new, Iminus_new, _, _, _ = solve_rate_equations(
        Iplus, Iminus, dt, xi, burn_idx
    )

    d_ip_burn = Iplus_new[burn_idx] - Iplus[burn_idx]
    d_im_burn = Iminus_new[burn_idx] - Iminus[burn_idx]
    d_ip_mirror = Iplus_new[mirror_idx] - Iplus[mirror_idx]
    d_im_mirror = Iminus_new[mirror_idx] - Iminus[mirror_idx]

    amp_burn = (Iplus_new[burn_idx] + Iminus_new[burn_idx]) - (Iplus[burn_idx] + Iminus[burn_idx])
    amp_mirror = (Iplus_new[mirror_idx] + Iminus_new[mirror_idx]) - (
        Iplus[mirror_idx] + Iminus[mirror_idx]
    )

    checks = {
        "amp_burn_over_amp_mirror": abs(amp_burn) / abs(amp_mirror),
        "iplus_burn_over_iminus_mirror": abs(d_ip_burn) / abs(d_im_mirror),
        "iminus_burn_over_iplus_mirror": abs(d_im_burn) / abs(d_ip_mirror),
    }

    expected = 2.0
    ps_burn_before = Iplus[burn_idx] + Iminus[burn_idx]
    ps_burn_after = Iplus_new[burn_idx] + Iminus_new[burn_idx]
    magnitude_decreased = abs(ps_burn_after) < abs(ps_burn_before)

    passed = magnitude_decreased and all(
        abs(ratio - expected) / expected < rtol for ratio in checks.values()
    )

    return {
        "passed": passed,
        "burn_idx": burn_idx,
        "mirror_idx": mirror_idx,
        "ps_burn_before": ps_burn_before,
        "ps_burn_after": ps_burn_after,
        "magnitude_decreased": magnitude_decreased,
        "amp_burn": amp_burn,
        "amp_mirror": amp_mirror,
        "d_iplus_burn": d_ip_burn,
        "d_iminus_burn": d_im_burn,
        "d_iplus_mirror": d_ip_mirror,
        "d_iminus_mirror": d_im_mirror,
        "ratios": checks,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from lineshape.Lineshape import GenerateVectorLineshape

    R = np.linspace(-3, 3, 500)
    P = 0.45
    P_REF = 0.25
    signal, Iplus, Iminus = GenerateVectorLineshape(P, R)
    ref_signal, ref_Iplus, ref_Iminus = GenerateVectorLineshape(P_REF, R)

    f_idx = -0.9
    burn_idx = np.argmin(np.abs(R - f_idx))
    xi = 0.1
    dt = 1.0

    iplus_new, iminus_new, rho_plus, rho_zero, rho_minus = solve_rate_equations(Iplus, Iminus, dt, xi, burn_idx)
    
    rates_check = verify_rates_response(Iplus, Iminus, burn_idx, xi, dt)

    plt.figure(figsize=(12, 8))
    plt.plot(R, iplus_new + iminus_new, label=r"$I_+ + I_-$")
    plt.plot(R, iplus_new, label=r"$I_+$")
    plt.plot(R, iminus_new, label=r"$I_-$")
    # plt.plot(R, ref_signal, label=f"ref_signal ({P_REF*100:.2f}%)", linestyle=":")
    # plt.plot(R, ref_Iplus, label=f"ref_Iplus ({P_REF*100:.2f}%)", linestyle=":")
    # plt.plot(R, ref_Iminus, label=f"ref_Iminus ({P_REF*100:.2f}%)", linestyle=":")
    plt.legend()
    plt.show()
    plt.close()
