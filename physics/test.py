from lineshape.Lineshape import GenerateVectorLineshape
import numpy as np
import matplotlib.pyplot as plt


def calibrate_constant(reference_polarization, i_plus_ref, i_minus_ref):
    """Map integrated area to polarization."""
    total_signal = np.sum(i_plus_ref + i_minus_ref)
    return reference_polarization / total_signal


def solve_populations_from_totals(i_plus_total, i_minus_total, c_value, n_total):
    """
    Solve using:
    Iplus  = C (n+ - n0)
    Iminus = C (n0 - n-)
    n+ + n0 + n- = N_total
    """
    delta_plus = i_plus_total / c_value
    delta_minus = i_minus_total / c_value

    n_naught = (n_total - delta_plus + delta_minus)
    n_plus = n_naught + delta_plus
    n_minus = n_naught - delta_minus
    return n_plus, n_minus, n_naught


def fractional_cc_per_bin(i_plus, i_minus, c_global):
    """
    Fractional calibration constants per bin.
    """
    total_signal = np.sum(i_plus + i_minus)
    frac_plus = i_plus / total_signal
    frac_minus = i_minus / total_signal
    frac_total = (i_plus + i_minus) / total_signal

    cc_plus_bin = c_global * frac_plus
    cc_minus_bin = c_global * frac_minus
    cc_total_bin = c_global * frac_total
    return cc_plus_bin, cc_minus_bin, cc_total_bin


def solve_populations_per_theta_bins(i_plus, i_minus, c_global, n_total):
    """
    Apply the theta-bin procedure for all bins using mirrored pairing:
    Iplus[i] with Iminus[(N - i) mod N].
    """
    n_bins = len(i_plus)
    total_signal = np.sum(i_plus + i_minus)

    n_plus_bins = np.zeros(n_bins)
    n_minus_bins = np.zeros(n_bins)
    n_naught_bins = np.zeros(n_bins)
    cc_theta_bins = np.zeros(n_bins)
    mirrored_indices = np.zeros(n_bins, dtype=int)

    for idx in range(n_bins):
        mirror_idx = (n_bins - idx - 1)
        mirrored_indices[idx] = mirror_idx

        i_plus_theta = i_plus[idx]
        i_minus_theta = i_minus[mirror_idx]
        theta_pair_area = i_plus_theta + i_minus_theta
        area_fraction = theta_pair_area / total_signal

        c_theta = c_global * area_fraction
        cc_theta_bins[idx] = c_theta
        n_total_theta = n_total * area_fraction

        if np.isclose(c_theta, 0.0):
            n_plus_bins[idx] = 0.0
            n_minus_bins[idx] = 0.0
            n_naught_bins[idx] = 0.0
            continue
 
        n_plus_i, n_minus_i, n_naught_i = solve_populations_from_totals(
            i_plus_total=i_plus_theta,
            i_minus_total=i_minus_theta,
            c_value=c_theta,
            n_total=n_total_theta,
        )
        n_plus_bins[idx] = n_plus_i
        n_minus_bins[idx] = n_minus_i
        n_naught_bins[idx] = n_naught_i

    return n_plus_bins, n_minus_bins, n_naught_bins, cc_theta_bins, mirrored_indices


if __name__ == "__main__":
    x = np.linspace(-3, 3, 500)
    reference_p = 0.45
    n_total = 100.0


    _, i_plus, i_minus, CC = GenerateVectorLineshape(reference_p, x)
    c_global = calibrate_constant(reference_polarization=reference_p, i_plus_ref=i_plus, i_minus_ref=i_minus)

    # Use total integrated transition strengths.
    i_plus_total = np.sum(i_plus)
    i_minus_total = np.sum(i_minus)
    n_plus_total, n_minus_total, n_naught_total = solve_populations_from_totals(
        i_plus_total=i_plus_total,
        i_minus_total=i_minus_total,
        c_value=c_global,
        n_total=n_total,
    )
    cc_plus_bin, cc_minus_bin, cc_total_bin = fractional_cc_per_bin(
        i_plus=i_plus,
        i_minus=i_minus,
        c_global=c_global,
    )

    n_plus_bins, n_minus_bins, n_naught_bins, cc_theta_bins, mirrored_indices = solve_populations_per_theta_bins(
        i_plus=i_plus,
        i_minus=i_minus,
        c_global=c_global,
        n_total=n_total,
    ) 

    theta_slice_index = 280
    mirrored_index = mirrored_indices[theta_slice_index]
    i_plus_theta = i_plus[theta_slice_index]
    i_minus_theta = i_minus[mirrored_index]
    cc_theta_pair = cc_theta_bins[theta_slice_index]

    print(f"C_global: {c_global:.6e}")
    print(f"Iplus_total: {i_plus_total:.6e}")
    print(f"Iminus_total: {i_minus_total:.6e}")
    print(f"n_plus_total: {n_plus_total:.6e}")
    print(f"n_minus_total: {n_minus_total:.6e}")
    print(f"n_naught_total: {n_naught_total:.6e}")
    print(f"theta_slice_index: {theta_slice_index}")
    print(f"mirrored_index: {mirrored_index}")
    print(f"Iplus[{theta_slice_index}]: {i_plus_theta:.6e}")
    print(f"Iminus[{mirrored_index}]: {i_minus_theta:.6e}")
    print(f"CC_theta_pair: {cc_theta_pair:.6e}")
    print(f"CC_plus_bin[{theta_slice_index}]: {cc_plus_bin[theta_slice_index]:.6e}")
    print(f"CC_minus_bin[{mirrored_index}]: {cc_minus_bin[mirrored_index]:.6e}")
    print(f"n_plus_bin[{theta_slice_index}]: {n_plus_bins[theta_slice_index]:.6e}")
    print(f"n_minus_bin[{theta_slice_index}]: {n_minus_bins[theta_slice_index]:.6e}")
    print(f"n_naught_bin[{theta_slice_index}]: {n_naught_bins[theta_slice_index]:.6e}")
    print(
        "bin_sum_check: "
        f"{(n_plus_bins[theta_slice_index] + n_minus_bins[theta_slice_index] + n_naught_bins[theta_slice_index]):.6e}"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    theta_bins = np.arange(len(i_plus))
    ax.plot(theta_bins, n_plus_bins, label="$n_+$", color="red")
    ax.plot(theta_bins, n_naught_bins, label="$n_0$", color="green")
    ax.plot(theta_bins, n_minus_bins, label="$n_-$", color="blue")
    ax.set_xlabel("Theta bin index")
    ax.set_ylabel("Population")
    ax.set_title("Population levels per theta bin")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
