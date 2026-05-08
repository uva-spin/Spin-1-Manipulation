from lineshape.Lineshape import GenerateVectorLineshape
from afp import AFP
import numpy as np
import matplotlib.pyplot as plt

R = np.linspace(-6, 6, 500)
signal, Iplus, Iminus, CC = GenerateVectorLineshape(0.45, R)

afp = AFP(0.0, 0.0, 0.0)
n_plus = afp.calculate_n_plus(Iplus, Iminus)
n_minus = afp.calculate_n_minus(Iplus, Iminus)
n_naught = afp.calculate_n_naught(Iplus, Iminus)

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
    # if np.isclose(c_value, 0.0):
    #     return 0.0, 0.0, 0.0

    delta_plus = i_plus_total / c_value
    delta_minus = i_minus_total / c_value

    n_naught = (n_total + delta_plus - delta_minus)
    n_plus = n_naught + delta_plus
    n_minus = n_naught - delta_minus

    pop_sum = n_plus + n_minus + n_naught
    # if np.isclose(pop_sum, 0.0):
    #     return 0.0, 0.0, 0.0

    # Enforce normalized populations: n+ + n0 + n- = 1.
    n_plus /= pop_sum
    n_minus /= pop_sum
    n_naught /= pop_sum

    return n_plus, n_minus, n_naught


def fractional_cc_per_bin(i_plus, i_minus, c_global):
    """
    Fractional calibration constants per bin.
    """
    total_signal = np.sum(i_plus + i_minus)
    frac_plus = i_plus / total_signal
    frac_minus = i_minus / total_signal

    cc_plus_bin = c_global * frac_plus
    cc_minus_bin = c_global * frac_minus
    cc_total_bin = c_global
    return cc_plus_bin, cc_minus_bin, cc_total_bin


def fractional_cc_per_theta_bin(i_plus, i_minus, c_global):
    """
    Fractional calibration constants in theta-space mirrored pairs.
    """
    n_bins = len(i_plus)
    cc_plus_bin, cc_minus_bin, _ = fractional_cc_per_bin(i_plus, i_minus, c_global)
    cc_theta_bins = np.zeros(n_bins)
    mirrored_indices = np.zeros(n_bins, dtype=int)

    for idx in range(n_bins):
        mirror_idx = n_bins - idx - 1
        mirrored_indices[idx] = mirror_idx

        cc_theta_bins[idx] = (
            cc_plus_bin[idx]
            + cc_minus_bin[mirror_idx]
            + cc_minus_bin[idx]
            + cc_plus_bin[mirror_idx]
        )

    return cc_theta_bins, mirrored_indices


def solve_populations_per_theta_bins(i_plus, i_minus, c_global, n_total):
    """
    Apply the theta-bin procedure for all bins using mirrored pairing:
    Iplus[i] with Iminus[(N - i) mod N].
    """
    n_bins = len(i_plus)
    n_plus_bins = np.zeros(n_bins)
    n_minus_bins = np.zeros(n_bins)
    n_naught_bins = np.zeros(n_bins)
    cc_theta_bins, mirrored_indices = fractional_cc_per_theta_bin(i_plus, i_minus, c_global)

    for idx in range(n_bins):
        mirror_idx = mirrored_indices[idx]

        i_plus_theta = i_plus[idx] + i_minus[mirror_idx]
        i_minus_theta = i_minus[idx] + i_plus[mirror_idx]

        c_theta = cc_theta_bins[idx]
        n_total_theta = n_total * (c_theta / c_global)

        # if np.isclose(c_theta, 0.0):
        #     n_plus_bins[idx] = 0.0
        #     n_minus_bins[idx] = 0.0
        #     n_naught_bins[idx] = 0.0
        #     continue
 
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
    n_total = 1.0


    _, i_plus, i_minus, c_global = GenerateVectorLineshape(reference_p, x)

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
    ax.plot(theta_bins, n_plus_bins - n_naught_bins, label="$I_+$", color="red")
    # ax.plot(theta_bins, n_naught_bins - n_minus_bins, label="$n_0$", color="green")
    ax.plot(theta_bins, n_naught_bins - n_minus_bins, label="$I_-$", color="blue")
    ax.set_xlabel("Theta bin index")
    ax.set_ylabel("Population")
    ax.set_title("Population levels per theta bin")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(theta_bins, n_plus_bins, label="$n_+$", color="red")
    ax.plot(theta_bins, n_minus_bins, label="$n_-$", color="blue")
    ax.plot(theta_bins, n_naught_bins, label="$n_0$", color="green")
    ax.set_xlabel("Theta bin index")
    ax.set_ylabel("Population")
    ax.set_title("Population levels per theta bin")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
