import numpy as np
import tqdm

class AFP:
    """
    Spin-1 AFP workflow: load I+/I- → per-bin populations, optional sweep; use to_intensities() for I±.
    """

    __slots__ = ("n_plus", "n_naught", "n_minus")

    def __init__(self, rho_plus=None, rho_zero=None, rho_minus=None):
        if rho_plus is None:
            self.n_plus = self.n_naught = self.n_minus = None
        else:
            self.n_plus = np.asarray(rho_plus, dtype=float).copy()
            self.n_naught = np.asarray(rho_zero, dtype=float).copy()
            self.n_minus = np.asarray(rho_minus, dtype=float).copy()

    @staticmethod
    def intensities_to_populations(Iplus, Iminus):
        n = len(Iplus)
        tot_area = np.sum(Iplus + Iminus)

        rho_plus = np.zeros(n)
        rho_zero = np.zeros(n)
        rho_minus = np.zeros(n)

        for i in range(n):
            m = n - 1 - i
            mu = (Iplus[i] + Iminus[m]) / tot_area
            rho_zero[i] = mu / 3.0
            rho_plus[i] = rho_zero[i] + Iplus[i] / tot_area
            rho_minus[i] = rho_zero[i] - Iminus[m] / tot_area

        total = rho_plus.sum() + rho_zero.sum() + rho_minus.sum()
        return rho_plus / total, rho_zero / total, rho_minus / total

    @staticmethod
    def populations_to_intensities(rho_plus, rho_zero, rho_minus):
        n = len(rho_plus)
        Iplus = np.zeros(n)
        Iminus = np.zeros(n)

        for i in range(n):
            m = n - 1 - i
            Iplus[i] = rho_plus[i] - rho_zero[i] 
            Iminus[m] = rho_zero[i] - rho_minus[i] 

        return Iplus, Iminus

    @staticmethod
    def _resolve_afp_sweep(n, steps=None, subset_indices=None, bin_range=None):
        if subset_indices is not None:
            return list(subset_indices)
        if bin_range is not None:
            start, stop = bin_range
            start = int(start)
            stop = int(stop)
            if start < 0 or stop > n or start > stop:
                raise ValueError(
                    f"bin_range must satisfy 0 <= start <= stop <= n_bins ({n}); "
                    f"got ({start}, {stop})."
                )
            return list(range(start, stop))
        return list(range(steps if steps is not None else n))

    @staticmethod
    def _perform_afp_on_populations(
        rho_plus, rho_zero, rho_minus, sweep, efficiency=1.0, show_progress=True
    ):
        """AFP sweep on per-bin populations (in place)."""
        n = len(rho_plus)
        iterator = tqdm.tqdm(sweep, desc="AFP") if show_progress else sweep

        for i in iterator:
            m = n - 1 - i

            rho_plus[i], rho_zero[i] = (
                efficiency * rho_zero[i] + (1 - efficiency) * rho_plus[i],
                efficiency * rho_plus[i] + (1 - efficiency) * rho_zero[i],
            )

            if m == i:
                continue

            rho_zero[m], rho_minus[m] = (
                efficiency * rho_minus[m] + (1 - efficiency) * rho_zero[m],
                efficiency * rho_zero[m] + (1 - efficiency) * rho_minus[m],
            )

    @classmethod
    def from_intensities(cls, Iplus, Iminus):
        """Build state from absorption intensities (copies arrays)."""
        rp, rz, rm = cls.intensities_to_populations(
            np.asarray(Iplus, dtype=float), np.asarray(Iminus, dtype=float)
        )
        return cls(rp, rz, rm)

    @staticmethod
    def sweep_from_intensities(
        Iplus,
        Iminus,
        steps=None,
        subset_indices=None,
        bin_range=None,
        efficiency=1.0,
        return_intensities=False,
    ):
        """
        One-shot: intensities → AFP sweep → per-bin populations ``(n_plus, n_naught, n_minus)``.

        Set ``return_intensities=True`` to get ``(Iplus_new, Iminus_new)`` instead.
        Prefer ``AFP.from_intensities(...)`` when you need multiple steps on the same state.
        """
        runner = AFP.from_intensities(Iplus, Iminus)
        runner.perform_afp(
            steps=steps,
            subset_indices=subset_indices,
            bin_range=bin_range,
            efficiency=efficiency,
            show_progress=False,
        )
        if return_intensities:
            return runner.to_intensities()
        return runner.n_plus, runner.n_naught, runner.n_minus

    def load_intensities(self, Iplus, Iminus):
        """Replace populations from I+, I-."""
        rp, rz, rm = self.intensities_to_populations(
            np.asarray(Iplus, dtype=float), np.asarray(Iminus, dtype=float)
        )
        self.n_plus, self.n_naught, self.n_minus = rp, rz, rm
        return self.n_plus, self.n_minus, self.n_naught

    def to_intensities(self):
        """Current populations → I+, I-."""
        if self.n_plus is None:
            raise RuntimeError("No populations loaded; use from_intensities or load_intensities.")
        return self.populations_to_intensities(self.n_plus, self.n_naught, self.n_minus)

    def perform_afp(
        self,
        steps=None,
        subset_indices=None,
        bin_range=None,
        efficiency=1.0,
        show_progress=True,
    ):
        """
        AFP sweep on stored per-bin populations (in place).

        Sweep: subset_indices, or bin_range, or first `steps` bins, or full grid.

        Returns ``(n_plus, n_naught, n_minus)`` (same arrays held on the instance).
        """
        if self.n_plus is None:
            raise RuntimeError("No populations loaded; use from_intensities or load_intensities.")
        n_bins = len(self.n_plus)
        sweep = self._resolve_afp_sweep(n_bins, steps, subset_indices, bin_range)
        self._perform_afp_on_populations(
            self.n_plus,
            self.n_naught,
            self.n_minus,
            sweep,
            efficiency,
            show_progress=show_progress,
        )
        return self.n_plus, self.n_naught, self.n_minus
