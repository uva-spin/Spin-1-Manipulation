import numpy as np
import pandas as pd
from typing import Tuple
from scipy.special import wofz

class ssRFMapper:
    def __init__(self, f, sigma, gamma, x0, amp):
        self.f = f
        self.sigma = sigma
        self.gamma = gamma
        self.x0 = x0
        self.amp = amp
        self.center_freq = 0  

    def compute_lookup_tables(self, df: pd.DataFrame):

        self.signal_to_iplus_lookup = {}  
        self.signal_to_iminus_lookup = {} 
        
        for freq_idx in range(len(self.f)):
            ps_values_at_freq = []
            iminus_values_at_freq = []
            iplus_values_at_freq = []
            row_indices = []
            
            for row_idx, (_, ps_array, iminus_array, iplus_array) in enumerate(zip(df['P'], df['Ps'], df['Iminus'], df['Iplus'])):
                ps_values_at_freq.append(ps_array[freq_idx])
                iminus_values_at_freq.append(iminus_array[freq_idx])
                iplus_values_at_freq.append(iplus_array[freq_idx])
                row_indices.append(row_idx)
            
            ps_values = np.array(ps_values_at_freq)
            iminus_values = np.array(iminus_values_at_freq)
            iplus_values = np.array(iplus_values_at_freq)
            
            sort_indices = np.argsort(ps_values)
            sorted_ps = ps_values
            sorted_iminus = iminus_values
            sorted_iplus = iplus_values[sort_indices]
            
            self.signal_to_iplus_lookup[freq_idx] = {
                'ps_values': sorted_ps,  
                'iplus_values': sorted_iplus,  
                'row_indices': np.array(row_indices)[sort_indices]
            }
            self.signal_to_iminus_lookup[freq_idx] = {
                'ps_values': sorted_ps,  
                'iminus_values': sorted_iminus,  
                'row_indices': np.array(row_indices)[sort_indices]
            }

    @staticmethod
    def _map_ps(ps_values: np.ndarray, target_ps: float) -> int:
        idx = int(np.searchsorted(ps_values, target_ps, side="left"))
        if idx <= 0:
            return 0
        if idx >= len(ps_values):
            return len(ps_values) - 1
        if abs(target_ps - ps_values[idx - 1]) <= abs(target_ps - ps_values[idx]):
            return idx - 1
        return idx

    def map_signal(self, signal: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        iplus_results = np.empty(len(signal), dtype=float)
        iminus_results = np.empty(len(signal), dtype=float)

        for freq_idx in indices:
            signal_value = signal[freq_idx]
            iplus_lookup = self.signal_to_iplus_lookup[freq_idx]

            ps_values = iplus_lookup['ps_values']
            iplus_values = iplus_lookup['iplus_values']
            iminus_values = self.signal_to_iminus_lookup[freq_idx]['iminus_values']

            nearest_idx = self._map_ps(ps_values, signal_value)
            iplus_results[freq_idx] = iplus_values[nearest_idx]
            iminus_results[freq_idx] = iminus_values[nearest_idx]

        return iplus_results, iminus_results

    def _voigt_profile(self, x: np.ndarray, x0: float) -> np.ndarray:
        """
        Calculate discretized Voigt profile.
        
        Args:
            x: Frequency array (bin centers)
            x0: Center frequency
            
        Returns:
            Voigt profile values at each bin
        """
        # Normalize x to center
        x_norm = (x - x0) / (self.sigma * np.sqrt(2))
        
        # Voigt function using Faddeeva function
        z = x_norm + 1j * (self.gamma / (self.sigma * np.sqrt(2)))
        voigt = np.real(wofz(z)) / (self.sigma * np.sqrt(2 * np.pi))
        
        return voigt
    
    def _gaussian_profile(self, x: np.ndarray, x0: float) -> np.ndarray:
        """
        Calculate discretized Gaussian profile.
        
        Args:
            x: Frequency array (bin centers)
            x0: Center frequency
            
        Returns:
            Gaussian profile values at each bin
        """
        # Gaussian profile
        gaussian = np.exp(-0.5 * ((x - x0) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))
        
        return gaussian
    
    def _discretize_profile(self, profile_type: str = 'voigt') -> np.ndarray:
        """
        Generate discretized burn profile across frequency bins.
        
        Args:
            profile_type: 'voigt' or 'gaussian'
            
        Returns:
            Discretized profile array of same length as self.f
        """
        if profile_type.lower() == 'voigt':
            profile = self._voigt_profile(self.f, self.x0)
        elif profile_type.lower() == 'gaussian':
            profile = self._gaussian_profile(self.f, self.x0)
        else:
            raise ValueError(f"Unknown profile_type: {profile_type}. Use 'voigt' or 'gaussian'")
        
        # Normalize so peak value is 1.0 (for proper amplitude scaling)
        max_val = np.max(profile)
        if max_val > 0:
            profile = profile / max_val
        
        # Scale by amplitude
        profile = profile * self.amp
        
        return profile

    @staticmethod
    def _clip_burn_to_preserve_sign(ps_values: np.ndarray, burn_values: np.ndarray) -> np.ndarray:
        """Clip burn amplitudes so values do not cross zero."""
        clipped = np.zeros_like(burn_values)
        for i, (p_val, burn_val) in enumerate(zip(ps_values, burn_values)):
            if p_val >= 0:
                clipped[i] = np.maximum(burn_val, -p_val) if burn_val < 0 else burn_val
            elif p_val <= 0:
                clipped[i] = np.minimum(burn_val, -p_val) if burn_val > 0 else burn_val
            else:
                clipped[i] = burn_val
        return clipped

    @staticmethod
    def _scale_burn_until_nonzero(ps_values: np.ndarray, burn_values: np.ndarray) -> np.ndarray:
        """Reduce burn amplitude until no Ps value reaches or crosses zero."""
        scaled_burn = burn_values.copy()
        for _ in range(200):
            ps_after_burn = ps_values + scaled_burn
            reaches_or_crosses_zero = (
                ((ps_values > 0) & (ps_after_burn <= 0))
                | ((ps_values < 0) & (ps_after_burn >= 0))
            )
            if not np.any(reaches_or_crosses_zero):
                break
            scaled_burn *= 0.95
        return scaled_burn

    @classmethod
    def _constrain_burn_no_zero_crossing(
        cls, ps_values: np.ndarray, burn_values: np.ndarray
    ) -> np.ndarray:
        """Scale then clip burn amplitudes so Ps does not cross or hit zero."""
        scaled = cls._scale_burn_until_nonzero(ps_values, burn_values)
        clipped = cls._clip_burn_to_preserve_sign(ps_values, scaled)
        ps_after = ps_values + clipped

        # Keep Ps strictly on its original side of zero (tiny epsilon margin).
        eps = np.finfo(float).eps
        pos_mask = ps_values > 0
        neg_mask = ps_values < 0

        if np.any(pos_mask):
            too_low = ps_after[pos_mask] <= 0
            if np.any(too_low):
                max_negative = np.minimum(clipped[pos_mask][too_low], -(ps_values[pos_mask][too_low] - eps))
                clipped[pos_mask][too_low] = max_negative

        if np.any(neg_mask):
            too_high = ps_after[neg_mask] >= 0
            if np.any(too_high):
                max_positive = np.maximum(clipped[neg_mask][too_high], -(ps_values[neg_mask][too_high] + eps))
                clipped[neg_mask][too_high] = max_positive

        return clipped

    def mirror_burn_over_center(self, burn_effect: np.ndarray, burn_indices: np.ndarray) -> np.ndarray:

        mirrored_burn = np.zeros_like(self.f)
        mirrored_indices = []

        
        for i, burn_idx in enumerate(burn_indices):

            freq = self.f[burn_idx]


            mirrored_freq = 2 * self.center_freq - freq
            
            freq_diff = np.abs(self.f - mirrored_freq)

            mirrored_idx = np.argmin(freq_diff)
            
            if freq_diff[mirrored_idx] < (self.f[1] - self.f[0]) / 2:
                mirrored_burn[mirrored_idx] = burn_effect[i]
                mirrored_indices.append(mirrored_idx)
        
        return mirrored_burn, np.array(mirrored_indices)


    def apply_bin_burn(
        self,
        ps: np.ndarray,
        iplus: np.ndarray,
        iminus: np.ndarray,
        bin_idx: int,
        amp: float,
        return_burn_info: bool = False,
    ) -> Tuple:
        """
        Apply a single-bin ss-RF burn: decrease Ps amplitude at ``bin_idx`` only
        (no Voigt/Gaussian profile). Mirrored burn at the symmetric frequency is
        applied when a matching bin exists.

        Input arrays are copied; callers keep their originals unchanged.
        """
        ps = ps.copy()
        iplus = iplus.copy()
        iminus = iminus.copy()
        initial_ps = ps.copy()
        burn_indices = np.array([int(bin_idx)], dtype=int)
        burn_val = float(amp)
        if initial_ps[burn_indices[0]] < 0:
            burn_val = abs(burn_val)
        else:
            burn_val = -abs(burn_val)

        ps_at_burn = ps[burn_indices]
        clipped_burn_values = self._constrain_burn_no_zero_crossing(
            ps_at_burn, np.array([burn_val])
        )

        original_iplus_at_burn = iplus[burn_indices].copy()
        original_iminus_at_burn = iminus[burn_indices].copy()

        ps[burn_indices] += clipped_burn_values
        # ps[burn_indices] += np.array([burn_val])

        iplus_burn_effect, iminus_burn_effect = self.map_signal(ps, burn_indices)
        # swap_mask = iplus_burn_effect[burn_indices] < iminus_burn_effect[burn_indices]
        # new_iplus = np.where(
        #     swap_mask,
        #     iminus_burn_effect[burn_indices],
        #     iplus_burn_effect[burn_indices],
        # )
        # new_iminus = np.where(
        #     swap_mask,
        #     iplus_burn_effect[burn_indices],
        #     iminus_burn_effect[burn_indices],
        # )
        if ps[burn_indices] < 0:
            swap_mask = abs(iplus_burn_effect[burn_indices]) > abs(iminus_burn_effect[burn_indices])
            new_iplus = np.where(
                swap_mask,
                iminus_burn_effect[burn_indices],
                iplus_burn_effect[burn_indices],
            )
            new_iminus = np.where(
                swap_mask,
                iplus_burn_effect[burn_indices],
                iminus_burn_effect[burn_indices],
            )
        else:
            swap_mask = abs(iplus_burn_effect[burn_indices]) < abs(iminus_burn_effect[burn_indices])
            new_iplus = np.where(
                swap_mask,
                iminus_burn_effect[burn_indices],
                iplus_burn_effect[burn_indices],
            )
            new_iminus = np.where(
                swap_mask,
                iplus_burn_effect[burn_indices],
                iminus_burn_effect[burn_indices],
            )
        # Swapped mapping can point iplus upward; mirror about pre-burn iplus to flip direction.
        if ps[burn_indices] < 0:
            upward_mask = swap_mask & (abs(new_iminus) < abs(original_iplus_at_burn))
            new_iminus = np.where(
                upward_mask,
                2 * original_iminus_at_burn - new_iminus,
                new_iminus,
            )
            iplus[burn_indices] = new_iplus
            iminus[burn_indices] = new_iminus
        else:
            upward_mask = swap_mask & (abs(new_iplus) < abs(original_iminus_at_burn))
            new_iplus = np.where(
                upward_mask,
                2 * original_iplus_at_burn - new_iplus,
                new_iplus,
            )
            iplus[burn_indices] = new_iplus
            iminus[burn_indices] = new_iminus
            
            
        actual_iplus_change = np.abs(iplus[burn_indices] - original_iplus_at_burn)
        actual_iminus_change = np.abs(iminus[burn_indices] - original_iminus_at_burn)

        mirrored_burn, mirrored_indices = self.mirror_burn_over_center(
            np.array([burn_val]), burn_indices
        )

        mirrored_burn /= 2.0
        
        iplus_change_at_mirror = None
        iminus_change_at_mirror = None

        if np.any(mirrored_indices):
            mirrored_burn[mirrored_indices] *= -1.0
            burn_at_mirrored = mirrored_burn[mirrored_indices].copy()
            ps_at_mirror = ps[mirrored_indices].copy()
            burn_at_mirrored = self._constrain_burn_no_zero_crossing(
                ps_at_mirror, burn_at_mirrored
            )

            original_iplus_at_mirror = iplus[mirrored_indices].copy()
            original_iminus_at_mirror = iminus[mirrored_indices].copy()

            ps[mirrored_indices] += burn_at_mirrored

            iplus_burn_delta = iplus[burn_indices] - original_iplus_at_burn
            iminus_burn_delta = iminus[burn_indices] - original_iminus_at_burn

            iminus[mirrored_indices] -= 0.5 * iplus_burn_delta
            iplus[mirrored_indices] -= 0.5 * iminus_burn_delta

            iplus_change_at_mirror = np.abs(iplus[mirrored_indices] - original_iplus_at_mirror)
            iminus_change_at_mirror = np.abs(iminus[mirrored_indices] - original_iminus_at_mirror)

        if return_burn_info:
            burn_info = {
                "burn_indices": burn_indices,
                "mirror_indices": mirrored_indices if np.any(mirrored_indices) else None,
                "clipped_burn_values": clipped_burn_values.copy(),
                "clipped_mirror_burn_values": burn_at_mirrored.copy() if np.any(mirrored_indices) else None,
                "iplus_change_at_burn": actual_iplus_change,
                "iminus_change_at_burn": actual_iminus_change,
                "iplus_change_at_mirror": iplus_change_at_mirror,
                "iminus_change_at_mirror": iminus_change_at_mirror,
            }
            return ps, iplus, iminus, burn_info

        return ps, iplus, iminus

    def apply_ssRF(self, ps: np.ndarray, iplus: np.ndarray, iminus: np.ndarray, return_burn_info: bool = False, profile_type: str = 'voigt') -> Tuple:
        """
        Apply ss-RF burn to ``ps`` and map to ``iplus`` / ``iminus``.

        Input arrays are copied; callers keep their originals unchanged.
        """
        ps = ps.copy()
        iplus = iplus.copy()
        iminus = iminus.copy()
        initial_ps = ps.copy()
        
        # Generate discretized Voigt/Gaussian profile across all bins
        ssRF = self._discretize_profile(profile_type=profile_type)
        
        burn_threshold = self.amp * 1e-1
        burn_indices = np.where(ssRF > burn_threshold)[0]
        ssRF_burn_values = ssRF[burn_indices]


        original_burn_values = ssRF_burn_values.copy()
        ### burn effect - clip so ps doesn't cross zero
        
        # Get current ps values at burn indices
        ps_at_burn = ps[burn_indices]
        
        # Determine burn direction based on initial ps sign
        for i in range(len(burn_indices)):
            if initial_ps[burn_indices[i]] < 0:
                # Initial ps is negative, burn should be positive (reduce magnitude)
                ssRF_burn_values[i] = np.abs(ssRF_burn_values[i])
            else:
                # Initial ps is positive, burn should be negative (reduce magnitude)
                ssRF_burn_values[i] = -np.abs(ssRF_burn_values[i])
        
        clipped_burn_values = self._constrain_burn_no_zero_crossing(ps_at_burn, ssRF_burn_values)
        
        # Store original iplus and iminus values at burn indices before applying the burn
        original_iplus_at_burn = iplus[burn_indices].copy()
        original_iminus_at_burn = iminus[burn_indices].copy()
        
        ps[burn_indices] += clipped_burn_values

        ### map ps to iplus and iminus
        iplus_burn_effect, iminus_burn_effect = self.map_signal(ps, burn_indices)
        
        ### burn to iplus, iminus
        iplus[burn_indices] = iplus_burn_effect[burn_indices]
        iminus[burn_indices] = iminus_burn_effect[burn_indices]
        
        # Calculate the actual change in iplus and iminus (the "height" of the burned peaks)
        actual_iplus_change = np.abs(iplus[burn_indices] - original_iplus_at_burn)
        actual_iminus_change = np.abs(iminus[burn_indices] - original_iminus_at_burn)
        
        mirrored_burn, mirrored_indices = self.mirror_burn_over_center(clipped_burn_values, burn_indices)

        iplus_change_at_mirror = None
        iminus_change_at_mirror = None

        # mirrored_burn /= 2.0
        
        if np.any(mirrored_indices):
            mirrored_burn[mirrored_indices] *= -1.0

            burn_at_mirrored = mirrored_burn[mirrored_indices].copy()
            ps_at_mirror = ps[mirrored_indices].copy()
            burn_at_mirrored = self._constrain_burn_no_zero_crossing(
                ps_at_mirror, burn_at_mirrored
            )

            original_iplus_at_mirror = iplus[mirrored_indices].copy()
            original_iminus_at_mirror = iminus[mirrored_indices].copy()

            ps[mirrored_indices] += burn_at_mirrored

            iplus_burn_delta = iplus_burn_effect[burn_indices] - original_iplus_at_burn
            iminus_burn_delta = iminus_burn_effect[burn_indices] - original_iminus_at_burn

            mirror_iplus_adjust = np.zeros_like(iplus)
            mirror_iminus_adjust = np.zeros_like(iminus)
            for burn_idx, mirror_idx, ip_delta, im_delta in zip(
                burn_indices, mirrored_indices, iplus_burn_delta, iminus_burn_delta
            ):
                mirror_iminus_adjust[mirror_idx] -= 0.5 * ip_delta
                mirror_iplus_adjust[mirror_idx] -= 0.5 * im_delta

            unique_mirror_indices = np.unique(mirrored_indices)
            iplus[unique_mirror_indices] += mirror_iplus_adjust[unique_mirror_indices]
            iminus[unique_mirror_indices] += mirror_iminus_adjust[unique_mirror_indices]
            
            iplus_change_at_mirror = np.abs(iplus[mirrored_indices] - original_iplus_at_mirror)
            iminus_change_at_mirror = np.abs(iminus[mirrored_indices] - original_iminus_at_mirror)
        
        if return_burn_info:
            burn_info = {
                'burn_indices': burn_indices,
                'mirror_indices': mirrored_indices if np.any(mirrored_indices) else None,
                'iplus_change_at_burn': actual_iplus_change,
                'iminus_change_at_burn': actual_iminus_change,
                'iplus_change_at_mirror': iplus_change_at_mirror,
                'iminus_change_at_mirror': iminus_change_at_mirror
            }
            return ps, iplus, iminus, burn_info
        

        return ps, iplus, iminus