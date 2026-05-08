

import numpy as np
import tqdm

class AFP:
    def __init__(self, n_plus_init, n_minus_init, n_naught_init):
        self.n_plus = n_plus_init
        self.n_minus = n_minus_init
        self.n_naught = n_naught_init
        self.mu = 1.

    def calc_P_R(self, Iplus, Iminus):
        return Iplus + Iminus

    def calc_Q_R(self, Iplus, Iminus):
        return Iplus - Iminus

    def calc_P_theta(self, Iplus, Iminus):
        ### go over each bin and add Iplus(R) + Iminus(-R), opposite bin, to get theta space
        P_theta = np.zeros(len(Iplus))
        for i in range(len(Iplus)):
            P_theta[i] = Iplus[i] + Iminus[len(Iplus) - i - 1] + Iplus[len(Iplus) - i - 1] + Iminus[i]
        return P_theta

    def calc_Q_theta(self, Iplus, Iminus):
        ### go over each bin and add Iplus(R) - Iminus(-R), opposite bin, to get theta space
        Q_theta = np.zeros(len(Iplus))
        for i in range(len(Iplus)):
            Q_theta[i] = Iplus[i] - Iminus[len(Iplus) - i - 1] + Iplus[len(Iplus) - i - 1] - Iminus[i]
        return Q_theta

    def calculate_n_plus(self, Iplus, Iminus):
        self.n_plus = np.array( self.mu * ( (1./3.) + (1./2.)*self.calc_P_R(Iplus, Iminus) + (1./6.)*self.calc_Q_R(Iplus, Iminus)))
        return self.n_plus
    
    def calculate_n_minus(self, Iplus, Iminus):
        self.n_minus = np.array( self.mu *( (1./3.) - (1./2.)*self.calc_P_R(Iplus, Iminus) + (1./6.)*self.calc_Q_R(Iplus, Iminus)))
        return self.n_minus
    
    def calculate_n_naught(self, Iplus, Iminus):
        self.n_naught = np.array(( self.mu - self.calc_Q_R(Iplus, Iminus)) / 3.0)
        return self.n_naught

    def calculate_n_naught_theta(self, Iplus, Iminus):
        self.n_naught = np.array(( self.mu - self.calc_Q_theta(Iplus, Iminus)) / 3.0)
        return self.n_naught

    def norm_mu(self, Iplus, Iminus):
        p = self.calc_P_R(Iplus, Iminus)
        denom = self.n_plus + self.n_minus + self.n_naught
        inv = 1.0 / denom
        self.mu = np.where(p == 0, 0.0, inv)
        print(f"mu: {self.mu}")

    def swap_pops(self, pop1_name, pop1_idx, pop2_name, pop2_idx):
        pop1 = getattr(self, pop1_name)
        pop2 = getattr(self, pop2_name)
        pop1[pop1_idx], pop2[pop2_idx] = pop2[pop2_idx], pop1[pop1_idx]

    def perform_afp(self, steps, subset_indices=None):
        bins = len(self.n_plus)
        if subset_indices is None:
            sweep_indices = list(range(steps))
        else:
            sweep_indices = list(subset_indices)[:steps]

        for idx in tqdm.tqdm(sweep_indices, desc="Performing AFP"):
            mirror_idx = bins - idx - 1
            print(f"idx: {idx}, mirror_idx: {mirror_idx}")
            print(f"n_plus: {self.n_plus[idx]}, n_naught: {self.n_naught[idx]}")
            print(f"n_naught mirror: {self.n_naught[mirror_idx]}, n_minus: {self.n_minus[mirror_idx]}")
            ### swap m = 1 and m = 0 at selected index
            self.swap_pops("n_plus", idx, "n_naught", idx)
            
            ### and swap m = 0 and m = -1 at mirrored selected index
            self.swap_pops("n_naught", idx, "n_minus", mirror_idx)
            if idx == bins // 2:
                ### set all population levels at center index to average of all population levels
                average_pop = np.mean(self.n_plus[idx] + self.n_minus[idx] + self.n_naught[idx])/3.
                self.n_plus[idx] = average_pop
                self.n_minus[idx] = average_pop
                self.n_naught[idx] = average_pop
        return self.n_plus, self.n_minus, self.n_naught



    