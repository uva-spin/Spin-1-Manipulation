

import numpy as np

class AFP:
    def __init__(self, n_plus_init, n_minus_init, n_naught_init):
        self.n_plus = n_plus_init
        self.n_minus = n_minus_init
        self.n_naught = n_naught_init
        self.mu = 1

    def calc_P(self, Iplus, Iminus):
        return Iplus + Iminus

    def calc_Q(self, Iplus, Iminus):
        return Iplus - Iminus

    def calculate_n_plus(self, Iplus, Iminus):
        self.n_plus = self.mu * ( (1./3.) + (1./2.)*self.calc_P(Iplus, Iminus) + (1./6.)*self.calc_Q(Iplus, Iminus))
        return self.n_plus
    
    def calculate_n_minus(self, Iplus, Iminus):
        self.n_minus = self.mu * ( (1./3.) - (1./2.)*self.calc_P(Iplus, Iminus) + (1./6.)*self.calc_Q(Iplus, Iminus))
        return self.n_minus
    
    def calculate_n_naught(self, Iplus, Iminus):
        self.n_naught = self.mu * ( 1 - self.calc_Q(Iplus, Iminus)) / 3.0
        return self.n_naught

    def norm_mu(self, Iplus, Iminus):
        p = self.calc_P(Iplus, Iminus)
        denom = self.n_plus + self.n_minus + self.n_naught
        inv = 1.0 / denom
        self.mu = np.where(p == 0, 0.0, inv)

    