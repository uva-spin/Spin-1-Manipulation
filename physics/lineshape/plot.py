import matplotlib.pyplot as plt
import numpy as np

from Lineshape import GenerateVectorLineshape

f = np.linspace(-3, 3, 249)
sigma = 0.04
gamma = 0.05
x0 = 0.88
amp = 0.7

signal, iplus, iminus = GenerateVectorLineshape(-0.65, f)

plt.figure(figsize=(12, 8))
plt.plot(f, signal, label="signal")
plt.plot(f, iplus, label="iplus")
plt.plot(f, iminus, label="iminus")
plt.legend()
plt.show()
plt.close()