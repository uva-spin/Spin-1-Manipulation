import numpy as np
import matplotlib.pyplot as plt

### load unmapip_bin_0200.npz and plot range of I+ and I-

data = np.load('unmanip_bin_0227.npz')
iplus = data['iplus']
iminus = data['iminus']

plt.plot(iplus, label='I+')
plt.plot(iminus, label='I-')
plt.legend()
plt.show()