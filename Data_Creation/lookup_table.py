import numpy as np
import pandas as pd
import tqdm

g = 0.05
s = 0.04
bigy = np.sqrt(3 - s)


def lineshape(x, eps):
    def cosal(x, eps):
        return (1 - eps * x - s) / bigxsquare(x, eps)

    def bigxsquare(x, eps):
        return np.sqrt(g ** 2 + (1 - eps * x - s) ** 2)

    def mult_term(x, eps):
        return 1 / (2 * np.pi * np.sqrt(bigxsquare(x, eps)))

    def cosaltwo(x, eps):
        return np.sqrt((1 + cosal(x, eps)) / 2)

    def sinaltwo(x, eps):
        return np.sqrt((1 - cosal(x, eps)) / 2)

    def termone(x, eps):
        return np.pi / 2 + np.arctan((bigy**2 - bigxsquare(x, eps)) / (2 * bigy * np.sqrt(bigxsquare(x, eps)) * sinaltwo(x, eps)))

    def termtwo(x, eps):
        return np.log((bigy**2 + bigxsquare(x, eps) + 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)) /
                    (bigy**2 + bigxsquare(x, eps) - 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)))

    def icurve(x, eps):
        return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))

    return icurve(x, eps) / 10


def generate_vector_lineshape(polarization, x):
    r = (np.sqrt(4 - 3 * polarization**2) + polarization) / (2 - 2 * polarization)

    i_plus_sign = 1
    i_minus_sign = 1
    if polarization > 0:
        pass
    else:
        r = 1 / r
        i_plus_sign = -1
        i_minus_sign = -1

    iplus = i_plus_sign * r * lineshape(x, 1)
    iminus = i_minus_sign * lineshape(x, -1)

    total = iplus + iminus
    delta_p = (polarization / np.sum(total))
    iplus *= delta_p
    iminus *= delta_p

    signal = iplus + iminus
    return signal, iplus, iminus


def main():
    # p_negative = np.arange(0.0, -0.00001, 0.00001)
    # p_positive = np.arange(0.00001, 0.7, 0.00001)
    # polarizations = np.concatenate([p_negative, p_positive])
    polarizations = np.arange(-0.7, 0.70, 0.001)
    num_bins = 249
    x_grid = np.linspace(-3, 3, num_bins)

    ps_values = np.empty((len(polarizations), num_bins))
    qs_values = np.empty((len(polarizations), num_bins))
    i_minus_values = np.empty((len(polarizations), num_bins))
    i_plus_values = np.empty((len(polarizations), num_bins))

    for i, polarization in enumerate(tqdm.tqdm(polarizations, desc="Processing polarization")):
        _, iplus, iminus = generate_vector_lineshape(polarization, x_grid)

        # for j in range(num_bins):
        #     if iplus[j] > iminus[j]:
        #         i_plus_values[i, j] = iminus[j]
        #         i_minus_values[i, j] = iplus[j]
        #     else:
        #         i_plus_values[i, j] = iplus[j]
        #         i_minus_values[i, j] = iminus[j]
        #     ps_values[i, j] = i_plus_values[i, j] + i_minus_values[i, j]
        #     qs_values[i, j] = i_plus_values[i, j] - i_minus_values[i, j]
        
        i_plus_values[i, :] = iplus
        i_minus_values[i, :] = iminus
        ps_values[i, :] = iplus + iminus
        qs_values[i, :] = iplus - iminus

    df = pd.DataFrame(
        {
            "P": polarizations,
            "Ps": [ps_values[i, :] for i in range(len(polarizations))],
            "Qs": [qs_values[i, :] for i in range(len(polarizations))],
            "Iminus": [i_minus_values[i, :] for i in range(len(polarizations))],
            "Iplus": [i_plus_values[i, :] for i in range(len(polarizations))],
        }
    )
    df = df.dropna()
    df.to_pickle("lookup_table.pkl")


if __name__ == "__main__":
    main()

