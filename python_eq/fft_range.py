import numpy as np

def fft_range(Xcube, fft_Rang, Is_Windowed):
    Nr, Ne, Nd = Xcube.shape

    Rangedata = np.zeros((fft_Rang, Ne, Nd), dtype=np.complex128)

    for i in range(Ne):
        for j in range(Nd):
            if Is_Windowed:
                win_rng = Xcube[:, i, j] * np.hanning(Nr)
            else:
                win_rng = Xcube[:, i, j]
            Rangedata[:, i, j] = np.fft.fft(win_rng, fft_Rang)

    return Rangedata
