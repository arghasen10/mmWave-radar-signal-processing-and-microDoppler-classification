import numpy as np

def fft_doppler(Xcube, fft_Vel, Is_Windowed):
    Nr, Ne, Nd = Xcube.shape

    DopData = np.zeros((Nr, Ne, fft_Vel), dtype=np.complex128)

    for i in range(Ne):
        for j in range(Nr):
            for k in range(Nd):
                if Is_Windowed:
                    win_dop = Xcube[j, i, k] * np.hanning(Nd)
                else:
                    win_dop = Xcube[j, i, k]

                DopData[j, i, :] = np.fft.fftshift(np.fft.fft(win_dop, fft_Vel))

    return DopData
