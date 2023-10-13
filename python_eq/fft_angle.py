import numpy as np

def fft_angle(Xcube, fft_Ang, Is_Windowed):
    Nr = Xcube.shape[0]  # Length of Chirp
    Ne = Xcube.shape[1]  # Length of receiver
    Nd = Xcube.shape[2]  # Length of chirp loop

    AngData = np.zeros((Nr, fft_Ang, Nd), dtype=complex)

    for i in range(Nd):
        for j in range(Nr):
            if Is_Windowed:
                win_xcube = np.reshape(Xcube[j, :, i], Ne, 1) * np.hanning(Ne)
            else:
                win_xcube = np.reshape(Xcube[j, :, i], Ne, 1) * 1

            AngData[j, :, i] = np.fft.fftshift(np.fft.fft(win_xcube, fft_Ang))

    return AngData
