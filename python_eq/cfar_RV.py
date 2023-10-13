import numpy as np
from scipy.signal import convolve

def cfar_ca1D_square(x, num_train, num_guard, Pfa, method=0, offset=0):
    # Apply CA-CFAR algorithm
    N = len(x)
    num_reference = num_train + num_guard
    num_signal = num_train

    # Calculate threshold
    if method == 0:  # cell average method
        threshold = (1 + offset) * np.mean(x[:num_train])

    elif method == 1:  # greatest-of-constant false alarm rate
        q = np.percentile(x[:num_train], 100 * (1 - Pfa))
        threshold = (1 + offset) * q

    elif method == 2:  # greatest-of-constant false alarm rate (scaled)
        q = np.percentile(x[:num_train], 100 * (1 - Pfa))
        threshold = (1 + offset) * (q / num_train)

    else:
        raise ValueError("Invalid method. Choose from 0, 1, or 2.")

    # Convolve to get the sliding window sum
    sliding_sum = convolve(x, np.ones(num_reference), mode='valid')

    # Apply threshold
    detections = np.where(sliding_sum[num_guard:] > threshold)[0] + num_guard

    return detections

def cfar_RV(Dopdata_sum, fft_Rang, num_crop, Pfa):
    x_dop = []  # Temporary storage
    Resl_indx = []  # Store CFAR detections

    for rani in range(num_crop + 1, fft_Rang - num_crop):
        x_detected = cfar_ca1D_square(Dopdata_sum[rani, :], 4, 7, Pfa, 0, 0.7)
        x_dop.extend(x_detected)

    # Make unique
    C = np.unique(x_dop)

    # CFAR for each specific doppler bin
    for dopi in C:
        y_detected = cfar_ca1D_square(Dopdata_sum[:, int(dopi)], 4, 8, Pfa, 0, 0.7)
        if y_detected.size == 0:
            continue
        for yi in range(y_detected.shape[0]):
            # Saving format: [doppler index, range index (start from index 1), cell power]
            Resl_indx.append([int(dopi), int(y_detected[yi, 0]), y_detected[yi, 1]])

    return np.array(Resl_indx)
