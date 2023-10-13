import numpy as np

def angle_estim_dets(detout, Velocity_FFT, fft_Vel, fft_Ang, Rx, Tx, num_crop):
    Resel_agl = []
    vel_ambg_list = []
    rng_excd_list = []
    fft_Rang = Velocity_FFT.shape[0]

    for ai in range(detout.shape[1]):
        rx_vect = np.squeeze(Velocity_FFT[:, detout[1, ai], detout[0, ai]])

        # Phase Compensation on the range-velocity bin for virtual elements
        pha_comp_term = np.exp(-1j * (np.pi * (detout[0, ai] - fft_Vel/2 - 1) / fft_Vel))
        rx_vect[Rx:Rx*Tx] = rx_vect[Rx:Rx*Tx] * pha_comp_term

        # Estimate Angle on set1
        Angle_FFT1 = np.fft.fftshift(np.fft.fft(rx_vect, fft_Ang))
        II = np.argmax(np.abs(Angle_FFT1))
        Resel_agl.append(II)

        # Velocity disambiguation on set2 -- flip the sign of the symbols 
        # corresponding to Tx2
        rx_vect[Rx:Rx*Tx] = -rx_vect[Rx:Rx*Tx]
        Angle_FFT1_flip = np.fft.fftshift(np.fft.fft(rx_vect, fft_Ang))
        II_flip = np.argmax(np.abs(Angle_FFT1_flip))

        MM_flip = np.max(np.abs(Angle_FFT1_flip))
        MM = np.max(np.abs(Angle_FFT1))

        if MM_flip > 1.2 * MM:
            # now has velocity ambiguration, need to be corrected 
            vel_ambg_list.append(ai)

        if detout[1, ai] <= num_crop or detout[1, ai] > fft_Rang - num_crop:
            rng_excd_list.append(ai)

    return Resel_agl, vel_ambg_list, rng_excd_list
