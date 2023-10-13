import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.io import loadmat
import seaborn as sns
# Helper functions
def get_params_value():
    params = {}

    # constant parameters
    params['c'] = 299792458  # Speed of light in air (m/s)
    params['fc'] = 77e9  # Center frequency (Hz)
    params['lambda_'] = params['c'] / params['fc']
    params['Rx'] = 4
    params['Tx'] = 2

    # configuration parameters
    params['Fs'] = 4e6
    params['sweepSlope'] = 21.0017e12
    params['samples'] = 128
    params['loop'] = 255

    params['Tc'] = 120e-6  # us
    params['fft_Rang'] = 134  # 134=>128
    params['fft_Vel'] = 256
    params['fft_Ang'] = 128
    params['num_crop'] = 3
    params['max_value'] = 1e4  # data WITH 1843

    # Create grid table
    freq_res = params['Fs'] / params['fft_Rang']
    freq_grid = np.arange(params['fft_Rang']) * freq_res
    params['rng_grid'] = freq_grid * params['c'] / (params['sweepSlope'] * 2)

    w = np.linspace(-1, 1, params['fft_Ang'])
    params['agl_grid'] = np.arcsin(w) * 180 / np.pi

    dop_grid = np.fft.fftshift(np.fft.fftfreq(params['fft_Vel'], 1/params['Tc']))
    params['vel_grid'] = dop_grid * params['lambda_'] / 2

    return params

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


def fft_doppler(Xcube, fft_Vel, Is_Windowed):
    Nr, Ne, Nd = Xcube.shape

    DopData = np.zeros((Nr, Ne, fft_Vel), dtype=np.complex128)

    for i in range(Ne):
        for j in range(Nr):
            if Is_Windowed:
                win_dop = Xcube[j, i, :] * np.hanning(Nd)
            else:
                win_dop = Xcube[j, i, :]

            DopData[j, i, :] = np.fft.fftshift(np.fft.fft(win_dop, fft_Vel))

    return DopData


def plot_rangeDop(Dopdata_sum, rng_grid, vel_grid):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.heatmap(Dopdata_sum, cmap='viridis', cbar=True, cbar_kws={'label': 'Power'}, vmax=3e4, vmin=0)
    ax.set_xlabel('Doppler Velocity (m/s)')
    ax.set_ylabel('Range (meters)')
    ax.set_title('Range-Doppler heatmap')

    # plt.xlim([-8, 8])
    # plt.ylim([2, 25])
    plt.gca().invert_yaxis()
    plt.show()

def peakGrouping(detMat):
    numDetectedObjects = detMat.shape[1]
    objOut = np.empty((3, 0))

    # Sort the detMat matrix according to the cell power
    order = np.argsort(detMat[2, :])[::-1]
    detMat = detMat[:, order]

    for ni in range(numDetectedObjects):
        detectedObjFlag = 1
        rangeIdx = int(detMat[1, ni])
        dopplerIdx = int(detMat[0, ni])
        peakVal = detMat[2, ni]
        kernal = np.zeros((3, 3))

        # Fill the middle column of the kernel
        kernal[1, 1] = peakVal

        need_index = np.where((detMat[0, :] == dopplerIdx) & (detMat[1, :] == rangeIdx + 1))[0]
        if need_index.size > 0:
            kernal[0, 1] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx) & (detMat[1, :] == rangeIdx - 1))[0]
        if need_index.size > 0:
            kernal[2, 1] = detMat[2, need_index[0]]

        # Fill the left column of the kernel
        need_index = np.where((detMat[0, :] == dopplerIdx - 1) & (detMat[1, :] == rangeIdx + 1))[0]
        if need_index.size > 0:
            kernal[0, 0] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx - 1) & (detMat[1, :] == rangeIdx))[0]
        if need_index.size > 0:
            kernal[1, 0] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx - 1) & (detMat[1, :] == rangeIdx - 1))[0]
        if need_index.size > 0:
            kernal[2, 0] = detMat[2, need_index[0]]

        # Fill the right column of the kernel
        need_index = np.where((detMat[0, :] == dopplerIdx + 1) & (detMat[1, :] == rangeIdx + 1))[0]
        if need_index.size > 0:
            kernal[0, 2] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx + 1) & (detMat[1, :] == rangeIdx))[0]
        if need_index.size > 0:
            kernal[1, 2] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx + 1) & (detMat[1, :] == rangeIdx - 1))[0]
        if need_index.size > 0:
            kernal[2, 2] = detMat[2, need_index[0]]

        # Compare the detected object to its neighbors. Detected object is at index [1, 1]
        if kernal[1, 1] != np.max(kernal):
            detectedObjFlag = 0

        if detectedObjFlag == 1:
            objOut = np.hstack((objOut, detMat[:, ni].reshape(-1, 1)))

    return objOut


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

def normalize(Xcube, max_val):
    Xcube = Xcube / max_val
    Angdata = Xcube.astype('float32')  # Assuming you want single precision (32-bit) floating point
    return Angdata

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



def plot_rangeAng(Xcube, rng_grid, agl_grid):
    Nr = Xcube.shape[0]  # Length of Chirp (num of rangeffts)
    Ne = Xcube.shape[1]  # Number of angleffts
    Nd = Xcube.shape[2]  # Length of chirp loop

    # Polar coordinates
    yvalue = np.zeros((len(agl_grid), len(rng_grid)))
    xvalue = np.zeros((len(agl_grid), len(rng_grid)))

    for i in range(len(agl_grid)):
        yvalue[i, :] = (np.sin(agl_grid[i] * np.pi / 180) * rng_grid)
        xvalue[i, :] = (np.cos(agl_grid[i] * np.pi / 180) * rng_grid)

    # Plot 2D(range-angle)
    Xpow = np.abs(Xcube)
    Xpow = np.squeeze(np.sum(Xpow, axis=2) / Xpow.shape[2])

    Xsnr = Xpow
    # Assuming you have a variable noisefloor
    # Xsnr = Xpow / noisefloor

    fig = plt.figure(figsize=(7, 5))
    axh = fig.add_subplot(111, projection='3d')
    agl_grid, rng_grid = np.meshgrid(agl_grid, rng_grid)

    axh.plot_surface(agl_grid, rng_grid, Xsnr, cmap='viridis', rstride=1, cstride=1, alpha=0.9)
    axh.set_xlabel('Angle of Arrival (degrees)')
    axh.set_ylabel('Range (meters)')
    axh.set_zlabel('Power')
    axh.set_title('Range-Angle heatmap')
    axh.view_init(elev=90, azim=0)

    return axh

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

import matplotlib.pyplot as plt

def plot_pointclouds(detout):
    fig = plt.figure()
    axh = fig.add_subplot(111, projection='3d')
    axh.scatter3D(detout[:, 5], detout[:, 7], detout[:, 4], c=detout[:, 3], cmap='viridis', s=30)
    axh.set_xlabel('Doppler velocity (m/s)')
    axh.set_ylabel('Azimuth angle (degrees)')
    axh.set_zlabel('Range (m)')
    axh.set_title('3D point clouds')
    axh.set_xlim([-5, 5])
    axh.set_ylim([-60, 60])
    axh.set_zlim([2, 25])
    axh.grid(True)

    return axh



# Define constants and parameters
params = get_params_value()
c = params['c']  # Speed of light in air (m/s)
fc = params['fc']  # Center frequency (Hz)
lambda_ = params['lambda_']
Rx = params['Rx']
Tx = params['Tx']
Fs = params['Fs']
sweepSlope = params['sweepSlope']
samples = params['samples']
loop = params['loop']
Tc = params['Tc']  # us
fft_Rang = params['fft_Rang']
fft_Vel = params['fft_Vel']
fft_Ang = params['fft_Ang']
num_crop = params['num_crop']
max_value = params['max_value']

rng_grid = params['rng_grid']
agl_grid = params['agl_grid']
vel_grid = params['vel_grid']


data_each_frame = samples * loop * Tx
set_frame_number = 30
frame_start = 1
frame_end = set_frame_number
Is_Windowed = 1
Is_plot_rangeDop = 1

seq_name = 'pms1000_30fs.mat'
seq_dir = './template data/' + seq_name
data_frames = loadmat(seq_dir)
data_frames = data_frames['data_frames']

for i in range(frame_start, frame_end + 1):
    data_frame = data_frames[:, (i - 1) * data_each_frame:i * data_each_frame]
    data_chirp = np.zeros((4, samples, Tx*loop), dtype=np.complex128)

    for cj in range(1, Tx*loop + 1):
        temp_data = data_frame[:, (cj-1)*samples:cj*samples]
        data_chirp[:, :, cj-1] = temp_data

    chirp_odd = np.array(data_chirp[:, :, ::2])
    chirp_even = np.array(data_chirp[:,:,1::2])

    chirp_odd = np.transpose(chirp_odd, (1, 0, 2))
    chirp_even = np.transpose(chirp_even, (1, 0, 2))
    Rangedata_odd = fft_range(chirp_odd, fft_Rang, Is_Windowed)
    Rangedata_even = fft_range(chirp_even, fft_Rang, Is_Windowed)
    Dopplerdata_odd = fft_doppler(Rangedata_odd, fft_Vel, 0)
    Dopplerdata_even = fft_doppler(Rangedata_even, fft_Vel, 0)
    Dopdata_sum = np.mean(np.abs(Dopplerdata_odd), axis=1)
    if Is_plot_rangeDop:
        plot_rangeDop(Dopdata_sum, rng_grid, vel_grid)

    Pfa = 1e-4
    Resl_indx = cfar_RV(Dopdata_sum, fft_Rang, num_crop, Pfa)
    detout = peakGrouping(Resl_indx)

    for ri in range(num_crop + 1, fft_Rang - num_crop):
        find_idx = np.where(detout[1, :] == ri)[0]
        if len(find_idx) == 0:
            continue
        else:
            pick_idx = find_idx[0]
            pha_comp_term = np.exp(-1j * np.pi * (detout[0, pick_idx] - fft_Vel / 2 - 1) / fft_Vel)
            Rangedata_even[ri, :, :] = Rangedata_even[ri, :, :] * pha_comp_term

    Rangedata_merge = np.concatenate((Rangedata_odd, Rangedata_even), axis=2)

    Angdata = fft_angle(Rangedata_merge, fft_Ang, Is_Windowed)
    Angdata_crop = Angdata[num_crop + 1:fft_Rang - num_crop, :, :]
    Angdata_crop = normalize(Angdata_crop, max_value)

    Dopplerdata_merge = np.transpose([Dopplerdata_odd, Dopplerdata_even], (1, 0, 2))
    Resel_agl, _, rng_excd_list = angle_estim_dets(detout, Dopplerdata_merge, fft_Vel, fft_Ang, Rx, Tx, num_crop)

    Resel_agl_deg = agl_grid[0, Resel_agl]
    Resel_vel = vel_grid[detout[0, :], 0]
    Resel_rng = rng_grid[detout[1, :], 0]

    save_det_data = np.column_stack((detout[1, :], detout[0, :], Resel_agl, detout[2, :], Resel_rng, Resel_vel, Resel_agl_deg))

    if len(rng_excd_list) > 0:
        save_det_data = np.delete(save_det_data, rng_excd_list, axis=0)

    plot_rangeAng(Angdata_crop, rng_grid[num_crop + 1:fft_Rang - num_crop], agl_grid)
    plot_pointclouds(save_det_data)

    break
