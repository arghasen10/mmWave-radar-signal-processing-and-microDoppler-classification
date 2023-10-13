import numpy as np

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
    params['vel_grid'] = dop_grid * params['lambda'] / 2

    return params
