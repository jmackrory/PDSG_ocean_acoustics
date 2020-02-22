from datetime import datetime
from time import time
from multiprocessing import Pool
from functools import lru_cache
import os
import sys
import pickle

import numpy as np
from numpy.fft import fft, ifft
from pydub import AudioSegment
from const import SAMPLE_LENGTH, SAMPLES_PER_SEC, NUM_WORKERS


def get_var_size():
    for obj in locals.items():
        if isinstance(obj, np.ndarray):
            print(obj, obj.nbytes/(1024**2))
        else:
            print(obj, getsizeof(obj))


def get_file_path(datetime, data_type='every_other_hour'):
    """get_file_path

    Args datetime - datetime for

    """
    year = datetime.year
    month = datetime.month
    day = datetime.day
    hour = datetime.hour
    minute = datetime.minute

    # stuff specific to reduce dataset
    if minute %5 !=0:
        raise Exception('Files only allowed to being on 5 minute mark')
    if data_type == 'month' and month!=2:
        raise Exception('Only have data for February!')
    if data_type =='every_other_hour' and hour%2 != 0:
        raise Exception('every_other_hour has data only for even hours!')

    dirpath = f'data/{data_type}/{year}/{month:0=2}/{day:0=2}/'
    fpath = f'{year}-{month:0=2}-{day:0=2}--{hour:0=2}.{minute:0=2}.mp3'
    return dirpath + fpath


def load_mp3_arr(fpath):
    mp3 = AudioSegment.from_file(fpath, format='mp3')
    return mp3.get_array_of_samples()


def load_mp3_arr_from_date(datetime, data_type='every_other_hour'):
    fpath = get_file_path(datetime)
    return load_mp3_arr(fpath)


def get_power_spectrum(ts):
    # subtract off DC component

    #dt = 1.0 / SAMPLES_PER_SEC
    #dw = 2*np.pi / dt

    Nt = len(ts)
    fw = fft(ts)
    fw = np.abs(fw)*np.abs(fw)
    return fw


def high_pass_filter(w, w0, n):
    return w**n / (w0**n + w**n)


def low_pass_filter(w, w0, n):
    return w0**n / (w0**n + w**n)


def get_freqs():
    fmax = SAMPLES_PER_SEC / 2
    df = 1.0 / SAMPLE_LENGTH
    return np.arange(0, fmax, df)


def load_file_and_get_power(fpath):
    marr = load_mp3_arr(fpath)
    return get_power_spectrum(marr)


def get_wiener_filter(data_path='data/every_other_hour/2015/01/17'):
    # load all files.
    power_spec = 0
    ncount = 0
    fpaths = []
    for root, dirs, files in os.walk(data_path):
        fpaths += [f'{root}/{fn}' for fn in files if 'mp3' in fn]
    ncount = len(fpaths)
    print(fpaths)
    num_batch = int(ncount / NUM_WORKERS)+1
    t0 = time()
    print(num_batch)
    for i in range(num_batch):
        t1 = time()
        sl = slice(i*num_batch, (i+1)*num_batch)
        with Pool(processes=NUM_WORKERS) as p:
            power_specs = p.map(load_file_and_get_power, fpaths[sl])
        power_spec += np.array(power_specs).sum(axis=0)
        t2 = time()
        print(f'{t2 - t1} sec.  {(t2 - t0) / (i + 0.001) * (num_batch - i)} sec remaining')
    return power_spec / (num_batch + 0.001)


def load_filter_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as pp:
        filt = pickle.load(pp)
    return filt


def get_band_pass_filter(low_f=20, high_f = 8000):
    pass


def remove_noise_wiener(ts, avg_noise_power):
    """
    Cheap version of Wiener filter.  Tries to filter out average noise.
    Assume input filter $h$ is delta-response.  Also assumes spectrum of signal
    is just given by the file.  
    Arg:
    ts - np.array input sample of length 300 sec
    avg_noise_power - np.array with average power spectrum.  (Must be same sample_rate, length as ts!)
    Return:
    filtered_fw - ts with noise hopefully reduced.
    """
    fw = fft(ts)
    Ns = len(ts)
    filtered_fw = fw * (Ns* fw / (Ns*fw + avg_noise_power))
    print(max(abs(fw[100:-100:])**2), max(avg_noise_power[100:-100]))
    filtered_ts = np.real(ifft(filtered_fw))
    return filtered_ts.astype(np.int16), np.abs(fw).astype(np.float32), np.abs(filtered_fw).astype(np.float32)

# add

