from datetime import datetime
from time import time
from multiprocessing import Pool
import os

import numpy as np
from numpy.fft import fft, ifft
from pydub import AudioSegment
from const import FILE_LENGTH, SAMPLES_PER_SEC, NUM_WORKERS


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
    return power_spec / (ncount + 0.001)

def get_band_pass_filter(low_f=20, high_f = 8000):
    pass

def remove_noise(ts, freq_filter):


    pass


# add
