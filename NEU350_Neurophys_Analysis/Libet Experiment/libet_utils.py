"""
Utils file for NEU350 libet experiment data wrangling. Written by JRB 21-03-31
"""

import os
import pandas as pd
from scipy.io import wavfile
import numpy as np
import pickle as pkl

def load_libet(file_prefix, student_initials):
    """
    Load data from the Libet Task Experiment. The data (.wav, .txt, .csv)
    should all be in the same folder as the notebook and script
    
    Parameters
    ----------
    file_prefix      : str, prefix of file used for experiment, e.g. 'libet_0'
    student_initials : str, initials of student running this function, e.g. 'JRB'
    
    Returns
    -------
    libet            : dict, containing voltage snipets allgined to the two different
                       task protocols (p1 and p2), along with summary data
    """
    
    ## === initial imports
    # initialize names
    wav_file = file_prefix + ".wav"
    event_file = file_prefix + "-events.txt"
    csv_file = file_prefix + '.csv'
    
    # load wav
    sample_rate, wav_data = wavfile.read(wav_file)
    wav_data_matx = wav_data.reshape(wav_data.shape[0], 1)
    wav_length = len(wav_data)/sample_rate
    
    # load text
    event_data = pd.read_csv(event_file, header=1)
    event_data = clean_event_data(event_data)
    
    # load csv
    task_data = pd.read_csv(csv_file, header=2)
    
    # make dict
    libet = {}
    libet['student'] = student_initials
    libet['sample_rate'] = sample_rate
    
    ## === get info for first protocol (trials 0 - 19)
    
    libet = get_p1_data(sample_rate, event_data, wav_data, libet,pre_s=1.5, post_s=0.5)
    
    # === get info for second protocol (trials 20 - 39)
    
    libet = get_p2_data(sample_rate, event_data, wav_data, task_data, libet, pre_s=1.5, post_s=0.5)

    # === save out
    # saves as 'libet_student_initals.pkl' in cwd
    output = open('libet_{name}.pkl'.format(name = student_initials), 'wb')
    pkl.dump(libet, output)
    output.close()
    
    print('Your experiment data has been saved! \nCheck this directory for a .pkl file to submit')
    

def clean_event_data(df):
    """
    Quick function for creating a dataframe that has start and stop
    times as the columns rather than a all times as one column
    """
    df_cleaned = pd.DataFrame()

    # grab columns out for cleaner indexing
    event = df['# Marker ID']
    times = df['\tTime (in s)']

    # initialize space
    start_times = []
    stop_times =[]
    
    # deal with extra trials
    N_trials = 40

    # iterate over trils from event texts
    for ievent, _ in df.iterrows():

        if 'Next' in event[ievent]:
            start_times.append(times[ievent])

        elif 'Stop' in event[ievent]:
            stop_times.append(times[ievent])

        else:
            print('trial type unknown')

    df_cleaned['trial_num'] = np.arange(N_trials)
    df_cleaned['start_times'] = start_times[N_trials] 
    df_cleaned['stop_times'] = stop_times[:N_trials]

    return df_cleaned

def get_p1_data(sample_rate, event_data, wav_data, libet, pre_s=1.5, post_s=0.5):
    """
    Function fot getting .wav data for libet protocol 1 aligned to the stop
    time (ie button press)
    
    Parameters
    ----------
    sample_rate : int, sample rate of BYB spikerboard from .wav import
    event_data  : df, containing output from BYB spike recorder that has been wrangled
    wav_data    : arr, .wav recording file from task
    pre_s       : int, window of time in s to grab before stop time
    pos_s       : int, window of time in s to grab after stop time
    
    Returns
    -------
    libet dictionary with appended contents : 
        p1_wavs     : list, len = n p1 trials containing .wav snippet from pre to post aligned
                      to stop time of the trial
        p1_stop_idx : int, index to use for plotting when you want to mark the stop_time event 
        p1_mean_wav : arr, mean of p1_wavs
        p1_std_wav  : arr, standard dev of p1_wavs

    """
    # define window & initilize space
    pre = pre_s * sample_rate
    post = post_s * sample_rate
    p1_wavs = []

    for itrial in range(20):

        # convert event time from s to sample rate index
        stop_sample = event_data['stop_times'][itrial] * sample_rate
        t1 = int(stop_sample - pre)
        t2 = int(stop_sample + post)

        # grab wav snippet and alignment index
        p1_wavs.append(wav_data[t1:t2])
        p1_stop_idx = int(pre)

    # get average waveform and sd
    p1_mean_wav = np.mean(p1_wavs, axis = 0)
    p1_std_wav = np.std(p1_wavs, axis = 0)
    
    # append to dictionary
    libet['p1_wavs'] = p1_wavs
    libet['p1_stop_idx'] = p1_stop_idx
    libet['p1_mean_wav'] = p1_mean_wav
    libet['p1_std_wav'] = p1_std_wav

    return libet


def get_p2_data(sample_rate, event_data, wav_data, task_data, libet, pre_s=1.5, post_s=0.5):
    
    """
    Function fot getting .wav data for libet protocol 2 aligned to the urge
    time
    
    Parameters
    ----------
    sample_rate : int, sample rate of BYB spikerboard from .wav import
    event_data  : df, containing output from BYB spike recorder that has been wrangled
    wav_data    : arr, .wav recording file from task
    task_data   : df, containing output from libet GUI
    pre_s       : int, window of time in s to grab before stop time
    pos_s       : int, window of time in s to grab after stop time
    
    Returns
    -------
    libet dictionary with appended contents : 
    
        p2_wavs      : list, len = n p2 trials, containing .wav snippet from pre to post aligned
                       to stop time of the trial
        p2_urge_idx  : int, index to use for plotting when you want to mark the urge_time event 
        p2_stop_idxs : arr, len = n p2 trials, stop idx for each trial 
        p2_mean_wav  : arr, mean of p2_wavs
        p2_std_wav   : arr, standard dev of p2_wavs

    """
    
    # define window & initialize
    pre = pre_s * sample_rate
    post = post_s * sample_rate

    p2_wavs = []
    
    # these change each trial because we are aligning to urge
    p2_stop_idxs = [] 

    for itrial in range(20,40):
        
        # these are defined relative to the start of a trial in the GUI
        # additionally, there is a bug where urge time can be greater than stop time
        # due to the clock resetting at the top
        task_stop_time = task_data['stop_time_msecs'][itrial]
        task_urge_time = task_data['urge_time_msecs'][itrial]

        # if the bug is present, subtract off one revolution of clock to correct
        if task_urge_time > task_stop_time:
            task_urge_time = task_urge_time - 1000
           
        # compute difference in stop and urge & convert to seconds
        stop_urge_diff = (task_stop_time - task_urge_time)/1000
        
        # infer urge time in BYB data given task info from GUI
        stop_time = event_data['stop_times'][itrial]
        urge_time = stop_time - stop_urge_diff

        # convert from s to sample rate
        stop_sample = stop_time * sample_rate
        urge_sample = urge_time * sample_rate

        # time delay between stop and urge in sample rate to use for idxs
        delay = stop_sample - urge_sample

        # grab window of wav around urge
        t1 = int(urge_sample - pre)
        t2 = int(urge_sample + post)

        # grab wave snipped in window around the urge time
        p2_wavs.append(wav_data[t1:t2])

        # save the idx for easier plotting
        p2_urge_idx = int(pre)
        p2_stop_idxs.append(int(pre + delay))


    # get average waveform and sd
    p2_mean_wav = np.mean(p2_wavs, axis = 0)
    p2_std_wav = np.std(p2_wavs, axis = 0)
    
    # append to dictionary
    libet['p2_wavs'] = p2_wavs
    libet['p2_urge_idx'] = p2_urge_idx
    libet['p2_stop_idxs'] = p2_stop_idxs
    libet['p2_mean_wav'] = p2_mean_wav
    libet['p2_std_wav'] = p2_std_wav
    
    return libet
    