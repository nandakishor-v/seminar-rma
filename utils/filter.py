import pandas as pd
from scipy import signal

def butterworth_lowpass_filter(data, cutoff, fs, order=2):
    """Apply a zero-lag Butterworth low-pass filter to the data.
    
    Parameters:
    - data: pandas DataFrame with the data to be filtered
    - cutoff: cutoff frequency in Hz
    - fs: sampling frequency in Hz
    - order: order of the Butterworth filter (default is 2)
    
    Returns:
    - filtered_data: pandas DataFrame with the filtered data
    """

    # Design the Butterworth filter
    # 1. Calculate the Nyquist frequency (half the sampling rate)
    nyq = 0.5 * fs
    # 2. Normalize the cutoff frequency
    normal_cutoff = cutoff / nyq
    # 3. Get the filter coefficients (b, a)
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Create a copy of the dataframe to avoid modifying the original data
    filtered_data = data.copy()

    # Iterate over each column and apply the filter
    for col in filtered_data.columns:
        # !Do not filter time or frame number columns!
        # Checking if 'time' or 'frame' is in the column name (case-insensitive)
        if 'time' in col.lower() or 'frame' in col.lower():
            continue
            
        # Apply the forward-backward filter to avoid phase shift
        filtered_data[col] = signal.filtfilt(b, a, data[col])

    return filtered_data