import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

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
    
    # Butterworth filter
    b, a = butter(order, cutoff, fs=fs, btype='low')
    
    # Fallback to handle numpy arrays directly 
    if isinstance(data, np.ndarray):
        return filtfilt(b, a, data, axis=0)
        
    filtered_data = data.copy()
    
    # Iterate over each column and apply the filter
    for col in filtered_data.columns:
        if col.lower() not in ['time', 'frame', 'frame#']:
            if np.issubdtype(filtered_data[col].dtype, np.number): # ONLY apply the filter if the column contains numeric data
                filtered_data[col] = filtfilt(b, a, filtered_data[col])
            
    return filtered_data