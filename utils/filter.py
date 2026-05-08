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
    
    # Design the Butterworth filter
    b, a = butter(order, cutoff, fs=fs, btype='low')
    
    # Fallback to handle numpy arrays directly 
    if isinstance(data, np.ndarray):
        return filtfilt(b, a, data, axis=0)
        
    # Create a copy so we don't modify the original DataFrame
    filtered_data = data.copy()
    
    # Iterate over each column and apply the filter
    for col in filtered_data.columns:
        # Added 'frame#' to our exclusion list
        if col.lower() not in ['time', 'frame', 'frame_number', 'frame#']:
            # ONLY apply the filter if the column contains numeric data
            if np.issubdtype(filtered_data[col].dtype, np.number):
                filtered_data[col] = filtfilt(b, a, filtered_data[col])
            
    return filtered_data