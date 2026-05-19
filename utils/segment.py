import numpy as np
import pandas as pd

def segment_gait_cycles(grf_y_column, data, threshold=60):
    """Segment gait cycles based on vertical ground reaction force (GRF) data.
    
    Parameters:
    - grf_y_column: pandas Series with vertical GRF data
    - threshold: force threshold to detect foot contact (default is 60 N)
    - data: optional pandas DataFrames with additional data to segment (e.g. kinematics)
    
    Returns:
    - segments: lists of DataFrames with segmented data if additional_data is provided
    """
    grf_arr = np.asarray(grf_y_column)
    
    # IDENTIFY HEEL STRIKES
    above_thresh = (grf_arr > threshold).astype(int)    # boolean array where force is above the threshold
    heel_strikes = np.where(np.diff(above_thresh) == 1)[0] + 1 # exact moment the signal goes from 0 to 1
    
    # EXTRACT CYCLES & APPLY FILTER 1 (Min Force >= 300N)
    raw_cycles = []
    for i in range(len(heel_strikes) - 1):
        start_idx = heel_strikes[i]
        end_idx = heel_strikes[i+1]
        
        cycle_df = data.iloc[start_idx:end_idx].copy()
        
        if grf_arr[start_idx:end_idx].max() >= 300:
            raw_cycles.append(cycle_df)
            
    if not raw_cycles:
        return []
        
    durations = [len(cycle) for cycle in raw_cycles]
    mean_dur = np.mean(durations)
    std_dur = np.std(durations)
    
    segments = []
    for cycle in raw_cycles:
        # Keeping cycles within +/- 2 standard deviations
        if abs(len(cycle) - mean_dur) <= 2 * std_dur:
            
            # checking if the time starts at 0 in each returned segment
            if 'time' in cycle.columns:
                cycle['time'] = cycle['time'] - cycle['time'].iloc[0]
                
            segments.append(cycle)
            
    return segments


def ensemble_average(cycles):
    """Compute the ensemble average and standard deviation of segmented gait cycles.
    
    Parameters:
    - cycles: list of pandas DataFrames, each containing one gait cycle
    
    Returns:
    - mean_cycle: pandas DataFrame with the mean values across all cycles
    - std_cycle: pandas DataFrame with the standard deviation across all cycles
    """
    if len(cycles) == 0:
        return None, None
        
    num_points = 100
    norm_t = np.linspace(0, 1, num_points)  # Normalized time vector from 0 to 1 for interpolation
    
    # Identify numeric columns to average (skipping string labels if any exist)
    numeric_cols = [col for col in cycles[0].columns if np.issubdtype(cycles[0][col].dtype, np.number)]
    resampled_data = {col: [] for col in numeric_cols}
    
    # Resample each cycle to exactly 100 points
    for cycle in cycles:
        orig_t = np.linspace(0, 1, len(cycle))
        
        for col in numeric_cols:
            y = cycle[col].to_numpy()
            y_interp = np.interp(norm_t, orig_t, y)
            resampled_data[col].append(y_interp)
            
    # Calculate mean and standard deviation
    mean_dict = {}
    std_dict = {}
    
    for col in numeric_cols:
        matrix = np.array(resampled_data[col])
        mean_dict[col] = np.mean(matrix, axis=0)
        std_dict[col] = np.std(matrix, axis=0)
        
    mean_cycle = pd.DataFrame(mean_dict)
    std_cycle = pd.DataFrame(std_dict)
    
    return mean_cycle, std_cycle