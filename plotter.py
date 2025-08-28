import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self) -> None:
        pass
    
    def trajectory_plot(self, 
                        time_snaps: np.ndarray, 
                        time: np.ndarray, 
                        snapshots: np.ndarray, 
                        samples: np.ndarray, 
                        figsize: tuple = (12, 8), 
                        title: str = "Trajectory",
                        xtitle: str = "t",
                        ytitle: str | None = None,
                        title_y: float = 0.95,  # Reduced default value
                        title_x: float = 0.5,   
                        shade: float = 0,
                        grid: bool = False,
                        plot_all_samples: bool = True,  # New parameter
                        confidence_level: float = 0.95,  # New parameter for CI
                        truth_time: np.ndarray | None = None,  # Ground truth time axis
                        truth_data: np.ndarray | None = None,  # Ground truth data
                        plot_training_data: bool = True,  # Whether to plot observed data
                        ):
        '''
        time_snaps: the time axis of the observed data shape = (r, t)
        time: the time axis of the results shape = (r, t1)
        snapshots: the observed data compressed or not, shape = (r, t)
        samples: the results of many draws from a model, shape = (n, r, t1)
        title_x: horizontal position of title (0=left, 0.5=center, 1=right)
        title_y: vertical position of title (0=bottom, 1=top)
        plot_all_samples: if True, plot all samples; if False, plot mean with confidence intervals
        confidence_level: confidence level for intervals (e.g., 0.95 for 95% CI)
        truth_time: time axis for ground truth data, shape = (r, t2) or (t2,)
        truth_data: ground truth data, shape = (r, t2)
        plot_training_data: if True, plot the training/observed data
        
        Plotting combinations:
        - Training domain: training data + predictions (truth_time=None, truth_data=None)
        - Training domain with truth: training data + predictions + truncated truth
        - Test domain: training data + predictions + truth (full truth domain)
        - Test domain without training: predictions + truth (plot_training_data=False)
        '''
        modes = samples.shape[1]
        ndraws = samples.shape[0]
        print(f"Modes: {modes}, Number of draws: {ndraws}")
        
        # Determine if we have ground truth data
        has_truth = truth_time is not None and truth_data is not None
        
        fig, ax = plt.subplots(modes, 1, figsize=figsize, sharex=True)
        
        if modes == 1:
            ax = [ax]
        
        # Set x-axis label on the bottom subplot only (due to sharex=True)
        ax[-1].set_xlabel(xtitle)
        
        # Set y-axis label if provided
        if ytitle is not None:
            # Set ylabel on the middle subplot for better positioning
            middle_idx = modes // 2
            ax[middle_idx].set_ylabel(ytitle)
        
        # Plot ground truth data first (so it appears behind other plots)
        if has_truth:
            for i in range(modes):
                if len(truth_time.shape) == 1:  # 1d case
                    ax[i].plot(truth_time, truth_data[i], color='gray', linewidth=2,
                              label='Ground truth' if i == 0 else "", alpha=0.8)
                else:  # Handle multi-dimensional time if needed
                    ax[i].plot(truth_time[i], truth_data[i], color='gray', linewidth=2,
                              label='Ground truth' if i == 0 else "", alpha=0.8)
        
        # Plot samples based on the chosen method
        if plot_all_samples:
            # Plot all sample trajectories (semi-transparent lines)
            for i in range(ndraws):
                for j in range(modes):
                    if len(time.shape) == 1:  # 1d case
                        ax[j].plot(time, samples[i, j], alpha=0.2, color='tab:blue',
                                  label='Model samples' if i == 0 and j == 0 else "")
                    else:  # Handle multi-dimensional time if needed
                        ax[j].plot(time[j], samples[i, j], alpha=0.2, color='tab:blue',
                                  label='Model samples' if i == 0 and j == 0 else "")
        else:
            # Plot mean with confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            for j in range(modes):
                # Calculate mean and percentiles across samples
                mean_trajectory = np.mean(samples[:, j, :], axis=0)
                lower_bound = np.percentile(samples[:, j, :], lower_percentile, axis=0)
                upper_bound = np.percentile(samples[:, j, :], upper_percentile, axis=0)
                
                if len(time.shape) == 1:  # 1d case
                    time_axis = time
                else:  # Handle multi-dimensional time if needed
                    time_axis = time[j]
                
                # Plot mean line
                ax[j].plot(time_axis, mean_trajectory, color='tab:blue', linewidth=2,
                          label='Mean trajectory' if j == 0 else "")
                
                # Plot confidence interval as filled area
                ax[j].fill_between(time_axis, lower_bound, upper_bound, 
                                  color='tab:blue', alpha=0.3,
                                  label=f'{confidence_level*100:.0f}% CI' if j == 0 else "")
        
        # Plot the observed/training data (black stars)
        if plot_training_data:
            for i in range(modes):
                if len(time_snaps.shape) == 1:  # 1d case
                    ax[i].plot(time_snaps, snapshots[i], 'k*', markersize=8,
                              label='Training data' if i == 0 else "")
                else:  # Handle multi-dimensional time_snaps if needed
                    ax[i].plot(time_snaps[i], snapshots[i], 'k*', markersize=8,
                              label='Training data' if i == 0 else "")
        
        # Add legend to the first subplot
        ax[0].legend(loc='upper right')
        
        # Add grid and styling
        for i in range(modes):
            if grid:
                ax[i].grid(True, alpha=0.3)
            if shade != 0:
                ax[i].axvspan(0, shade, alpha=0.1, color="gray")
            
            # Set x-axis limits based on available data
            all_times = []
            
            # Always include prediction time domain
            if len(time.shape) == 1:
                all_times.extend([min(time), max(time)])
            else:
                all_times.extend([min(time[i]), max(time[i])])
            
            # Include training data time domain if being plotted
            if plot_training_data:
                if len(time_snaps.shape) == 1:
                    all_times.extend([min(time_snaps), max(time_snaps)])
                else:
                    all_times.extend([min(time_snaps[i]), max(time_snaps[i])])
            
            # Include ground truth time domain if available (this extends to test domain)
            if has_truth:
                if len(truth_time.shape) == 1:
                    all_times.extend([min(truth_time), max(truth_time)])
                else:
                    all_times.extend([min(truth_time[i]), max(truth_time[i])])
            
            ax[i].set_xlim(min(all_times), max(all_times))
            # ax[i].set_title(f'Mode {i+1}', fontsize=10)
        
        # First do tight_layout to get proper spacing
        plt.tight_layout()
        
        # Then add the title with proper positioning
        fig.suptitle(title, x=title_x, y=title_y, fontsize=14)
        
        # Adjust the top margin to accommodate the main title
        # Use a more conservative adjustment based on title_y
        top_margin = title_y - 0.03
        plt.subplots_adjust(top=top_margin)
        
        plt.show()