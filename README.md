# photon-doppler-velocimetry-data-analysis
Webapp built with Bokeh and Python for analysis of Photon Doppler Velocimetry (PDV) data.

# Installation

The following python packages are required:

* bokeh (tested with 0.12.7, but may work with later versions)
* numpy
* scipy
* librosa
* wavelets (https://github.com/aaren/wavelets)
* bottle (for the file download/upload functionality)

Install these using pip or your preferred method.

# Usage

`bokeh serve pdvAnalysisBokehApp`

The UI is divided via tabs into 4 different windows.

## Raw Data Tab

Data files can be uploaded here (csv only). "Load File" shows a preview of the file structure to the right, and the starting column/row of the
data can be updated accordingly to ensure proper import. Once done, "Plot Data" shows the imported data in the main plot. Pan/zoom to select
the data range of interest (scroll wheel on the horizontal/vertical axis will zoom in one dimension only), which will then propagate this selected 
region of the data to the other analysis tabs. Optionally, set the velocity breakout point with the "Find Velocity Breakout" button,
and fine adjustment slider. This will define a global starting point for all analysis tabs, where the velocity at any earlier times is set to zero.

## Zero Crossing Tab

This tab allows for the manual calculation of velocity by peak/vally/zero-crossing tracking. The analysis steps are as follows:

1) "Fit Spline Curve" fits a 4th order smoothing spline to the raw data. The smoothing width parameter of the spline can be adjusted, where the
optimal fit minimizes both residuals and the overall "smoothness" of the spline. That is, a spline curve that matches the raw data well **and** has
the minimum number of inflection points. Once a satisfactory fit is found, the "Find Peaks" button will identify the peaks/valleys/inflection points 
and label them with blue dots. It is likely that there will be at least a few spurious points; these can be manually removed with the "lasso select" 
tool on the plot. This process requires user judgement, as removal of a true peak/valley/inflection point will change the calculated velocity substantially.

2) Once only appropriate points remain, "Extract Velocity" will use the selected period intervals (0.25 up to 1.5) to calculate velocity. Intervals can 
be deselected to remove the associated points from the velocity curve. If desired, a guassian smooth of increasing width can be applied, using the labeled slider
to adjust the degree of smoothing.

## Continuous Wavelet Transform Tab

This tab performs a continous wavelet transform on the raw data, producing a time-frequency "scalogram" (which is very similar to the spectrogram from a STFT).
There are currently two adjustable parameters:

**Wavelet Width:** This parameter controls the width of the wavelet, with lower widths generally increasing time-resolution at the cost of frequency (i.e.,
velocity) resolution. The default value is 3, which is typically appropriate for most data. Values as low as 2 can be useful for certain data, especially 
when there are rapid velocity changes.

**Wavelet Frequency Step:** This parameter controls the number of frequency "steps" (specifically, the number of wavelet scales) along the vertical axis, which 
gives increasing frequency (velocity) resolution with decreasing step size. Note that the frequency scale is non-linear, with wider frequency bins at higher
frequencies.

After calculating the wavelet transform, the corresponding velocity is extracted ("Extract Velocity" button)
 by finding the highest magnitude frequency bin from the scalogram for 
each timestep. By using the "lasso" tool on the scalogram plot, an arbitrary-shape can be selected in order to focus the velocity extraction algorithm 
on a particular region. This is useful if there are unwanted frequency modes or baseline noise that negatively effects the extracted velocity profile.

After the desired continuous wavelet transform parameters are set and the velocity is extracted, the final velocity profile can be downloaded as pairs of 
velocity and time points in a CSV file with the "Download Velocity Trace" button.

## Short Time Fourier Transform Tab

This tab performs the short time fourier transform (STFT) on the raw data, producting a time-frequency spectrogram. There are three adjustable parameters.

**Window Width:** This parameter controls with width (in time steps) of the sliding window (currently a Hamming window) using the STFT.
Powers of 2 will be for the width are computationally efficient, but any size width can be used. High quality data can use widths of 128 (or even smaller),
but typically data will be optical with windows ranging between 128 and 512 time steps.

**Frequency Bins:** This parameter controls the number of frequency bins in the STFT. It is important to note that increasing the number of frequency bins
performs interpolation in frequency space, so very large bin numbers will make the extracted velocity smoother but not neccessarily more accurate.

**Time Step:** This parameter controls the stepsize of the moving window in the STFT. The default value is one, meaning the window slides a single time 
interval (i.e., 1/sample rate). For large datasets where long time lengths are desired, increasing the time step can make the STFT computation and velocity
extraction more efficient, though at the cost of decreased time-resolution.

After calculating the STFT, the corresponding velocity is extracted ("Extract Velocity" button)
 by finding the highest magnitude frequency bin from the spectrogram for 
each timestep. By using the "lasso" tool on the spectrogram plot, an arbitrary-shape can be selected in order to focus the velocity extraction algorithm 
on a particular region. This is useful if there are unwanted frequency modes or baseline noise that negatively effects the extracted velocity profile.

After the desired STFT parameters are set and the velocity is extracted, the final velocity profile can be downloaded as pairs of 
velocity and time points in a CSV file with the "Download Velocity Trace" button.

## Velocity Comparison Tab

This final tab simple compares the extracted velocity curves from the three methods (zero crossing, continuous wavelet transform, and short time fourier transform)
on a single plot.


