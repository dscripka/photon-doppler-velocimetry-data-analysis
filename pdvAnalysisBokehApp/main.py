# Bokeh Web app for analysis of PDV data.
# Launch by running "bokeh serve <app folder>", and navigating to http://localhost:5006/ with your web browser

###################
# Import modules
###################

import sys
from os.path import dirname, join
import tempfile
import numpy as np
import scipy as sp
import scipy.ndimage.filters
from scipy.interpolate import UnivariateSpline
from wavelets import WaveletAnalysis, Morlet
import socket

import librosa

from bokeh.layouts import layout, column, row, widgetbox, gridplot
from bokeh.models import Button, Div, ColumnDataSource, TextInput, Span, Range1d
from bokeh.models.widgets import Panel, Tabs, Slider, CheckboxButtonGroup
from bokeh.models.callbacks import CustomJS
from bokeh import events
from bokeh.plotting import figure, curdoc
from bokeh.palettes import viridis
from bokeh.events import SelectionGeometry

import ipdb

import matplotlib.path

##

test_mode = False

##############################################
# Helper functions
##############################################

def round_sig(x, sig=2):
	if x != 0.0:
		return round(x, sig-int(np.floor(np.log10(np.abs(x))))-1)
	if x == 0.0:
		return 0.0

##############################################
# Data load and inspection tab for raw data
##############################################

# Define global tools for all bokeh plots

plotTools = "lasso_select,reset,pan,wheel_zoom,box_select,save,hover"

# create a plot and style its properties
p_rawData = figure(title="Raw Data", x_axis_label='Time (s)', y_axis_label='Voltage (V)', 
           lod_threshold=None, plot_width=1200, plot_height=450, active_scroll="wheel_zoom", tools=plotTools)

# Add plots and data sources

ds = ColumnDataSource(dict(x=[], y=[]))
ds_analysis = ColumnDataSource(dict(x=[], y=[]))
p_rawData_line = p_rawData.line(x='x', y='y', source=ds, line_width=1)

# Define global raw data
global X
X = np.array([])
global Y
Y = np.array([])

sampling_dt = 0
breakoutPt = 0

# Define max points to to show on any given plot
maxPts = 5000

def plot_data():
    # Get file read parameters
    startingRow = int(startRow.value)
    startingCol = int(startCol.value)

    # Read in data
    dat = np.genfromtxt(join(tempfile.gettempdir(), "dataFile"), delimiter=',', skip_header=startingRow)

    global X
    #X = np.arange(0,1000e-9,40e-12)
    X = dat[:,startingCol-1]

    global Y
    #Y = np.sin(X/1e-9)
    Y = dat[:,startingCol]

    global sampling_dt
    sampling_dt = X[1] - X[0]

    # Update plot data source
    spc = int(((X[-1] - X[0])/maxPts)/sampling_dt)
    #print(spc)
    x = X[np.arange(0, X.size, spc)]
    y = Y[np.arange(0, X.size, spc)]
    ds.data = dict(x=x,y=y)

    # Change button color to green
    button2.button_type = "success"

def downsample(attr, old, new):
    
    # Access global raw data variables
    global X
    global Y
    global sampling_dt

    # Get current range min and max
    lwr, upr = p_rawData.x_range.start, p_rawData.x_range.end

    # Calculate spacing for max points
    if (upr-lwr)/sampling_dt > maxPts:
        spc = int(((upr-lwr)/maxPts)/sampling_dt)
        
        x = X[np.arange(0, X.size, spc)]
        y = Y[np.arange(0, X.size, spc)]
        ds.data = dict(x=x,y=y)

    if (upr-lwr)/sampling_dt < maxPts and len(ds.data['x']) < maxPts:
        #print(len(ds.data['x']))
        x = X
        y = Y
        ds.data = dict(x=x,y=y)

def UpdatePlots():
	try:
		# Update all plots to represent current data selection, breakout points, etc.

		if X.size > 0:
			# Get current range min and max on raw data tab
			lwr, upr = p_rawData.x_range.start, p_rawData.x_range.end

			# Update data sources on secondary plots on other tabs to the current visible data on the main tab (raw data)
			x = X[(np.abs(X - lwr)).argmin():(np.abs(X - upr)).argmin()]
			y = Y[(np.abs(X - lwr)).argmin():(np.abs(X - upr)).argmin()]
			ds_analysis.data = dict(x=x, y=y)

		if breakoutPt != 0 and wvt_ds_velocity.data['velocity'] != []:
			# Update all velocity plots to incorporate defined breakout point
			#ipdb.set_trace()
			wv_v = np.array(wvt_ds_velocity.data['velocity'])
			ndx = (np.abs(wvt_ds_velocity.data['x'] - ds_analysis.data['x'][breakoutPt])).argmin()
			wv_v[0:ndx] = 0
			wvt_ds_velocity.data['velocity'] = wv_v

		if breakoutPt != 0 and fft_ds_velocity.data['velocity'] != []:
			fft_v = np.array(fft_ds_velocity.data['velocity'])
			ndx = (np.abs(fft_ds_velocity.data['x'] - ds_analysis.data['x'][breakoutPt])).argmin()
			fft_v[0:ndx] = 0
			fft_ds_velocity.data['velocity'] = fft_v
	except TypeError:
		pass

# Define callbacks for zooming and panning on primary plot

p_rawData.x_range.on_change('end', downsample)

# Create customJS to get HTML5 file input information

# Get host's IP address
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
host_ip = s.getsockname()[0]
s.close()
server_info = ColumnDataSource(dict(server_ip=[host_ip]))

js_callback = CustomJS(args=dict(server_ip=server_info), code=open(join(dirname(__file__), "readFile.js")).read())

# add a button widget and configure with the call back
button = Button(label="Load File", id="uploadButton", width=300, button_type="warning")
button2 = Button(label="Plot Data", width=300, button_type="warning")
button2.on_click(plot_data)

button.js_on_event(events.ButtonClick, js_callback)

# Create html input in Div widget

fileInput = Div(text="""
<h1>Select File</h1>
<input type="file" id="upload">
""", width=350)

filePreview = Div(text = '<h1 id="fileContentsTitle"> File Preview </h1> <div id="fileContents">x,y</div>', id="filePreview", width=600)
fileReadError = Div(text='', width=600)
startCol = TextInput(value="1", title="Data Starting Column:")
startRow = TextInput(value="1", title="Data Starting Row:")

# Define vertical line marking position of velocity breakout

def SetBreakout(attr, old, new):
	# Find index of breakout time
	ndx = (np.abs(ds_analysis.data['x']-breakoutSlider.value*1e-9)).argmin()

	global breakoutPt
	breakoutPt = ndx

	# Update position of vertical line marking velocity breakout
	p_breakoutLine.location = breakoutSlider.value*1e-9
	#y_spline = ds_splinefit.data['y_spline']
	#y_spline[0:ndx] = 0
	#ds_splinefit.data['y_spline'] = y_spline

breakoutSlider = Slider(title = "Velocity breakout position (ns)",
			value = 0,
			start = 0,
			end = 10,
			step= sampling_dt)

breakoutSlider.on_change('value', SetBreakout)

p_breakoutLine = Span(location=0, dimension='height', line_color='green', line_width=1)
p_rawData.add_layout(p_breakoutLine)

def DetectBreakout():
	detectBreakout.button_type = "warning"
	# Find velocity breakout
	x = ds_analysis.data['x']
	y = ds_analysis.data['y']
	ySpline = ds_analysis.data['y']
	dt = np.abs(np.abs(ds_analysis.data['x'][1]) - np.abs(ds_analysis.data['x'][0]))
	yMean = np.mean(y[0:30])
	ySd = np.std(y[0:30])
	for ndx,val in enumerate(y):
		if abs(val) > yMean + 5*ySd:  # > 99.999% confidence interval for value outside of baseline signal
			#ipdb.set_trace()
			breakoutPoint = x[ndx]

			# Set breakout line on plot and update slider values
			p_breakoutLine.location = breakoutPoint
			breakoutSlider.value = breakoutPoint/1e-9
			breakoutSlider.start = p_rawData.x_range.start/1e-9
			breakoutSlider.end = p_rawData.x_range.end/1e-9
			breakoutSlider.step = dt/1e-9
			detectBreakout.button_type = "success"
			break

detectBreakout = Button(label="Find Velocity Breakout", button_type="success")
detectBreakout.on_click(DetectBreakout)

# If in testing mode, load default data

if test_mode:
    # Generate sample data
    t0 = 0
    t1 = 25e-9
    x = np.arange(t0, t1, 40e-12)
    x_overs = np.arange(t0, t1, 4e-12)

    f = 900e6 / (1 + np.exp(-2.0e9 * x + 10))

    for i in range(1, len(x)):
        if x[i] > 8.5e-9:
            f[i] = f[i - 1] * .9995
        if x[i] > 9.5e-9:
            f[i] = f[i - 1] * .997

    phi = 2 * np.pi * np.cumsum(f) * 40e-12

    y = np.sin(phi) * np.cos(x * 35e6) - 10e6 * x + np.random.uniform(size=len(x)) * 0.05

    #global X
    X = x
    #global Y
    Y = y

    #global sampling_dt
    sampling_dt = X[1] - X[0]

    # Update plot data source
    spc = int(np.ceil(((X[-1] - X[0]) / maxPts) / sampling_dt))
    # print(spc)
    #ipdb.set_trace()
    x = X[np.arange(0, X.size, spc)]
    y = Y[np.arange(0, X.size, spc)]
    ds.data = dict(x=x, y=y)


############################################
# Zero Crossing Analysis
############################################

p_zeroCrossing = figure(title="Zero Crossing Analysis", x_axis_label='Time (s)', y_axis_label='Voltage',
                        lod_threshold=None, plot_width=800, plot_height=500, responsive=True, active_scroll="wheel_zoom", tools=plotTools)
p_zeroCrossing.min_border_left=70

p_zeroCrossing_velocity = figure(title="Velocity", x_axis_label='Time (s)', y_axis_label='Velocity (m/s)',
                                 lod_threshold=None, plot_width=800, plot_height=500, responsive=True, active_scroll="wheel_zoom", tools=plotTools)
p_zeroCrossing_velocity.min_border_left=70

ds_splinefit = ColumnDataSource(dict(x=[], y=[], y_spline=[]))
ds_splinefit_peaks = ColumnDataSource(dict(x=[], y=[]))
ds_zeroCrossing_velocity = ColumnDataSource(dict(x=[], velocity=[], smoothed=[]))
p_zeroCrossing_line = p_zeroCrossing.line(x='x', y='y', source=ds_splinefit, line_alpha=0.5)
p_zeroCrossing_peaks = p_zeroCrossing.scatter(x='x', y = 'y', source=ds_splinefit_peaks, fill_color="blue")
p_zeroCrossing_velocity_points = p_zeroCrossing_velocity.circle(x='x', y='velocity', source=ds_zeroCrossing_velocity, fill_color="blue")
p_zeroCrossing_velocity_fit_line = p_zeroCrossing_velocity.line(x='x', y='smoothed', source=ds_zeroCrossing_velocity, line_color="red")


def FitSpline(attr, old, new):
    x = ds_analysis.data['x']
    y = ds_analysis.data['y']
    spl = UnivariateSpline(ds_analysis.data['x'], ds_analysis.data['y'], k=4, s=splineSlider.value)
    ds_splinefit.data = dict(x=x,y=y,y_spline=spl(x))

    # Function to find peaks/valleys using first derivative of spline fit

def GetPeaks():
    x = ds_analysis.data['x']
    y = ds_analysis.data['y']
    spl = UnivariateSpline(x, y, k=4, s=splineSlider.value)
    spl_prime = spl.derivative()
    spl_prime_vals = spl_prime(x)
    spl_prime2 = spl_prime.derivative()

    peaks_valleys = spl_prime.roots()
    peaks_valleys_ndx = [np.abs(x - i).argmin() for i in peaks_valleys]

    inflection_points_ndx = []
    for i in range(1, len(spl_prime_vals)-1):
        if spl_prime_vals[i] < spl_prime_vals[i-1] and spl_prime_vals[i] < spl_prime_vals[i+1]:
            inflection_points_ndx.append(i)
        elif spl_prime_vals[i] > spl_prime_vals[i-1] and spl_prime_vals[i] > spl_prime_vals[i+1]:
            inflection_points_ndx.append(i)

    ds_splinefit_peaks.data = dict(x=np.concatenate((peaks_valleys, x[inflection_points_ndx])),
                                   y=spl(x)[np.concatenate((peaks_valleys_ndx, inflection_points_ndx))])

def CalculateVelocity():
    pts = ds_splinefit_peaks.data['x']
    pts.sort()

    # Get 1/4 periods
    fs0 = [1 / (4 * (pts[i + 1] - pts[i])) for i in range(0, len(pts) - 1)]
    ts0 = [(pts[i + 1] - pts[i]) / 2 + pts[i] for i in range(0, len(pts) - 1)]

    # Get 1/2 periods
    fs1 = [1 / (2 * (pts[i + 2] - pts[i])) for i in range(0, len(pts) - 2)]
    ts1 = [(pts[i + 2] - pts[i]) / 2 + pts[i] for i in range(0, len(pts) - 2)]

    # Get lead 3/4 periods
    fs2 = [3 / (4 * (pts[i + 3] - pts[i])) for i in range(0, len(pts) - 3)]
    ts2 = [(pts[i + 3] - pts[i]) / 2 + pts[i] for i in range(0, len(pts) - 3)]

    # Get full periods
    fs3 = [1 / (pts[i + 4] - pts[i]) for i in range(0, len(pts) - 4)]
    ts3 = [(pts[i + 4] - pts[i]) / 2 + pts[i] for i in range(0, len(pts) - 4)]

    # Get 1 and 1/4 periods
    fs4 = [5 / (4 * (pts[i + 5] - pts[i])) for i in range(0, len(pts) - 5)]
    ts4 = [(pts[i + 5] - pts[i]) / 2 +pts[i] for i in range(0, len(pts) - 5)]

    # Get 1 and 1/2 periods
    fs5 = [6 / (4 * (pts[i + 6] - pts[i])) for i in range(0, len(pts) - 6)]
    ts5 = [(pts[i + 6] - pts[i]) / 2 + pts[i] for i in range(0, len(pts) - 6)]

    # Add velocity points based on checkbox values for period fractions
    t = np.concatenate(np.array((ts0, ts1, ts2, ts3, ts4, ts5))[[periodFractions.active]])
    vel = np.concatenate(np.array((fs0, fs1, fs2, fs3, fs4, fs5))[[periodFractions.active]])*1550e-9/2.0

	# Sort time and velocity points
    pts = [(i,j) for i,j in zip(t,vel)]
    pts.sort(key=lambda x: x[0])
    t = [ds_analysis.data['x'][breakoutPt]] + [i[0] for i in pts]
    vel = [0.0] + [i[1] for i in pts]

    ds_zeroCrossing_velocity.data = dict(x=t, velocity=vel, smoothed=vel)

extractVelocityZC = Button(label="Extract Velocity", button_type="success")
extractVelocityZC.on_click(CalculateVelocity)

periodFractions = CheckboxButtonGroup(labels=["1/4 Period", "1/2 Period", "3/4 Period", "1 Period", "4/3 Period", "3/2 Period"], active=[0,0,0,1,0,0], name="Period Fractions", width=150)

downloadVelocityTraceZC = Button(label="Download Velocity Trace", button_type="success")
downloadVelocityTraceZC.callback = CustomJS(args=dict(vel_source=ds_zeroCrossing_velocity, base_source = ds_splinefit_peaks),
                           code=open(join(dirname(__file__), "download.js")).read())

def FitVelocity(attr, old, new):
	# Perform a spline fit to the velocity data
	x = ds_zeroCrossing_velocity.data['x'][1:]
	y = ds_zeroCrossing_velocity.data['velocity']

	gSmooth = sp.ndimage.filters.gaussian_filter1d(y, velSmoothSlider.value)
	ds_zeroCrossing_velocity.data = dict(x=[ds_analysis.data['x'][breakoutPt]] + x, velocity=y, smoothed=[0.0] + list(gSmooth))
	#ipdb.set_trace()

#extractVelocitySplineFit = Button(label="Smooth Velocity Data", button_type="success")
#extractVelocitySplineFit.on_click(FitVelocity)

velSmoothSlider = Slider(title = "Gaussian Smooth Width",
			value = 0,
			start = 0,
			end = 9,
			step= 0.1)
velSmoothSlider.on_change('value', FitVelocity)

def InitFitSpline():
    fitSpline.button_type = "warning"
    dataSamp = ds_analysis.data['y']
    gSmooth = sp.ndimage.filters.gaussian_filter1d(dataSamp, 7)
    init_s = np.sum((dataSamp - gSmooth)**2)
    splineSlider.value = init_s
    splineSlider.start = init_s/500
    splineSlider.end = init_s*2
    splineSlider.step = (splineSlider.end - splineSlider.start)/400
    #print(init_s)
    p_zeroCrossing_spline = p_zeroCrossing.line(x='x', y='y_spline', source=ds_splinefit, line_color="red", line_width=1)
    fitSpline.button_type = "success"

fitSpline = Button(label="Fit Spline Curve", button_type="success")
fitSpline.on_click(InitFitSpline)

findPeaks = Button(label="Find Peaks", button_type="success")
findPeaks.on_click(GetPeaks)

splineSlider = Slider(title = "Spline width parameter",
		      value = 0,
                      start = 0,
                      end = 1,
                      step = .1)

splineSlider.on_change('value', FitSpline)

# Manage removal of erroneous peak/inflection points

def SelectionUpdateZC(event):
    if event.final:
        #ipdb.set_trace()
        selectedPoints = ds_splinefit_peaks.selected['1d']['indices']
        new_x = [val for ndx, val in enumerate(ds_splinefit_peaks.data['x']) if ndx not in selectedPoints]
        new_y = [val for ndx, val in enumerate(ds_splinefit_peaks.data['y']) if ndx not in selectedPoints]
        ds_splinefit_peaks.data = dict(x=new_x, y=new_y)

		# Clear selected object to update plot
        ds_splinefit_peaks.selected = {'0d': {'glyph': None, 'get_view': {}, 'indices': []}, '1d': {'indices': []}, '2d': {'indices': {}}}

p_zeroCrossing.on_event(SelectionGeometry, SelectionUpdateZC)

###########################
# Wavelet Analysis
###########################

# Callback Function to perform continous wavelet transform
def cwt(attr, old, new):
	calculateCWT.button_type = "warning"
	# Perform wavelet transform
	tStep = np.int(wvtTimestep.value)
	yWvt = np.array(ds_analysis.data['y'])[0::tStep]
	wa = WaveletAnalysis(yWvt, wavelet=Morlet(w0=np.float(wvtWidth.value)), dt=sampling_dt*tStep, dj=np.float(wvtFreqstep.value))
	wvt = wa.time
	wvfreq = wa.fourier_frequencies
	wvpower = np.flipud(wa.wavelet_power)
	wvt_ds.data = {'image': [wvpower], 'dw': [wvpower.shape[1]], 'dh': [wvpower.shape[0]], 'wv_time': [wvt], 'wv_freq': [wvfreq]}
	p_wavelet.x_range.end = wvpower.shape[1]  # need to set ranges before creating image, and not using Range1d?
	p_wavelet.y_range.end = wvpower.shape[0]  # need to set ranges before creating image, and not using Range1d?

	calculateCWT.button_type = "success"
	#ipdb.set_trace()
	#print(wvt_ds.data)

# Callback function for initial plotting of wavelet transform spectrogram
def Plot_cwt():
	calculateCWT.button_type = "warning"
	# Perform initial wavelet transform
	tStep = np.int(wvtTimestep.value)
	yWvt = np.array(ds_analysis.data['y'])[0::tStep]
	wa = WaveletAnalysis(yWvt, wavelet=Morlet(w0=np.float(wvtWidth.value)), dt=sampling_dt*tStep, dj=np.float(wvtFreqstep.value))
	wvt = wa.time
	wvfreq = wa.fourier_frequencies
	wvpower = np.flipud(wa.wavelet_power)

	# Plot wavelet transform as image
	wvt_ds.data = {'image': [wvpower], 'dw': [wvpower.shape[1]], 'dh': [wvpower.shape[0]], 'wv_time': [wvt], 'wv_freq': [wvfreq]}
	p_wavelet.x_range.end = wvpower.shape[1]  # need to set ranges before creating image, and not using Range1d?
	p_wavelet.y_range.end = wvpower.shape[0]  # need to set ranges before creating image, and not using Range1d?

	newLabels = {}
	for i in range(p_wavelet.x_range.start, p_wavelet.x_range.end):
		newLabels[i] = str(round_sig(wvt[i], 2))
	p_wavelet.xaxis.major_label_overrides = newLabels
	#ipdb.set_trace() 

	newLabelsY = {}
	for i in range(p_wavelet.y_range.start, p_wavelet.y_range.end):
	    newLabelsY[i] = str(np.round(np.array(np.flipud(wvfreq))[i]*1550.0e-9/2.0,1))
	p_wavelet.yaxis.major_label_overrides = newLabelsY

	p_wavelet.image(image='image', source=wvt_ds, x=0, y=0, dw='dw', dh='dh', palette=viridis(200))
	calculateCWT.button_type = "success"


# Callback function to extract velocity from wavelet spectrogram

def ExtractVelocityWVT():
	extractVelocityWVT.button_type = "warning"
	# Get current power spectrum
	wvpower = wvt_ds.data['image'][0]
	freqs = np.flipud(np.array(wvt_ds.data['wv_freq'][0]))

	# Extract velocity from vertical lineouts
	wvvpks = [np.argmax(wvpower[:,i]) for i in range(0,wvpower.shape[1])]
	wvvpks = np.zeros(wvpower.shape[1], dtype=np.int16)
	wvvpks[-1] = np.argmax(wvpower[:,-1])

	for i in np.arange(wvpower.shape[1]-2,0,-1):
		coli = np.copy(wvpower[:,i])
		pk = np.argmax(coli)
		if pk - wvvpks[i+1] > 5000:
			while abs(pk - wvvpks[i+1]) > 5000:
			    coli[pk] = 0
			    pk = np.argmax(coli)
			wvvpks[i] = pk
		else:
			wvvpks[i] = np.argmax(coli)
	extractVelocityWVT.button_type = "success"

	# Update velocity plot
	
	# wvt_ds_velocity.data['x'] = ds_analysis.data['x']
	# wvt_ds_velocity.data['velocity'] = [freqs[i]*1550.0e-9/2 for i in wvvpks]
	# wvt_ds_velocity.data['parameters'] = [np.float(wvtWidth.value), np.float(wvtFreqstep.value)]
	tStep = np.int(wvtTimestep.value)
	xWvtVel = np.array(ds_analysis.data['x'])[0::tStep]
	wvt_ds_velocity.data = dict(x=xWvtVel, velocity=[freqs[i]*1550.0e-9/2 for i in wvvpks])
	#print(wvvpks)
	#ipdb.set_trace()
	
# Add UI elements
calculateCWT = Button(label="Calculate Wavelet Transform", button_type="success")
calculateCWT.on_click(Plot_cwt)

wvtWidth = TextInput(value="3.0", title="Wavelet Width")
wvtFreqstep = TextInput(value="0.15", title="Wavelet Frequency Step")
wvtTimestep = TextInput(value="1", title="Wavelet Time Step (samples)")

wvtWidth.on_change('value', cwt)
wvtFreqstep.on_change('value', cwt)
wvtTimestep.on_change('value', cwt)

extractVelocityWVT = Button(label="Extract Velocity", button_type="success")
extractVelocityWVT.on_click(ExtractVelocityWVT)

# Create figure for spectrogram
wvt_ds = ColumnDataSource(data = dict(image=[], dw=[10], dh=[10]))
p_wavelet = figure(title="Wavelet Transform Spectrogram", tools=plotTools, x_axis_label = 'Time (s)', y_axis_label = 'Velocity (m/s)',
	    active_scroll="wheel_zoom", x_range=(0,10), y_range=(0,10), width=800, height=500, responsive=True)  # need to set ranges initially for updates to work?
p_wavelet.min_border_left=70

# Create figure for velocity extraction

wvt_ds_velocity = ColumnDataSource(data = dict(x=[], velocity = [], parameters = []))
p_wavelet_velocity = figure(title="Extracted Velocity", x_axis_label='Time (s)', y_axis_label='Velocity (m/s)',
           					lod_threshold=None, plot_width=800, plot_height=500, responsive=True, active_scroll="wheel_zoom", tools=plotTools)
p_wavelet_velocity.min_border_left=70
p_wavelet_velocity.line(x='x', y='velocity', source=wvt_ds_velocity, line_width=1)

downloadVelocityTraceWVT = Button(label="Download Velocity Trace", button_type="success")
downloadVelocityTraceWVT.callback = CustomJS(args=dict(vel_source=wvt_ds_velocity, base_source = wvt_ds),
                           code=open(join(dirname(__file__), "download.js")).read())

# Get selected region

def SelectionUpdateWVT(event):
	if event.final:
		calculateCWT.button_type = "warning"
		# Get x and y points of selection region
		lassoRegionX = event.geometry['x']
		lassoRegionY = event.geometry['y']

		# Get current spectrogram data
		wvpower = wvt_ds.data['image'][0]

		# Find points in wavelet power spectrogram inside selection region
		X, Y = np.meshgrid(np.arange(wvpower.shape[1]), np.arange(wvpower.shape[0]))
		powerPts = np.vstack((Y.flatten(),X.flatten())).T
		polygon = [(int(row),int(col)) for row,col in zip(lassoRegionY, lassoRegionX)]  #flip X and Y for row/col orientation
		path = matplotlib.path.Path(polygon)
		selected = path.contains_points(powerPts)
		selected = selected.reshape((wvpower.shape[0], wvpower.shape[1]))

		# Update image data source
		#wvt_ds.data = {'image': [np.multiply(selected, wvpower)], 'dw': [wvpower.shape[1]], 'dh': [wvpower.shape[0]]}
		wvt_ds.data['image'] = [np.multiply(selected, wvpower)]
		calculateCWT.button_type = "success"

#p_wavelet.tool_events.on_change('geometries', SelectionUpdateWVT) 
#wvt_ds.on_change('selected', SelectionUpdate)
p_wavelet.on_event(SelectionGeometry, SelectionUpdateWVT)


###########################
# FFT Analysis
###########################

# Callback Function to calculate fft spectrogram
def spectrogram(attr, old, new):
	calculateFFT.button_type = "warning"
	# Perform fft transform to get spectrogram
	yFFT = ds_analysis.data['y']
	t = ds_analysis.data['x']
	
	# freq, t, power = sp.signal.spectrogram(yFFT, fs=1/sampling_dt, window=sp.signal.hamming(np.int(fftWidth.value), sym=False),
	# 									noverlap=np.int(fftWidth.value) - np.int(fftTimestep.value),
	# 				nfft = np.int(fftFreqbins.value), nperseg=np.int(fftWidth.value), mode='psd')

	spec = librosa.core.stft(yFFT, n_fft=np.int(fftFreqbins.value), hop_length=np.int(fftTimestep.value), win_length=np.int(fftWidth.value), window=sp.signal.hamming(np.int(fftWidth.value), sym=False), center=True)
	power = np.abs(spec)**2
	freq = np.linspace(0, 1/sampling_dt/2, np.int(fftFreqbins.value)/2+1)
	print(power.shape)

	# # Pad array to account for edge effects of spectrogram calculation and correct time axis
	# padLeft = np.zeros((power.shape[0], np.int(fftWidth.value)//2-1))
	# padRight = np.zeros((power.shape[0], np.int(fftWidth.value)//2))
	# #print(power.shape, padLeft.shape, padRight.shape)
	# power = np.concatenate((padLeft, power, padRight), axis=1)
	# print(power.shape)

	fft_ds.data = {'image': [power], 'dw': [power.shape[1]], 'dh': [power.shape[0]], 'fft_time': [t[0::np.int(fftTimestep.value)]], 'fft_freq': [freq]}
	p_fft.x_range.end = power.shape[1]  # need to set ranges before creating image, and not using Range1d?
	#p_fft.y_range.end = [np.unravel_index(power.argmax(), power.shape)[0]*2 if power.shape[0] > np.unravel_index(power.argmax(), power.shape)[0]*2 else power.shape[0]][0]  # need to set ranges before creating image, and not using Range1d?
	p_fft.y_range.end = np.int(power.shape[0]/12)  # need to set ranges before creating image, and not using Range1d?

	calculateFFT.button_type = "success"

	# Create new labels for x and y axis (time and velocity, respectively)
	newLabelsX = {}
	#print(p_fft.x_range.start, p_fft.x_range.end)
	#print(len(fft_ds.data['fft_time'][0]))
	for i in range(p_fft.x_range.start, p_fft.x_range.end-1):  #Need to fix this for cases when timestep is > 1
		newLabelsX[i] = str(round_sig(fft_ds.data['fft_time'][0][i], 2))
	p_fft.xaxis.major_label_overrides = newLabelsX

	newLabelsY = {}
	for i in range(p_fft.y_range.start, p_fft.y_range.end):
	    newLabelsY[i] = str(np.round(np.array(freq)[i]*1550.0e-9/2.0,0))
	p_fft.yaxis.major_label_overrides = newLabelsY

# Callback function for initial plotting of fft transform spectrogram
def Plot_FFT():
	calculateFFT.button_type = "warning"
	# Perform initial stft transform
	yFFT = ds_analysis.data['y']
	t = ds_analysis.data['x']
	# freq, t, power = sp.signal.spectrogram(yFFT, fs=1/sampling_dt, window=sp.signal.hamming(np.int(fftWidth.value), sym=False),
    #                                     noverlap=np.int(fftWidth.value) - np.int(fftTimestep.value),
	# 				nfft = np.int(fftFreqbins.value), nperseg=np.int(fftWidth.value), mode='psd')

	spec = librosa.core.stft(yFFT, n_fft=np.int(fftFreqbins.value), hop_length=np.int(fftTimestep.value), win_length=np.int(fftWidth.value), window=sp.signal.hamming(np.int(fftWidth.value), sym=False), center=True)
	power = np.abs(spec)**2
	freq = np.linspace(0, 1/sampling_dt/2, np.int(fftFreqbins.value)/2+1)

	# # Pad array to account for edge effects of spectrogram calculation and correct time axis
	# padLeft = np.zeros((power.shape[0], np.int(fftWidth.value)//2-1))
	# padRight = np.zeros((power.shape[0], np.int(fftWidth.value)//2))
	# power = np.concatenate((padLeft, power, padRight), axis=1)

	# Plot stft transform as image, and set x and y range intelligently based on power spectrum

	fft_ds.data = {'image': [power], 'dw': [power.shape[1]], 'dh': [power.shape[0]], 'fft_time': [t[0::np.int(fftTimestep.value)]], 'fft_freq': [freq]}
	p_fft.x_range.end = power.shape[1]  # need to set ranges before creating image, and not using Range1d?
	#p_fft.y_range.end = [np.unravel_index(power.argmax(), power.shape)[0]*2 if power.shape[0] > np.unravel_index(power.argmax(), power.shape)[0]*2 else power.shape[0]][0]  # need to set ranges before creating image, and not using Range1d?
	p_fft.y_range.end = np.int(power.shape[0]/12)  # need to set ranges before creating image, and not using Range1d?
	#ipdb.set_trace()
	#print(power.shape)

	# Create new labels for x and y axis (time and velocity, respectively)
	newLabelsX = {}
	for i in range(p_fft.x_range.start, p_fft.x_range.end-1):  #Need to fix this for cases when timestep is > 1
	    newLabelsX[i] = str(round_sig(fft_ds.data['fft_time'][0][i], 2))
	p_fft.xaxis.major_label_overrides = newLabelsX

	newLabelsY = {}
	for i in range(p_fft.y_range.start, p_fft.y_range.end):
		newLabelsY[i] = str(np.round(np.array(freq)[i]*1550.0e-9/2.0,0))
	p_fft.yaxis.major_label_overrides = newLabelsY
	#ipdb.set_trace()

	p_fft.image(image='image', source=fft_ds, x=0, y=0, dw='dw', dh='dh', palette=viridis(200))
	calculateFFT.button_type = "success"



# Callback function to extract velocity from wavelet spectrogram

def ExtractVelocityFFT():
	extractVelocityFFT.button_type = "warning"
	# Get current power spectrum
	power = fft_ds.data['image'][0]
	freqs = fft_ds.data['fft_freq'][0]
	t = fft_ds.data['fft_time'][0]

	# Extract velocity from vertical lineouts
	vpks = [np.argmax(power[:,i]) for i in range(0,power.shape[1])]
	vpks = np.zeros(power.shape[1], dtype=np.int16)
	vpks[-1] = np.argmax(power[:,-1])

	for i in np.arange(power.shape[1]-2,0,-1):
		coli = np.copy(power[:,i])
		pk = np.argmax(coli)
		if pk - vpks[i+1] > 5000:
			while abs(pk - vpks[i+1]) > 5000:
			    coli[pk] = 0
			    pk = np.argmax(coli)
			vpks[i] = pk
		else:
			vpks[i] = np.argmax(coli)

	# Update velocity plot
	
	# fft_ds_velocity.data['x'] = ds_analysis.data['x']
	# fft_ds_velocity.data['velocity'] = [freqs[i]*1550.0e-9/2 for i in vpks]
	# fft_ds_velocity.data['parameters'] = [np.float(fftWidth.value), np.float(fftFreqbins.value), np.float(fftTimestep.value)]
	fft_ds_velocity.data = dict(x=t, velocity=[freqs[i]*1550.0e-9/2 for i in vpks][0:-1])
	extractVelocityFFT.button_type = "success"
	#ipdb.set_trace()

# Add UI elements
calculateFFT = Button(label="Calculate STFT", button_type="success")
calculateFFT.on_click(Plot_FFT)

fftWidth = TextInput(value="128", title="Window Width (samples)")
fftFreqbins = TextInput(value="1024", title="Frequency Bins")
fftTimestep = TextInput(value="1", title="Time Step (samples)")

fftWidth.on_change('value', spectrogram)
fftFreqbins.on_change('value', spectrogram)
fftTimestep.on_change('value', spectrogram)

extractVelocityFFT = Button(label="Extract Velocity", button_type="success")
extractVelocityFFT.on_click(ExtractVelocityFFT)	

# Create figure for spectrogram
fft_ds = ColumnDataSource(data = dict(image=[], dw=[len(ds_analysis.data['x'])], dh=[10]))
p_fft = figure(title="Short Time Fourier Transform Spectrogram", tools=plotTools, x_axis_label = 'Time (s)', y_axis_label='Velocity (m/s)',
	    active_scroll="wheel_zoom", x_range=(0,10), y_range=(0,10), width=800, height=500)  # need to set ranges initially for updates to work?
p_fft.min_border_left=70

# Create figure for velocity extraction

fft_ds_velocity = ColumnDataSource(data = dict(x=[], velocity = [], parameters = []))
p_fft_velocity = figure(title="Extracted Velocity", x_axis_label='Time (s)', y_axis_label='Velocity (m/s)', 
           				lod_threshold=None, plot_width=800, plot_height=500, responsive=True, active_scroll="wheel_zoom", tools=plotTools)
p_fft_velocity.min_border_left=70

p_fft_velocity.line(x='x', y='velocity', source=fft_ds_velocity, line_width=1)

downloadVelocityTraceFFT = Button(label="Download Velocity Trace", button_type="success")
downloadVelocityTraceFFT.callback = CustomJS(args=dict(vel_source=fft_ds_velocity, base_source=fft_ds),
                           code=open(join(dirname(__file__), "download.js")).read())

# Get selected region

def SelectionUpdateFFT(event):
	if event.final:
		calculateFFT.button_type = "warning"
		# Get x and y points of selection region
		lassoRegionX = event.geometry['x']
		lassoRegionY = event.geometry['y']

		# Get current spectrogram data
		power = fft_ds.data['image'][0]

		# Find points in wavelet power spectrogram inside selection region
		X, Y = np.meshgrid(np.arange(power.shape[1]), np.arange(power.shape[0]))
		powerPts = np.vstack((Y.flatten(),X.flatten())).T
		polygon = [(int(row),int(col)) for row,col in zip(lassoRegionY, lassoRegionX)]  #flip X and Y for row/col orientation
		path = matplotlib.path.Path(polygon)
		selected = path.contains_points(powerPts)
		selected = selected.reshape((power.shape[0], power.shape[1]))

		# Update image data source
		fft_ds.data['image'] = [np.multiply(selected, power)]
		calculateFFT.button_type = "success"

p_fft.on_event(SelectionGeometry, SelectionUpdateFFT)

##############################
# Velocity comparison tab
##############################

p_velocity_comparison = figure(title="Extracted Velocities", x_axis_label='Time', y_axis_label='Velocity (m/s)', 
           				lod_threshold=None, plot_width=1200, plot_height = 500, responsive=True, active_scroll="wheel_zoom", tools=plotTools)

p_velocity_comparison.line(x='x', y='velocity', source=fft_ds_velocity, line_width=1, legend="STFT", line_color="green")
p_velocity_comparison.line(x='x', y='velocity', source=wvt_ds_velocity, line_width=1, legend="CWT", line_color="red")
p_velocity_comparison.scatter(x='x', y='velocity', source=ds_zeroCrossing_velocity, fill_color="blue", legend='Zero-Crossing')
p_velocity_comparison.line(x='x', y='smoothed', source=ds_zeroCrossing_velocity, line_color="blue", legend='Zero-Crossing (smoothed)')

if test_mode:
    x = np.arange(t0, t1, 40e-12)
    f = 900e6 / (1 + np.exp(-2.0e9 * x + 10))
    for i in range(1, len(x)):
        if x[i] > 8.5e-9:
            f[i] = f[i - 1] * .9995
        if x[i] > 9.5e-9:
            f[i] = f[i - 1] * .997

    p_velocity_comparison.line(x=x, y=f*1550e-9/2.0, line_color="gray", legend='True Velocity')

p_velocity_comparison.legend.location = "top_left"
p_velocity_comparison.legend.click_policy="hide"

# Download all velocities
downloadVelocityTraces = Button(label="Download All Velocity Trace", button_type="success")
downloadVelocityTraces.callback = CustomJS(args=dict(s1=fft_ds_velocity, s2=wvt_ds_velocity, s3=ds_zeroCrossing_velocity),
                           code=open(join(dirname(__file__), "download.js")).read())

#########################
# Define Layouts
#########################

# Raw data tab
rawData_layout = column(p_rawData, row(widgetbox(fileInput, button, startCol, startRow, button2), column(filePreview, fileReadError), column(detectBreakout, breakoutSlider)))
#rawData_layout = layout([[p_rawData], [fileInput, button, startCol, startRow, button2], [filePreview, fileReadError]])
tab_rawData = Panel(child=rawData_layout, title="Raw Data")

# Zero crossing analysis tab
zeroCrossing_layout = row(column(p_zeroCrossing, row(fitSpline, splineSlider), row(findPeaks, )),
                          column(p_zeroCrossing_velocity, row(column(extractVelocityZC, downloadVelocityTraceZC), periodFractions,  velSmoothSlider)))
tab_zeroCrossing = Panel(child=zeroCrossing_layout, title="Zero Crossing")

# Wavelet analysis tab
wavelet_layout = row(column(p_wavelet, calculateCWT, wvtWidth, wvtFreqstep, wvtTimestep), column(p_wavelet_velocity, row(extractVelocityWVT, downloadVelocityTraceWVT)))
#wavelet_layout = row(column(p_wavelet, calculateCWT, wvtWidth, wvtFreqstep,), column(p_wavelet_velocity, row(extractVelocityWVT, downloadVelocityTraceWVT)))
tab_wavelet = Panel(child=wavelet_layout, title="Continuous Wavelet Transform")

# FFT analysis tab
fft_layout = row(column(p_fft, calculateFFT, fftWidth, fftFreqbins, fftTimestep), column(p_fft_velocity, row(extractVelocityFFT, downloadVelocityTraceFFT)))
tab_fft = Panel(child=fft_layout, title="Short Time Fourier Transform")

# Velocity Comparison tab

#velocity_comparison_layout = column(p_velocity_comparison, downloadVelocityTraces)
velocity_comparison_layout = column(p_velocity_comparison)
tab_velocity_comparison = Panel(child=velocity_comparison_layout, title="Velocity Comparisons")

# Final arrangement of tabs
tabs = Tabs(tabs=[tab_rawData, tab_zeroCrossing, tab_wavelet, tab_fft, tab_velocity_comparison])
#tabs = Tabs(tabs=[tab_rawData, tab_zeroCrossing, tab_wavelet])
curdoc().add_root(tabs)
curdoc().add_periodic_callback(UpdatePlots, 1000)

#######################################
# Start bottle server for file upload
#######################################

import subprocess
import os

wd = os.getcwd()
proc = subprocess.Popen(['python', wd + '/pdvAnalysisBokehApp/bottleServer.py'])
