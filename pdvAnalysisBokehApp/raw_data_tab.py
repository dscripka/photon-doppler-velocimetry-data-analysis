# Define global raw data
global X
X = np.array([])
global Y
Y = np.array([])

sampling_dt = 0
breakoutPt = 0

# Define max points to to show on any given plot
maxPts = 5000

def load_raw_data():
	"""
	Loads the raw data from the temporary file created by the javscript function <readFile.js>
	
	Args: None

	Returns:
		list: horizontal (time) axis data
		list: vertical (voltage) axis data
		float: the spacing between the points on the time axis (inverse of the sampling frequency)

	Except:
		IOError: Raised when temporary file is not found
	
	"""
    # Get file read parameters from UI elements
    startingRow = int(startRow.value)
    startingCol = int(startCol.value)

	try:
		# Read in data from temp file created via javascript function
		dat = np.genfromtxt(join(tempfile.gettempdir(), "dataFile"), delimiter=',', skip_header=startingRow)
	except IOError:
		print("Raw data file not found!")

	# Define the return variables
    X = dat[:,startingCol-1]
    Y = dat[:,startingCol]
    sampling_dt = X[1] - X[0]

	return(X, Y, dt)

def plot_raw_data(X, Y)
	"""
	Plots the loaded raw data into the main bokeh plot

	Args:
		X: list floats containing the horizontal (time) axis data to be plotted
		Y: list of floats containing the vertical (voltage) axis data to be plotted

	Returns:
		A bokeh plot object
	"""
	# Define global tools for all bokeh plots

	plotTools = "lasso_select,reset,pan,wheel_zoom,box_select,save,hover"

	# create a plot and style its properties
	p_rawData = figure(title="Raw Data", x_axis_label='Time (s)', y_axis_label='Voltage (V)', 
			lod_threshold=None, plot_width=1200, active_scroll="wheel_zoom", tools=plotTools)

	# Add plots and data sources

	ds = ColumnDataSource(dict(x=[], y=[]))
	ds_analysis = ColumnDataSource(dict(x=[], y=[]))
	p_rawData_line = p_rawData.line(x='x', y='y', source=ds, line_width=1)

	return(p_rawData)

def update_raw_data_plot(X, Y, DataSource):
    # Update plot data source
    spc = int(((X[-1] - X[0])/maxPts)/(X[1]-X[0]))
    x = X[np.arange(0, X.size, spc)]
    y = Y[np.arange(0, X.size, spc)]
    DataSource.data = dict(x=x,y=y)

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
        print(len(ds.data['x']))
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
			wv_v[0:breakoutPt] = 0
			wvt_ds_velocity.data['velocity'] = wv_v

		if breakoutPt != 0 and fft_ds_velocity.data['velocity'] != []:
			fft_v = np.array(fft_ds_velocity.data['velocity'])
			fft_v[0:breakoutPt] = 0
			fft_ds_velocity.data['velocity'] = fft_v
	except TypeError:
		pass

# Define callbacks for zooming and panning on primary plot

p_rawData.x_range.on_change('end', downsample)

# Create customJS to get HTML5 file input information
js_callback = CustomJS(code=open(join(dirname(__file__), "readFile.js")).read())

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

filePreview = Div(text = '<h1> File Preview </h1> <div id="fileContents">x,y</div>', id="filePreview", width=600)
fileReadError = Div(text='', width=600)
startCol = TextInput(value="1", title="Starting Column:")
startRow = TextInput(value="1", title="Starting Row:")

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