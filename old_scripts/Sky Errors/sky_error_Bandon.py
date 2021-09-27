#import tools
import os
import numpy as np
from scipy import optimize
from astropy.table import Table,Column
import matplotlib.pyplot as plt
from astropy.io import fits,ascii

def aperturesEverywhere(imfn,coordfn,apsize,minxpx=0,maxxpx=None,minypx=0,maxypx=None):
	'''Writes a qphot coordinates file for the specified \
image. It simply makes a grid of aperture placings within the size of the \
image that qphot can read in.
	
	INPUT
	------------------------------------------------------
	
	imfn: str
		The full path and filename of the fits image.
	coordfn: str
		The name of the coordinates file for qphot to read.
	apsize: float
		The radius of the aperture, in pixels. Note it is the /radius/ \
		not the diameter; that is how qphot takes the input.
	minxpx: int
		The minimum pixel value to use along the x-axis. Defaults to 0 if left blank.
	maxxpx: int
		The maximum pixel value to use along the x-axis. Defaults to NAXIS1 if left blank.
	minypx: int
		The minimum pixel value to use along the y-axis. Defaults to 0 if left blank.
	maxypx: int
		The maximum pixel value to use along the y-axis. Defaults to NAXIS2 if left blank.
	
	OUTPUT
	------------------------------------------------------
	
	None
	'''
	head = fits.getheader(imfn)
	size = np.ceil(apsize*2)
	if not maxxpx:
		maxxpx = head['NAXIS1']
	if not minypx:
		minypx = head['NAXIS2']
	xs = np.arange(minxpx+size/2,maxxpx,size)
	ys = np.arange(minypx+size/2,maxypx,size)
	coords = open(coordfn,'w')
	coords.write('X Y\n')
	for x in xs:
		for y in ys:
			coords.write('{0} {1}\n'.format(x,y))
	coords.close()

def makeIrafCommand(fn,images,coords,outputs,apsize,zp=0,speak=True):
	'''Makes a .cl file for Iraf to execute. Pretty optional unless one has \
a lot of files to check at once.
	
	INPUT
	------------------------------------------------------
	
	fn: str
		Name of the Iraf command file.
	images: 1D array-like
		A list of all the input images.
	coords: 1D array-like
		A list of all the coordinate files.
	outputs: 1D array-like
		A list of all the output files.
	apsize: float
		The radius of the aperture, in pixels. Note it is the /radius/ \
		not the diameter; that is how qphot takes the input.
	speak: bool
		Whtheror not to include a line in the Iraf file to announce the \
		completion of the task.
	
	OUTPUT
	------------------------------------------------------
	
	None
	'''
	iraf = open(fn,'w')
	iraf.write('noao\ndigiphot\napphot\n\n')
	for n in range(len(images)):
		iraf.write('qphot image={0} cbox=0 annulus=10 dannulus=3 aperture={1} coords={2} output={3} zmag={4} interac=no\n'.format(images[n],apsize,coords[n],outputs[n],zp))
	iraf.write('\n')
	if speak:
		iraf.write('!say -v Good News "Iraf has finished"\n')
	iraf.write('logout')
	iraf.close()

def runIraf(fn):
	'''This command doesn't actually work for me, because of the way \
os.system handles subshells. I've included it mostly for reference and on \
the off chance that it's a problem with my machine. Assuming it doesn't \
actually work though, go to the terminal and input:

cl < filename.cl >& testlog

where filename.cl is the file you created with makeIrafCommand. Alternatively \
open Iraf and do:

cl> task $skyerror = home$filename.cl
cl> skyerror

They should behave identically.'''
	os.system('cl < {0} >& testlog'.format(fn))

def catFilter(catfn,covfn,mincov):
	'''Filters out all the aperture data with coverage below a minimum \
value.
	
	INPUT
	------------------------------------------------------
	
	catfn: str
		The name of the Iraf output file with the aperture data.
	covfn: str
		The filename of the coverage map.
	mincov: float
		The minimum coverge to be allowed.
	
	OUTPUT
	------------------------------------------------------
	
	newcat: Astropy table
		A table containing the centre of each aperture (in pixels) and \
		theflux therein.
	'''
	cat = ascii.read(catfn)
	covmap = fits.getdata(covfn)
	#head = fits.getheader(covfn)
	#cpix = coordToPix(centre[0],centre[1],'/Users/bbdz86/Documents/Madcows/Mag_error/irac_full_ch1/covmaps/'+cl+'_irac1_full_cov.fits')
	xs = []
	ys = []
	fluxes = []
	#radii = []
	for n in range(len(cat)):
		if covmap[np.int(cat['YCENTER'][n])][np.int(cat['XCENTER'][n])] > mincov:
			#radius = np.sqrt((cat['XCENTER'][n] - cpix[0])**2 + (cat['YCENTER'][n] - cpix[1])**2)
			xs.append(cat['XCENTER'][n])
			ys.append(cat['YCENTER'][n])
			fluxes.append(cat['FLUX'][n])
			#radii.append(radius)
	#rMpc = np.array(radii)*np.abs(head['PXSCAL1'])*tools.angDiaDist(z)/1000
	#rr500 = rMpc/r500
	newcat = Table()
	newcat.add_columns([Column(data=np.array(xs),name='x'),Column(data=np.array(ys),name='y'),Column(data=np.array(fluxes),name='flux')])#,Column(data=np.array(radii),name='rpix'),Column(data=rMpc,name='rMpc'),Column(data=rr500,name='r/r500')])
	return newcat

def catCats(cats):
	'''Takes a list of tables and concatenates them into one master \
table. Requires that all of the tables have the same column names.
	
	INPUT
	------------------------------------------------------
	
	cats: 1D array-like
		List of tables to concatenate.
	
	OUTPUT
	------------------------------------------------------
	
	allCats: Astropy table
		The new table
	'''
	allCats = Table(cats[0][0])
	for n in range(1,len(cats[0])):
		allCats.add_row(cats[0][n])
	for cat in cats[1:]:
		for n in range(len(cat)):
			allCats.add_row(cat[n])
	return allCats

def gaussian(x,a,mean,std):
	'''A Gaussian. You never have to use this, it's just to be read by \
other functions.'''
	return a*np.exp(-(x-mean)**2/(2*std**2))

def fitGauss(array,binnum=40,histrange=(-1,1),right=0.2):
	'''Fits an array to the Gaussian defined previously.
	
	INPUT
	------------------------------------------------------
	
	array: 1D array-like
		The list of fluxes to fit.
	binnum: int
		The number of histogram bins to use. Default = 40.
	histrange: 2-tuple
		A tuple of th maximum and minimum values over which to plot \
		the histogram. Default = (-1,1).
	right: float
		The highest value for which to include in the Gaussian fit. \
		Default = 0.2
	
	OUTPUT
	------------------------------------------------------
	
	xs: 1D array
		The centres of the histogram bins.
	hist: 1D array
		The heights of each histogram bin.
	fit: tuple
		The fitted values for each variable of the Gaussian: \
		height, mean and standard deviation.
	'''
	hist = np.histogram(array,bins=binnum,range=histrange)
	xs = []
	for n in range(len(hist[1])-1):
		x = (hist[1][n] + hist[1][n+1])/2		
		xs.append(x)
	for n in range(len(xs)):
		if xs[n] > right:
			break
	fit = optimize.curve_fit(gaussian,xs[:n],hist[0][:n])
	return np.array(xs),hist[0],fit[0]

def fluxError(fit,ZP,apcorrect):
	'''Takes the detailed output from fitGauss and simplifies it to one \
value for the sky error, given in rea flux units.
	
	INPUT
	------------------------------------------------------
	
	fit: tuple
		The output from fitGauss.
	ZP: float
		The conversion factor to go from ADU to flux units.
	apcorrect: float
		The correction from the size of the aperture being used to \
		a standard aperture.
	
	OUTPUT
	------------------------------------------------------
	
	fluxerr: float
		The sky error in flux units. The specific units will depend on \
		the input zero point.
	'''
	fluxerr = np.abs(fit[2][2])*ZP*apcorrect
	return fluxerr

def plotGauss(fit,w=0.05,save=True,fn=''):
	'''Plots thehistogram of data given to fitGauss and the Gaussian that \
was fit to it.
	
	INPUT
	------------------------------------------------------
	
	fit: tuple
		Output from fitGauss.
	save: bool
		Whether or not to save the image.
	fn: str
		file name to which to save the image.
	
	OUTPUT
	------------------------------------------------------
	
	None
	'''
	plt.bar(fit[0]-(w/2),fit[1],width=w)
	plt.plot(np.arange(-1,1,0.01),gaussian(np.arange(-1,1,0.01),fit[2][0],fit[2][1],fit[2][2]),c='r',ls='--')#,label='mu = {0:1.3f}\nsigma = {1:1.3f}\nfluxerr = {2:1.2f} uJy'.format(fit[2][1],np.abs(fit[2][2]),np.abs(fit[2][2])*8.46*1.4))
	plt.xlabel('Flux')
	plt.ylabel('N')
	#plt.legend(loc='upper left')
	if save:
		plt.savefig(fn)

