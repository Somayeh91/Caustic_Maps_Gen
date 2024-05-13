from convolver import convolver
import numpy as np
from scipy import interpolate
import random
from tqdm import tqdm

def convolve_map(map, rsrc):
	cv0 = convolver(rsrc = rsrc, tot_r_e=12.5)
	cv0.conv_map(map.reshape((map.shape[1], map.shape[1])))
	return cv0.magcon

def lc_gen (map_,
			angle=30,
			nxpix=1000,
			nypix=1000,
			factor=1,
			tot_r_e=12.5,
			nsamp = 100,
			distance = 10, # in maps units
			dt = (22*365)/1.5, # from Kochanek (2004) for Q2237+0305: The quasar takes 22 years to go through 1.5 R_Ein
			set_seed = 42):

	xr        = [-(tot_r_e/factor), (tot_r_e/factor)]
	yr        = [-(tot_r_e/factor), (tot_r_e/factor)]
	rsrc = 1

	x0    = xr[0]
	dx    = xr[1]-x0 #map units in R_E

	mappix = nxpix/dx
	tmax = distance * dt

	t = np.linspace(0, tmax, nsamp, dtype='int')
	if np.isnan(set_seed):
		random.seed()
	else:
		random.seed(set_seed)
	origin = [random.randint(1, nxpix) for i in map_.shape]
	rad   = (angle)*np.pi/180.0
	crad  = np.cos(rad)
	srad  = np.sin(rad)
	pixsz = dx/float(nxpix) #size of one pixel in R_E
	drpix = distance/dx * float(nxpix)/float(nsamp) # step's size in pixels
	ix    = [i*drpix*crad+origin[0] for i in range(nsamp)]
	iy    = [i*drpix*srad+origin[1] for i in range(nsamp)]

	# -------- check boundaries
	if (min(ix)<0) or (min(iy)<0) or (max(ix)>nxpix) or (max(iy)>nypix):
		# print("ML_MAGMAP: Error - lightcurve too long!!!")
		return (np.nan, np.nan, np.nan, np.nan)


	# -------- interpolate onto light curve pixels
	#          (using interp2d!!!)
	x, y   = np.arange(float(nxpix)), np.arange(float(nypix))
	mapint = interpolate.interp2d(x,y, map_,kind='cubic')
	return t, np.array([mapint(i,j)[0] for i,j in zip(*[iy,ix])]), origin, [ix[-1],iy[-1]]

def lc_gen_set(n_lc,
			   map_,
			   nxpix=1000,
			   nypix=1000,
			   factor=1,
			   tot_r_e=12.5,
			   nsamp = 100,
			   distance = 10, # in maps units
			   dt = (22*365)/1.5,
			   set_seed = 42):

	lc_set = np.zeros((n_lc, 2, nsamp))
	origin_end = np.zeros((n_lc, 2, 2))
	for n in tqdm(range(n_lc)):
		if np.isnan(set_seed):
			random.seed()
		else:
			random.seed(set_seed+n)
		ang = random.randint(0, 360)
		t = np.nan
		while isinstance(t, float):
			if np.isnan(set_seed):
				pass
			else:
				set_seed = set_seed+1
			t, lc, origin, end = lc_gen(map_,
									  angle=ang,
									  nxpix=1000,
									  nypix=1000,
									  factor=1,
									  tot_r_e=12.5,
									  nsamp = 100,
									  distance = 10, # in maps units
									  dt = (22*365)/1.5,
									  set_seed = set_seed)
		lc_set[n, 0, :] = t
		lc_set[n, 1, :] = lc
		origin_end[n, 0, :] = np.asarray(origin)
		origin_end[n, 1, :] = np.asarray(end)

	return(lc_set, origin_end)
