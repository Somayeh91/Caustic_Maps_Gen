import numpy as np
from scipy import interpolate, signal
from scipy.signal import fftconvolve
import sys
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


class convolver():

    def __init__(self, map_x_pixl=1000,
                 map_y_pixl=1000,
                 factor=1,
                 rsrc=1,
                 tot_r_e=25,
                 stype='gaussian'):

        self.nxpix = map_x_pixl
        self.nypix = map_y_pixl
        self.factor = factor
        self.tot_r_e = tot_r_e
        self.xr = [-(tot_r_e / factor), (tot_r_e / factor)]
        self.yr = [-(tot_r_e / factor), (tot_r_e / factor)]
        self.rsrc = rsrc
        self.stype = stype
        self.def_kernel()

    def def_kernel(self):

        if self.stype.lower() == 'gaussian':
            # set the appropriate side length for the kernel
            rmax = 2.0 * self.rsrc * np.sqrt(np.log(10.))  # 99% containment
            self.pixsz = (self.xr[1] - self.xr[0]) / float(self.nxpix)
            side = np.arange(-rmax, rmax, self.pixsz)

            # create the kernel assuming self.rsrc = sigma
            self.xm, self.ym = np.meshgrid(side, side, indexing='ij')
            self.kernel = np.exp(-(self.xm ** 2 + self.ym ** 2) /
                                 (2.0 * (self.rsrc) ** 2)) / \
                          (2.0 * np.pi * (self.rsrc) ** 2)

    def conv_map(self, map, padding=50):
        length = map.shape[0]
        map = np.pad(map, (padding, ), 'wrap')
        self.magcon = fftconvolve(map, self.kernel, mode='same')  # ndimage.convolve(map, self.kernel)
        self.magcon *= self.pixsz * self.pixsz
        map_min = np.min(map.flatten())

        if np.min(self.magcon.flatten()) > map_min:
            self.magcon[self.magcon < map_min] = map_min

        self.magcon = self.magcon[padding:length+padding, padding:length+padding]

    # def conv_map(self, map):

    #     # Pad the small array and large array to ensure they have the same dimensions
    #     pad_x = map.shape[0] - self.kernel.shape[0]
    #     pad_y = map.shape[1] - self.kernel.shape[1]
    #     padded_small = np.pad(self.kernel, [(0, pad_x), (0, pad_y)], mode='constant')

    #     # Perform the FFT on the padded arrays
    #     fft_small = np.fft.fft2(padded_small)
    #     fft_large = np.fft.fft2(map)

    #     # Multiply the transformed arrays
    #     fft_result = np.multiply(fft_small, fft_large)

    #     # Perform the inverse FFT on the result
    #     result = np.fft.ifft2(fft_result)

    #     # Return the real part of the result (ignoring the imaginary part)
    #     self.magcon = np.real(result)

    def lc_maker(self, angle=30, distance=15, n_samp=100):

        x0 = self.xr[0]
        dx = self.xr[1] - x0  # map units in R_E
        mmap = self.magcon

        mappix = 1000 / dx
        nsamp = n_samp
        distance = distance  # in maps units
        dt = (22 * 365) / 1.5  # from Kochanek (2004) for Q2237+0305: The quasar takes 22 years to go through 1.5 R_Ein
        tmax = distance * dt

        t = np.linspace(0, tmax, nsamp, dtype='int')

        # if origin==None:
        origin = [i // 2 for i in mmap.shape]
        # else:
        #     origin = [int(round((i-x0)*float(nxpix)/dx)) for i in origin]

        # -------- determine the x and y pixel values
        angle = angle
        rad = (angle) * np.pi / 180.0
        crad = np.cos(rad)
        srad = np.sin(rad)
        pixsz = dx / float(nxpix)  # size of one pixel in R_E
        drpix = distance / dx * float(nxpix) / float(nsamp)  # step's size in pixels
        ix = [i * drpix * crad + origin[0] for i in range(nsamp)]
        iy = [i * drpix * srad + origin[1] for i in range(nsamp)]

        # -------- check boundaries
        if (min(ix) < 0) or (min(iy) < 0) or (max(ix) > nxpix) or (max(iy) > nypix):
            print("ML_MAGMAP: Error - lightcurve too long!!!")
            sys.exit(-1)

        # -------- interpolate onto light curve pixels
        #          (using interp2d!!!)
        x, y = np.arange(float(nxpix)), np.arange(float(nypix))
        mapint = interpolate.interp2d(x, y, mmap, kind='cubic')

        self.lc = {}
        self.lc['lc'] = np.array([mapint(i, j)[0] for i, j in zip(*[iy, ix])])
        self.lc['t'] = t
        self.lc['origin'] = origin
