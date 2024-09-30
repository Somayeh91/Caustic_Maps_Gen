#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python version of the cf_fft.pro IDL code written by Alexy    #
# Chepurnov, and available at http://www.astro.wisc.edu/~lazarian/code.html.   #
# This function calculates the correlation function of an image or data cube,  #
# using a fast fourier transform. I have modified the original code so that    #
# this code can calculate the cross-correlation function of two data sets, and #
# not just the auto-correlation function of a single field as in the original  #
# code.                                                                        #
#                                                                              #
# Author: Chris Herron (adapted from code written by Alexy Chepurnov)          #
# Start Date: 3/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, sys to exit if a problem occurs
import numpy as np
import sys

# Define the function _centered, which simply returns the centre portion of
# a specified array, using a specified size.
# This code was copied from 
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/signaltools.py#L210
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

# Define the function _next_regular, which finds the next 5-smooth number 
# bigger than the given target. This is needed to optimise the FFT calculation
# This code was copied from
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/signaltools.py#L210
def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf') # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

# Define the function cf_fft, which calculates the correlation function of an
# image or data cube.
def cf_fft(field1, field2 = None, no_fluct = False, normalise = False, mirror = False):
	'''
	Description
		This function calculates the correlation function of an image or data
		cube, using a fast fourier transform. If only one field is specified,
		then the auto-correlation function of that field is calculated. If two
		fields are specified, then the cross-correlation field1 * field2 is
		calculated. The cross-correlation is defined as the expectation value
		of f(x) g(x + r), for some lag r and position x, where f and g are 
		functions of position, for example. For auto-correlation, g == f.
	
	Required Input
		field1: A numpy array containing an image or data cube. Must be one, two
			    or three dimensional. If this is the only field specified, then
			    the auto-correlation of this field is calculated.
		field2: A numpy array containing an image or data cube. Must be one, two
			    or three dimensional. If this field is specified, then a cross-
			    correlation field1 * field2 is calculated.
			    In this case, it must have the same size as field1.
			    If this is None, then an auto-correlation of field1 is 
			    calculated.
		no_fluct: A boolean value. If False, then the mean value of the data
				  is subtracted from the data before calculating the correlation
				  function. If True, then there is no subtraction of the mean.
		normalise: A boolean value. If False, then the correlation function is 
				   calculated. If True, then the correlation function is 
				   normalised so that it must lie between +/- 1. Only use if 
				   field2 = None.
		mirror: A boolean value. If True, then the mirror image of the 
				correlation function is returned. If False, then nothing happens
	
	Output
		cf: A numpy array with the same shape as the input image or data cube.
			This array gives the values for the correlation function of
			the data.
	'''

	# Determine the shape of the input data for field1
	sizefield = np.asarray(np.shape(field1))

	# Check to see if a cross-correlation function is being calculated
	if field2 != None:
		# In this case something has been provided for field 2
		# Determine the shape of the input data for field2
		sizefield2 = np.asarray(np.shape(field2))

		# # Set a variable which represents the shape of the zero-padded array to
		# # use when calculating a correlation function
		# shape = (sizefield + sizefield2) / 2.0

		# Check to see if one of the field matrices is complex. If one of the
		# matrices is complex, then this variable will be True, and then the 
		# complete FFT must be calculated. Otherwise, this is False, and a 
		# faster FFT calculation can be used.
		complex_result = (np.issubdtype(field1.dtype, np.complex) or\
                      np.issubdtype(field2.dtype, np.complex))

		# Check to see if field1 and field2 have the same shape
		if np.any(sizefield != sizefield2):
			# In this case the fields do not have the same shape, so print an
			# error message to the screen
			print ('cf_fft ERROR: Input data arrays must have the same shape')

			# Exit the program, since we should not proceed if the arrays
			# have different shapes
			sys.exit()
	else:
		# # In this case field2 is not provided, so we need to define the
		# # shape of the zero-padded array to use when calculating the auto-
		# # correlation function. The zero-padded array should be twice as large
		# # as the input in each dimension, so that a linear correlation function
		# # is calculated, rather than a cyclic correlation function
		# shape = 1.0 * sizefield

		# Check to see if the field matrix is complex. If this variable is True,
		# then the complete FFT must be calculated. Otherwise, this is False, 
		# and a faster FFT calculation can be used.
		complex_result = (np.issubdtype(field1.dtype, np.complex))

	# Calculate the length of the first dimension of the data
	N1 = sizefield[0]

	# Initialise variables to hold the lengths of the second and third 
	# dimensions, if they are present.
	N2 = 1
	N3 = 1

	# Check to see if the data has 2 or more dimensions
	if len(sizefield) >= 2:
		# In this case, there are two or more dimensions, so extract the length
		# of the second dimension
		N2 = sizefield[1]

	# Check to see if the data has 3 or more dimensions
	if len(sizefield) >= 3:
		# In this case, there are three or more dimensions, so extract the 
		# length of the third dimension
		N3 = sizefield[2]

	# Check to see if the data has four or more dimensions    
	if len(sizefield) >= 4:
		# In this case there are four or more dimensions, so print an error
		# message to the screen.
		print ('Well, please no more than 3 dimensions !')
		
		# Stop the function from continuing on, since this function cannot
		# handle mirroring the correlation function for data that is four 
		# dimensional or higher.
		sys.exit()

	# Check to see whether the mean should be subtracted from the data before
	# calculating the correlation function
	if no_fluct == False:
		# In this case we need to subtract the mean of the data before 
		# calculating the correlation function of the data
		field1 = field1 - np.mean(field1, dtype = np.float64)

		# Check to see if field 2 has been specified
		if field2 != None:
			# In this case, field2 has been specified, so subtract its mean
			field2 = field2 - np.mean(field2, dtype = np.float64)
	
	# If no_fluct == True, then we do not subtract the mean of the data before 
	# calculating the correlation function, so do nothing to the data

	# # Calculate the optimal shape of the array to use when performing the FFT
	# fshape = np.asarray([_next_regular(int(d)) for d in shape])

	# # Create a slice object that will automatically retrieve the original
	# # segment of the array, that was not created by extending the array to 
	# # optimise the speed of the FFT
	# fslice = tuple([slice(0, int(sz/1.0)) for sz in shape])

	# Check to see if one of the input matrices is complex
	if complex_result == True:
		# In this case one of the matrices is complex, so we need to use the 
		# full FFT calculation, without taking any shortcuts

		# Calculate the fourier transform of field1. 
		# NOTE: In the IDL convention for forward FFT, there is a normalisation
		# factor, but the Python convention does not involve the normalisation
		# factor. To ensure the same output as the IDL code, the result
		# of the FFT is divided by the number of data points, to undo the effect
		# of the normalisation.
		# NOTE2: We need to zero-pad the array to twice it's original size, so
		# that a linear correlation is calculated, instead of a cyclic 
		# correlation
		fftfield1 = np.fft.fftn(field1) / np.size(field1)
		
		# Check to see if field2 has been specified, as this determines whether
		# we need to calculate an auto- or cross-correlation function
		if field2 != None:
			# In this case we are calculating a cross-correlation function, so
			# calculate the Fourier transform of field2
			fftfield2 = np.fft.fftn(field2) / np.size(field2)
			
			# Multiply the complex conjugate of the fourier transform of field1 
			# by the fourier transform of field 2
			ps = np.conj(fftfield1) * fftfield2
		else:
			# In this case we are calculating an auto-correlation function for 
			# field 1

			# Multiply the fourier transform of the data by it's complex 
			# conjugate
			ps = fftfield1 * np.conj(fftfield1)

		# Perform an inverse fourier transform on the result obtained by 
		# multiplying the fourier transform with its conjugate
		# NOTE: In the IDL convention for inverse FFT, there is no normalisation
		# factor, but the Python convention involves dividing by the number of
		# data points. To ensure the same output as the IDL code, the result
		# of the inverse FFT is multiplied by the number of data points, to
		# undo the effect of the normalisation.
		cf = np.fft.ifftn(ps) * np.size(ps) 
	else:
		# In this case both matrices are real, and so we can use the rfftn 
		# function to save time

		# Calculate the fourier transform of field1. 
		# NOTE: In the IDL convention for forward FFT, there is a normalisation
		# factor, but the Python convention does not involve the normalisation
		# factor. To ensure the same output as the IDL code, the result
		# of the FFT is divided by the number of data points, to undo the effect
		# of the normalisation.
		# NOTE2: We need to zero-pad the array to twice it's original size, so
		# that a linear correlation is calculated, instead of a cyclic 
		# correlation
		fftfield1 = np.fft.fftn(field1) / np.size(field1)

		# Check to see if field2 has been specified, as this determines whether
		# we need to calculate an auto- or cross-correlation function
		if field2 != None:
			# In this case we are calculating a cross-correlation function, so
			# calculate the Fourier transform of field2
			fftfield2 = np.fft.fftn(field2) / np.size(field2)

			# Multiply the complex conjugate of the fourier transform of field1 
			# by the fourier transform of field 2
			ps = np.conj(fftfield1) * fftfield2
		else:
			# In this case we are calculating an auto-correlation function for 
			# field 1

			# Multiply the fourier transform of the data by it's complex 
			# conjugate
			ps = fftfield1 * np.conj(fftfield1)

		# Perform an inverse fourier transform on the result obtained by 
		# multiplying the fourier transform with its conjugate
		# NOTE: In the IDL convention for inverse FFT, there is no normalisation
		# factor, but the Python convention involves dividing by the number of
		# data points. To ensure the same output as the IDL code, the result
		# of the inverse FFT is multiplied by the number of data points, to
		# undo the effect of the normalisation.
		cf = np.fft.ifftn(ps) * np.size(ps) 

		# Due to numerical imprecision, there may be small imaginary parts in
		# every entry of the produced array. We are only interested in the real 
		# part, so extract that from the data
		cf = np.real(cf)

	# # Try using the fftconvolve function in Scipy to calculate the correlation
	# # function. First check to see if we are calculating an auto-correlation
	# # function
	# if field2 == None:
	# 	# In this case we are calculating the auto-correlation function,
	# 	# so run the fftconvolve function with field1 only.
	# 	# Note that this line uses the fact that convolution and correlation
	# 	# are virtually the same. The correlation is obtained from the 
	# 	# convolution by reversing one of the arrays before calculating the
	# 	# convolution. But this means we need to run slightly different code
	# 	# depending on how many dimensions were in the input data

	# 	# Check to see if the data is three-dimensional
	# 	if len(sizefield) == 3:
	# 		# In this case we need to reverse all three axes
	# 		cf = signal.fftconvolve(field1, field1[::-1,::-1,::-1])
	# 	elif len(sizefield) == 2:
	# 		# In this case we need to reverse two dimensions
	# 		cf = signal.fftconvolve(field1, field1[::-1,::-1])
	# 	else:
	# 		# In this case the input field is one dimensional, so reverse one
	# 		# dimension only
	# 		cf = signal.fftconvolve(field1, field1[::-1])
	# else:
	# 	# In this case we are calculating the cross-correlation function, and
	# 	# so we need to run the fftconvolve function with both field1 and
	# 	# field2. As for the calculation of the auto-correlation function, we
	# 	# need to reverse one of the input fields so that we calculate a 
	# 	# correlation, and not a convolution

	# 	# Check to see if the data is three-dimensional
	# 	if len(sizefield) == 3:
	# 		# In this case we need to reverse all three axes
	# 		cf = signal.fftconvolve(field1, field2[::-1,::-1,::-1])
	# 	elif len(sizefield) == 2:
	# 		# In this case we need to reverse two dimensions
	# 		cf = signal.fftconvolve(field1, field2[::-1,::-1])
	# 	else:
	# 		# In this case the input field is one dimensional, so reverse one
	# 		# dimension only
	# 		cf = signal.fftconvolve(field1, field2[::-1])

	# Check to see if the mirror image of the correlation function needs
	# to be returned.
	if mirror == True:
		# Let's do here the trick of producing the mirror images
		# Create some variables that will be used to create the mirror image
		nyq1 = N1/2.0
		nyq2 = N2/2.0
		nyq3 = N3/2.0
		
		# Loop over the third dimension, to reorder elements in the array
		for i3 in range(N3):
			# Create a new variable, to handle indexing of the mirrored array
			i3k = i3

			# Check to see if the iteration along the third dimension is past 
			# the halfway point
			if i3 >= nyq3:
				# In this case we are past the halfway point, so calculate i3k
				# to take this into account
				i3k = N3 - i3
			
			# Loop over the second dimension, to reorder elements
			for i2 in range(N2):
				# Create a new variable, to handle indexing of the 
				# mirrored array
				i2k = i2

				# Check to see if the iteration along the second dimension is 
				# past the halfway point
				if i2 >= nyq2:
					# In this case we are past the halfway point, so calculate 
					# i2k to take this into account
					i2k = N2 - i2
				
				# Loop over the first dimension, to reorder elements
				for i1 in range(N1):
					# Create a new variable, to handle indexing of the 
					# mirrored array
					i1k = i1

					# Check to see if the iteration along the first dimension is
					# past the halfway point
					if i1 >= nyq1:
						# In this case we are past the halfway point, so 
						# calculate i1k to take this into account
						i1k = N1 - i1
						
					# Now that i1k, i2k and i3k have been determined, update
					# entries of the correlation data to make it appear
					# mirrored
					cf[i1,i2,i3] = cf[i1k,i2k,i3k]

	# If required, calculate the normalised auto-correlation function
	if (field2 == None) and (normalise == True):
		# In this case we need to return the normalised correlation function 
		# Calculate the square of the mean of the field1 values
		field1_sq_mean = np.power( np.mean(field1, dtype = np.float64), 2.0 )

		# Calculate the mean of the field1 values squared
		field1_mean_sq = np.mean( np.power(field1, 2.0), dtype = np.float64 )

		# Calculate the normalised auto-correlation function 
		cf = (cf - field1_sq_mean) / (field1_mean_sq - field1_sq_mean)

	# Return the correlation function to the caller
	return cf