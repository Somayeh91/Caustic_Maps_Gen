#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python version of the sf_fft.pro IDL code written by Alexy    #
# Chepurnov, and available at http://www.astro.wisc.edu/~lazarian/code.html.   #
# This function calculates the structure function of an image or data cube,    #
# using a fast fourier transform.                                              #
#                                                                              #
# Author: Chris Herron (adapted from code written by Alexy Chepurnov)          #
# Start Date: 4/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, sys to exit if a problem occurs
import numpy as np
import sys

# Import cf_fft, to calculate the correlation function used to calculate the
# structure function
from cf_fft import cf_fft

# Define the function sf_fft, which calculates the structure function of an
# image or data cube.
def sf_fft(field, no_fluct = False, normalise = False, mirror = False, const = False):
	'''
	Description
		This function calculates the structure function of an image or data
		cube, using a fast fourier transform.
	
	Required Input
		field: A numpy array containing an image or data cube. Must be one, two
			   or three dimensional. 
		no_fluct: A boolean value. If False, then the mean value of the data
				  is subtracted from the data before calculating the structure
				  function. If True, then there is no subtraction of the mean.
		normalise: A boolean value. If False, then the structure function is 
				   calculated. If True, then the structure function is 
				   normalised so that it must lie between 0 and 2.
		mirror: A boolean value. If True, then the mirror image of the 
				structure function is returned. If False, then nothing happens
		const: A boolean value. If True, then the structure function is 
			   calculated from how the auto-correlation function deviates from
			   its maximum value.
	
	Output
		sf: A numpy array with the same shape as the input image or data cube.
			This array gives the values for the structure function of the data.
	'''

	# Calculate the auto-correlation function of the provided field, using
	# the cf_fft function.
	acf = cf_fft(field, no_fluct = no_fluct, normalise = normalise, mirror = mirror)

	# # Determine the shape of the input data
	# sizefield = np.shape(field)

	# # Calculate the length of the first dimension of the data
	# N1 = sizefield[0]

	# # Initialise variables to hold the lengths of the second and third 
	# # dimensions, if they are present.
	# N2 = 1
	# N3 = 1

	# # Check to see if the data has 2 or more dimensions
	# if len(sizefield) >= 2:
	# 	# In this case, there are two or more dimensions, so extract the length
	# 	# of the second dimension
	# 	N2 = sizefield[1]

	# # Check to see if the data has 3 or more dimensions
	# if len(sizefield) >= 3:
	# 	# In this case, there are three or more dimensions, so extract the 
	# 	# length of the third dimension
	# 	N3 = sizefield[2]

	# # Check to see if the data has four or more dimensions    
	# if len(sizefield) >= 4:
	# 	# In this case there are four or more dimensions, so print an error
	# 	# message to the screen.
	# 	print 'Well, please no more than 3 dimensions !'
		
	# 	# Stop the function from continuing on, since this function cannot
	# 	# handle mirroring the structure function for data that is four 
	# 	# dimensional or higher.
	# 	sys.exit()

	# Check to see whether the mean should be subtracted from the data before
	# calculating the structure function
	if no_fluct == False:
		# In this case we need to subtract the mean of the data before 
		# calculating the structure function of the data
		field1 = field - np.mean(field, dtype = np.float64)

	else:
		# In this case we do not subtract the mean of the data before 
		# calculating the structure function, so do nothing to the data
		field1 = field

	# # Calculate the fourier transform of the data
	# # NOTE: In the IDL convention for forward FFT, there is a normalisation
	# # factor, but the Python convention does not involve the normalisation
	# # factor. To ensure the same output as the IDL code, the result
	# # of the FFT is divided by the number of data points, to
	# # undo the effect of the normalisation.
	# fftfield = np.fft.fftn(field1) / np.size(field1)

	# # Multiply the fourier transform of the data by it's complex conjugate
	# ps = fftfield * np.conj(fftfield)

	# # Perform an inverse fourier transform on the result obtained by multiplying
	# # the fourier transform with its conjugate. This gives the auto-correlation
	# # function.
	# # NOTE: In the IDL convention for inverse FFT, there is no normalisation
	# # factor, but the Python convention involves dividing by the number of
	# # data points. To ensure the same output as the IDL code, the result
	# # of the inverse FFT is multiplied by the number of data points, to
	# # undo the effect of the normalisation.
	# acf = np.fft.ifftn(ps) * np.size(ps)

	# # Due to numerical imprecision, there may be small imaginary parts in
	# # every entry of the produced array. We are only interested in the real 
	# # part, so extract that from the data
	# acf = np.real(acf)

	# # Check to see if the mirror image of the auto-correlation function needs
	# # to be returned.
	# if mirror == True:
	# 	# Let's do here the trick of producing the mirror images
	# 	# Create some variables that will be used to create the mirror image
	# 	nyq1 = N1/2.0
	# 	nyq2 = N2/2.0
	# 	nyq3 = N3/2.0
		
	# 	# Loop over the third dimension, to reorder elements in the array
	# 	for i3 in range(N3):
	# 		# Create a new variable, to handle indexing of the mirrored array
	# 		i3k = i3

	# 		# Check to see if the iteration along the third dimension is past 
	# 		# the halfway point
	# 		if i3 >= nyq3:
	# 			# In this case we are past the halfway point, so calculate i3k
	# 			# to take this into account
	# 			i3k = N3 - i3
			
	# 		# Loop over the second dimension, to reorder elements
	# 		for i2 in range(N2):
	# 			# Create a new variable, to handle indexing of the 
	# 			# mirrored array
	# 			i2k = i2

	# 			# Check to see if the iteration along the second dimension is 
	# 			# past the halfway point
	# 			if i2 >= nyq2:
	# 				# In this case we are past the halfway point, so calculate 
	# 				# i2k to take this into account
	# 				i2k = N2 - i2
				
	# 			# Loop over the first dimension, to reorder elements
	# 			for i1 in range(N1):
	# 				# Create a new variable, to handle indexing of the 
	# 				# mirrored array
	# 				i1k = i1

	# 				# Check to see if the iteration along the first dimension is
	# 				# past the halfway point
	# 				if i1 >= nyq1:
	# 					# In this case we are past the halfway point, so 
	# 					# calculate i1k to take this into account
	# 					i1k = N1 - i1
						
	# 				# Now that i1k, i2k and i3k have been determined, update
	# 				# entries of the auto-correlation data to make it appear
	# 				# mirrored
	# 				acf[i1,i2,i3] = acf[i1k,i2k,i3k]

	# Check to see if the mean of the data was subtracted from the data before
	# calculating the auto-correlation function, as this changes the formula
	# used to calculate the structure function
	if (no_fluct == True) and (normalise == False):
		# In this case the mean of the data was not subtracted before 
		# calculating the the auto-correlation function.
		# Calculate the structure function from the auto-correlation function
		sf = 2.0 * (np.mean(np.power(field1, 2.0), dtype = np.float64) - acf)
		#sf = 2.0 * (np.max(acf) - acf)
	elif normalise == True:
		# Calculate the normalised structure function from the normalised 
		# auto-correlation function
		sf = 2.0 * (1.0 - acf)
	else:
		# In this case the mean of the data was subtracted before calculating
		# the auto-correlation function, so use a different formula to 
		# calculate the structure function. Note that the un-biased estimator
		# of the sample variance is used.
		sf = 2.0 * (np.var(field1, ddof = 1, dtype = np.float64) - acf)

	# Check to see if the structure function should be calculated based on
	# how the auto-correlation function deviates from its maximum value
	if const == True:
		# In this case we are calculating the structure function based on
		# how the auto-correlation function deviates from its maximum value.
		# Subtract the minimum value of the auto-correlation function from this
		# function
		acf = acf - np.min(acf)

		# Compute the structure function by looking at differences between
		# the maximum of the corrected auto-correlation function and the
		# function itself.
		sf = 2.0 * (np.max(acf) - acf)

	# Due to numerical imprecision, there may be small imaginary parts in
	# every entry of the produced array. We are only interested in the real 
	# part, so extract that from the data
	sf = np.real(sf)

	# Return the structure function to the caller
	return sf