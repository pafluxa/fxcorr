# coding: utf-8
import time
import pylab
from atsec.correlation import _correlation

import numpy
from numpy.fft import fft, ifft

# 1E6 Hz sampling
dt = 1./(2.8E6)
# 1 Second signal
T = 30.0
print 'nsamples = ', T/dt
t  = numpy.arange( 0, T, dt )

# Generate signal with a 1 SNR

# 0.3 second shifted, 0.01 second sigma Gaussian + noise
chirp = numpy.cos( 100*numpy.pi*t[:t.size/2]**3 ) 

s1    = numpy.zeros(t.size)
s1[0:s1.size/2] = chirp 
s1 += 4*numpy.random.random( s1.size )

# 0.5 second shifted, 0.01 second sigma  Gaussian + noise
s2  = numpy.zeros(t.size)
s2[ s2.size/2::] = chirp
s2 += 4*numpy.random.random( s2.size )

s1 -= numpy.mean(s1)
s2 -= numpy.mean(s2)

start = time.time()
corr = _correlation.correlate( s1.astype('float32'), numpy.zeros_like(s1).astype( 'float32' ),
                               s2.astype('float32'), numpy.zeros_like(s2).astype( 'float32' ) )
end = time.time()
print end - start

t = numpy.concatenate( (-t,t) )

pylab.subplot(211)
pylab.plot( t, corr )

# Compute correlation using numpy.fft

# Zero pad
s1 = numpy.concatenate( (s1, numpy.zeros_like(s1) ) )
s2 = numpy.concatenate( (s2, numpy.zeros_like(s2) ) ) 

start = time.time()
corr = numpy.fft.fftshift( ifft( fft(s1) * numpy.conj( fft(s2) ) ) )
end = time.time()
print end - start

pylab.subplot( 212 )
pylab.plot( t, corr )

pylab.show()
