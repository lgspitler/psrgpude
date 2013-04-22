#!/bin/env python

import numpy
from numpy.random import randn 
import ctypes as C
from numpy.ctypeslib import as_ctypes as as_c

lib=C.CDLL("./libspsearch.so")

fftlen=2**16
fwidth=4
dfloc=128

#Make some test data
fakedata=randn(fftlen)
#fakedata=numpy.zeros(fftlen, dtype='float32')
convdata=numpy.zeros(fftlen, dtype='float32')
#fakedata[dfloc]=1

lib.ts_convolve(as_c(fakedata), C.c_int(fwidth), C.c_int(fftlen), as_c(convdata))

print convdata[dfloc-fwidth:dfloc+fwidth]
print "Max: ", max(convdata)
print "Max loc:", convdata.argmax()
