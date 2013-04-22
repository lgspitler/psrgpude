import numpy as np
import ctypes as C
from numpy.ctypeslib import as_ctypes as as_c 
from sigpyproc.Readers import readTim
from sigpyproc.TimeSeries import TimeSeries
lib = C.CDLL("./libcuseek.so")

def error_check():
    if lib.checkError() != 0:
        raise Exception("CUDA exception")

class AccelSearch(object):
    def __init__(self,filename):
        self.tim = readTim(filename)
        self.header = self.tim.header
        
    def _get_accel_steps(self,max_accel=500.):
        #should be run before pad, just in case.
        h = self.header
        tsamp = h.tsamp
        fact  = 8300 * h.refdm * (abs(h.foff)/h.fch1**3)
        if fact >= 2:
            tsamp*=fact
        step_size = (64 * 299792458.0 * tsamp)/h.tobs**2
        accel_steps = np.arange(0,max_accel+step_size,step_size)
        accel_steps  = np.hstack((-accel_steps[1:][::-1],accel_steps))
        return accel_steps.astype("float32")

    def _pad(self):
        '''Forces the time series to be of length 2^N
        by padding the end with the mean of the time series
        '''
        n = 0
        while 2**n < self.tim.size:
            n+=1
        self.tim = self.tim.pad(2**n-self.tim.size)
        
    def _clean(self,zaplist=None):
        #run after pad to get correct RFI bins
        f_spec = self.tim.rFFT()
        f_spec = f_spec.rednoise()
        #f_spec.zapbirds(zaplist) 
        self.tim = f_spec.iFFT()

    def seek_gpu(self):
        steps = self._get_accel_steps()
        #Pad the time series to force it to have length 2^N
        self._pad()
        #Remove red noise from time series
        self._clean()
        lib.seekGPU(as_c(self.tim),
                    C.c_size_t(self.tim.size),
                    as_c(steps),
                    C.c_size_t(steps.size),
                    C.c_float(self.header.tsamp))
        error_check()
        
                
if __name__ == "__main__":
    z = AccelSearch("/work/jsc/lspitler/fake_data/J1906+0746.tim")
    print "First ten values: ", z.tim[0:10]
    z.seek_gpu()
    print "First ten values: ", z.tim[0:10]
        
    
            
    
