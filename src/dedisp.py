import ctypes as C
import numpy as np
from sigpyproc.Readers import FilReader
from numpy.ctypeslib import as_array,as_ctypes  
lib = C.CDLL("libdedisp.so")

error_dict={
0:"DEDISP_NO_ERROR",
1:"DEDISP_MEM_ALLOC_FAILED",
2:"DEDISP_MEM_COPY_FAILED",
3:"DEDISP_NCHANS_EXCEEDS_LIMIT",
4:"DEDISP_INVALID_PLAN",
5:"DEDISP_INVALID_POINTER",
6:"DEDISP_INVALID_STRIDE",
7:"DEDISP_NO_DM_LIST_SET",
8:"DEDISP_TOO_FEW_NSAMPS",
9:"DEDISP_INVALID_FLAG_COMBINATION",
10:"DEDISP_UNSUPPORTED_IN_NBITS",
11:"DEDISP_UNSUPPORTED_OUT_NBITS",
12:"DEDISP_INVALID_DEVICE_INDEX",
13:"DEDISP_DEVICE_ALREADY_SET",
14:"DEDISP_PRIOR_GPU_ERROR",
15:"DEDISP_INTERNAL_GPU_ERROR",
16:"DEDISP_UNKNOWN_ERROR"
}

class DedispError(Exception):
    pass
        
def error_check(result):
    if result is not 0:
        raise DedispError(error_dict[result])
        

class Dedisp(FilReader):
    def __init__(self,filename,dev=0):
        super(Dedisp,self).__init__(filename)
        self.set_device(dev)
        self.plan = self.create_plan()
    
    def set_device(self,dev):
        func = lib.dedisp_set_device
        error = func(C.c_int(dev));
        error_check(error)

    def create_plan(self):
        plan_pointer = C.c_void_p()
        func = lib.dedisp_create_plan
        error = func(C.byref(plan_pointer),
                     C.c_int(self.header.nchans),
                     C.c_float(self.header.tsamp),
                     C.c_float(self.header.fch1),
                     C.c_float(abs(self.header.foff)))
        error_check(error)
        return plan_pointer

    def create_dm_list(self, dm_start, dm_end, pulse_width=0.1, dm_tol=1.25):
        func = lib.dedisp_generate_dm_list
        error = func(self.plan,
                     C.c_float(dm_start),
                     C.c_float(dm_end),
                     C.c_float(pulse_width),
                     C.c_float(dm_tol))
        error_check(error)
        
    def get_dm_count(self):
        func = lib.dedisp_get_dm_count
        func.restype = C.c_size_t
        return func(self.plan)
    
    def get_max_delay(self):
        func = lib.dedisp_get_max_delay
        func.restype = C.c_size_t
        return func(self.plan)
    
    def get_dm_list(self):
        ndms = self.get_dm_count()
        func = lib.dedisp_get_dm_list
        c_float_p = C.POINTER(C.c_float)
        func.restype = c_float_p
        array_pointer = C.cast(func(self.plan),c_float_p)
        return as_array(array_pointer,shape=(ndms,)).copy()
    
    def dedisperse_gpu(self,output_dir=".",out_bits=32,gulp=160000):
        ndm = self.get_dm_count()
        delay = self.get_max_delay()
        if gulp-delay < 2*delay:
            gulp = 2*delay

        if out_bits == 32:
            dtype = "float32"
        elif out_bits == 8:
            dtype = "ubyte"
        else:
            raise ValueError("out_bits must be 8 or 32")

        outsamps = gulp-delay

        print outsamps,out_bits,dtype
        

        dms = self.get_dm_list()
        out_files = []
        changes = {"refdm" :0, "nchans":1, "nbits" :out_bits}
        basename = self.header.basename.split("/")[-1]
        for dm in dms:
            changes["refdm"] = dm
            filename = "%s/%s_DM%08.2f.tim"%(output_dir,basename,dm)
            out_files.append(self.header.prepOutfile(filename,changes,nbits=out_bits))
            
        out_size = outsamps * ndm * out_bits/8
        output = np.empty(out_size,dtype=dtype)
        func = lib.dedisp_execute
        for nsamps,ii,data in self.readPlan(gulp=gulp,skipback=delay):
            error = func(self.plan,
                         C.c_size_t(nsamps),
                         as_ctypes(data),
                         C.c_size_t(self.header.nbits),
                         as_ctypes(output),
                         C.c_size_t(out_bits),
                         C.c_int(0));
            error_check(error)
            for ii,out_file in enumerate(out_files):
                out_file.cwrite(output[ii*outsamps:(ii+1)*outsamps])
        
        for out_file in out_files:
           out_file.close()
                               
def main():
    import sys
    x = Dedisp("/work/zam/ebarr/FullMidLatPointing/6572_0001_01_8bit.fil")
    x.create_dm_list(0.0,3000.0,40,1.25)
    x.dedisperse_gpu(output_dir="/work/zam/ebarr/TimDumps",out_bits=32,gulp=int(sys.argv[1]))

if __name__ == "__main__":
    main()
        
        
        

      
    
