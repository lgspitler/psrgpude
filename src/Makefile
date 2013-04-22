all: libcuseek.so

libcuseek.so: cuseek.o
	icpc cuseek.o -O3 -shared -fPIC -I${CUDA_HOME}/include/ -L${CUDA_HOME}/lib64/ -lcuda -lcufft -lcudart -o libcuseek.so

cuseek.o:
	nvcc -O3 -c --machine 64 -arch=sm_20 -Xcompiler -fpic cuseek.cu

clean: 
	rm -rf *.o *.so  
