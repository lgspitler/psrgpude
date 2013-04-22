#include <cufft.h>

/**Based on common.cpp code from GPGPU workshop**/

const char* cufftGetErrorString(cufftResult_t rc) {

    switch(rc) {
        case CUFFT_SUCCESS:
            return "No errors";
        case CUFFT_INVALID_PLAN:
            return "Invalid plan handle";
        case CUFFT_ALLOC_FAILED:
            return "Failed to allocate CPU or GPU memory";
        case CUFFT_INTERNAL_ERROR:
            return "Driver or internal CUFFT library error";
        case CUFFT_EXEC_FAILED:
            return "Execution of FFT on GPU failed";
        case CUFFT_SETUP_FAILED:
            return "CUFFT library failed to initialize";
        case CUFFT_INVALID_SIZE:
            return "Invalid tranform size";
        default:
            return "Unspecified Error";
    }
}   
