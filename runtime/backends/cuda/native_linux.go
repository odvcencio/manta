//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -L/usr/lib/x86_64-linux-gnu -lnvrtc -lcuda -lcublas
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <cuda.h>
#include <nvrtc.h>
#include <cublas_v2.h>

typedef struct {
	CUcontext ctx;
	CUdevice device;
	int major;
	int minor;
	cublasHandle_t blas;
} BarrCudaRuntime;

typedef struct {
	CUmodule module;
	CUfunction function;
} BarrCudaKernel;

static char* barr_dup_cstr(const char* s) {
	if (s == NULL) {
		return NULL;
	}
	size_t n = strlen(s) + 1;
	char* out = (char*)malloc(n);
	if (out == NULL) {
		return NULL;
	}
	memcpy(out, s, n);
	return out;
}

static char* barr_dup_format(const char* prefix, const char* value) {
	if (prefix == NULL) prefix = "error";
	if (value == NULL) value = "unknown";
	size_t n = strlen(prefix) + strlen(value) + 3;
	char* out = (char*)malloc(n);
	if (out == NULL) {
		return NULL;
	}
	snprintf(out, n, "%s: %s", prefix, value);
	return out;
}

static char* barr_dup_cu_error(const char* prefix, CUresult res) {
	const char* name = NULL;
	const char* detail = NULL;
	cuGetErrorName(res, &name);
	cuGetErrorString(res, &detail);
	if (name == NULL) name = "CU_UNKNOWN";
	if (detail == NULL) detail = "unknown";
	size_t n = strlen(prefix) + strlen(name) + strlen(detail) + 6;
	char* out = (char*)malloc(n);
	if (out == NULL) {
		return NULL;
	}
	snprintf(out, n, "%s: %s (%s)", prefix, name, detail);
	return out;
}

static char* barr_dup_nvrtc_error(const char* prefix, nvrtcResult res) {
	const char* detail = nvrtcGetErrorString(res);
	if (detail == NULL) detail = "unknown";
	return barr_dup_format(prefix, detail);
}

static const char* barr_cublas_status_name(cublasStatus_t status) {
	switch (status) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	default:
		return "CUBLAS_STATUS_UNKNOWN";
	}
}

static char* barr_dup_cublas_error(const char* prefix, cublasStatus_t status) {
	return barr_dup_format(prefix, barr_cublas_status_name(status));
}

static int barrCudaRuntimeCreate(BarrCudaRuntime** out, char** err) {
	BarrCudaRuntime* rt = NULL;
	CUdevice device = 0;
	CUcontext ctx = NULL;
	int major = 0;
	int minor = 0;
	cublasHandle_t blas = NULL;
	CUresult cuRes = cuInit(0);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuInit", cuRes);
		return 1;
	}
	cuRes = cuDeviceGet(&device, 0);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuDeviceGet", cuRes);
		return 1;
	}
	cuRes = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)", cuRes);
		return 1;
	}
	cuRes = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)", cuRes);
		return 1;
	}
	cuRes = cuCtxCreate(&ctx, NULL, 0, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxCreate", cuRes);
		return 1;
	}
	cuRes = cuCtxSetCurrent(ctx);
	if (cuRes != CUDA_SUCCESS) {
		cuCtxDestroy(ctx);
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cublasStatus_t blasRes = cublasCreate(&blas);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		cuCtxDestroy(ctx);
		*err = barr_dup_cublas_error("cublasCreate", blasRes);
		return 1;
	}
	rt = (BarrCudaRuntime*)malloc(sizeof(BarrCudaRuntime));
	if (rt == NULL) {
		cublasDestroy(blas);
		cuCtxDestroy(ctx);
		*err = barr_dup_format("malloc", "failed to allocate runtime");
		return 1;
	}
	rt->ctx = ctx;
	rt->device = device;
	rt->major = major;
	rt->minor = minor;
	rt->blas = blas;
	*out = rt;
	return 0;
}

static void barrCudaRuntimeDestroy(BarrCudaRuntime* rt) {
	if (rt == NULL) {
		return;
	}
	if (rt->ctx != NULL) {
		cuCtxSetCurrent(rt->ctx);
	}
	if (rt->blas != NULL) {
		cublasDestroy(rt->blas);
	}
	if (rt->ctx != NULL) {
		cuCtxDestroy(rt->ctx);
	}
	free(rt);
}

static int barrCudaCompileKernel(BarrCudaRuntime* rt, const char* src, const char* entry, BarrCudaKernel** out, char** log, char** err) {
	nvrtcProgram program;
	nvrtcResult nvRes;
	size_t logSize = 0;
	size_t ptxSize = 0;
	char arch[64];
	char* ptx = NULL;
	BarrCudaKernel* kernel = NULL;
	CUmodule module = NULL;
	CUfunction function = NULL;

	nvRes = nvrtcCreateProgram(&program, src, entry, 0, NULL, NULL);
	if (nvRes != NVRTC_SUCCESS) {
		*err = barr_dup_nvrtc_error("nvrtcCreateProgram", nvRes);
		return 1;
	}

	snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", rt->major, rt->minor);
	const char* opts[] = {"--std=c++14", arch};
	nvRes = nvrtcCompileProgram(program, 2, opts);
	nvrtcGetProgramLogSize(program, &logSize);
	if (logSize > 1) {
		*log = (char*)malloc(logSize);
		if (*log != NULL) {
			nvrtcGetProgramLog(program, *log);
		}
	}
	if (nvRes != NVRTC_SUCCESS) {
		*err = barr_dup_nvrtc_error("nvrtcCompileProgram", nvRes);
		nvrtcDestroyProgram(&program);
		return 1;
	}

	nvRes = nvrtcGetPTXSize(program, &ptxSize);
	if (nvRes != NVRTC_SUCCESS) {
		*err = barr_dup_nvrtc_error("nvrtcGetPTXSize", nvRes);
		nvrtcDestroyProgram(&program);
		return 1;
	}
	ptx = (char*)malloc(ptxSize);
	if (ptx == NULL) {
		*err = barr_dup_format("malloc", "failed to allocate PTX buffer");
		nvrtcDestroyProgram(&program);
		return 1;
	}
	nvRes = nvrtcGetPTX(program, ptx);
	nvrtcDestroyProgram(&program);
	if (nvRes != NVRTC_SUCCESS) {
		free(ptx);
		*err = barr_dup_nvrtc_error("nvrtcGetPTX", nvRes);
		return 1;
	}

	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		free(ptx);
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
	free(ptx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuModuleLoadDataEx", cuRes);
		return 1;
	}
	cuRes = cuModuleGetFunction(&function, module, entry);
	if (cuRes != CUDA_SUCCESS) {
		cuModuleUnload(module);
		*err = barr_dup_cu_error("cuModuleGetFunction", cuRes);
		return 1;
	}

	kernel = (BarrCudaKernel*)malloc(sizeof(BarrCudaKernel));
	if (kernel == NULL) {
		cuModuleUnload(module);
		*err = barr_dup_format("malloc", "failed to allocate kernel");
		return 1;
	}
	kernel->module = module;
	kernel->function = function;
	*out = kernel;
	return 0;
}

static void barrCudaKernelDestroy(BarrCudaKernel* kernel) {
	if (kernel == NULL) {
		return;
	}
	if (kernel->module != NULL) {
		cuModuleUnload(kernel->module);
	}
	free(kernel);
}

static int barrCudaMemAlloc(BarrCudaRuntime* rt, CUdeviceptr* out, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemAlloc(out, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuMemAlloc", cuRes);
		return 1;
	}
	return 0;
}

static int barrCudaMemFree(BarrCudaRuntime* rt, CUdeviceptr ptr, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemFree(ptr);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuMemFree", cuRes);
		return 1;
	}
	return 0;
}

static int barrCudaMemcpyHtoD(BarrCudaRuntime* rt, CUdeviceptr dst, const void* src, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemcpyHtoD(dst, src, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuMemcpyHtoD", cuRes);
		return 1;
	}
	return 0;
}

static int barrCudaMemcpyDtoH(BarrCudaRuntime* rt, void* dst, CUdeviceptr src, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemcpyDtoH(dst, src, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuMemcpyDtoH", cuRes);
		return 1;
	}
	return 0;
}

static int barrCudaReadFloat32(BarrCudaRuntime* rt, CUdeviceptr src, int elementIndex, float* out, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	CUdeviceptr addr = src + ((CUdeviceptr)elementIndex * sizeof(float));
	cuRes = cuMemcpyDtoH(out, addr, sizeof(float));
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuMemcpyDtoH", cuRes);
		return 1;
	}
	return 0;
}

static int barrCudaBlasIsamax(BarrCudaRuntime* rt, CUdeviceptr src, int elements, int* outIndex, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	const float* ptr = (const float*)(uintptr_t)src;
	cublasStatus_t blasRes = cublasIsamax(rt->blas, elements, ptr, 1, outIndex);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		*err = barr_dup_cublas_error("cublasIsamax", blasRes);
		return 1;
	}
	return 0;
}

static int barrCudaLaunch1D(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, void** args, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuLaunchKernel(kernel->function, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuLaunchKernel", cuRes);
		return 1;
	}
	cuRes = cuCtxSynchronize();
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static int barrCudaLaunchRowWise(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr in0, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&in0, &out0, &rows, &cols};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchElementWise(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int elements, char** err) {
	void* args[] = {&lhs, &rhs, &out0, &elements};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchUnary(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr in0, CUdeviceptr out0, int elements, char** err) {
	void* args[] = {&in0, &out0, &elements};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchScore(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr docs, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&query, &docs, &out0, &rows, &cols};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchOptimizerUpdate(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr param, CUdeviceptr mom1, CUdeviceptr mom2, CUdeviceptr grad, int elements, int mode, float learningRate, float weightDecay, float beta1, float beta2, float corr1, float corr2, float epsilon, float scale, char** err) {
	void* args[] = {&param, &mom1, &mom2, &grad, &elements, &mode, &learningRate, &weightDecay, &beta1, &beta2, &corr1, &corr2, &epsilon, &scale};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchSoftmaxBackwardRows(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr probs, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&gradOut, &probs, &out0, &rows, &cols};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchLayerNormBackwardRows(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr normalized, CUdeviceptr pre, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&gradOut, &normalized, &pre, &out0, &rows, &cols};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchQuantizeInPlace(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr data, int elements, float levels, float scale, char** err) {
	void* args[] = {&data, &elements, &levels, &scale};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchContrastiveScores(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr positive, CUdeviceptr queryNorms, CUdeviceptr positiveNorms, CUdeviceptr scores, int rows, int width, char** err) {
	void* args[] = {&query, &positive, &queryNorms, &positiveNorms, &scores, &rows, &width};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchInfoNCEScales(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr scores, CUdeviceptr scales, CUdeviceptr rowLoss, CUdeviceptr rowScore, int rows, float temperature, char** err) {
	void* args[] = {&scores, &scales, &rowLoss, &rowScore, &rows, &temperature};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaLaunchContrastiveGrad(BarrCudaRuntime* rt, BarrCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr positive, CUdeviceptr queryNorms, CUdeviceptr positiveNorms, CUdeviceptr scores, CUdeviceptr scales, CUdeviceptr queryGrads, CUdeviceptr positiveGrads, int rows, int width, char** err) {
	void* args[] = {&query, &positive, &queryNorms, &positiveNorms, &scores, &scales, &queryGrads, &positiveGrads, &rows, &width};
	return barrCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int barrCudaMatMulCublas(BarrCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	int rows = transposeLeft ? lhsCols : lhsRows;
	int inner = transposeLeft ? lhsRows : lhsCols;
	int rhsInner = transposeRight ? rhsCols : rhsRows;
	int cols = transposeRight ? rhsRows : rhsCols;
	if (inner != rhsInner) {
		*err = barr_dup_format("cublasSgemm", "shape mismatch");
		return 1;
	}
	const float alpha = 1.0f;
	const float beta = 0.0f;
	const float* lhsPtr = (const float*)(uintptr_t)lhs;
	const float* rhsPtr = (const float*)(uintptr_t)rhs;
	float* outPtr = (float*)(uintptr_t)out0;
	cublasStatus_t blasRes = cublasSgemm(
		rt->blas,
		transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N,
		transposeLeft ? CUBLAS_OP_T : CUBLAS_OP_N,
		cols,
		rows,
		inner,
		&alpha,
		rhsPtr,
		rhsCols,
		lhsPtr,
		lhsCols,
		&beta,
		outPtr,
		cols
	);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		*err = barr_dup_cublas_error("cublasSgemm", blasRes);
		return 1;
	}
	cuRes = cuCtxSynchronize();
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static int barrCudaMatMulCublasStridedBatched(BarrCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int batches, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	if (batches <= 0 || lhsRows <= 0 || lhsCols <= 0 || rhsRows <= 0 || rhsCols <= 0) {
		*err = barr_dup_format("cublasSgemmStridedBatched", "invalid shape");
		return 1;
	}
	int rows = transposeLeft ? lhsCols : lhsRows;
	int inner = transposeLeft ? lhsRows : lhsCols;
	int rhsInner = transposeRight ? rhsCols : rhsRows;
	int cols = transposeRight ? rhsRows : rhsCols;
	if (inner != rhsInner) {
		*err = barr_dup_format("cublasSgemmStridedBatched", "shape mismatch");
		return 1;
	}
	const float alpha = 1.0f;
	const float beta = 0.0f;
	const float* lhsPtr = (const float*)(uintptr_t)lhs;
	const float* rhsPtr = (const float*)(uintptr_t)rhs;
	float* outPtr = (float*)(uintptr_t)out0;
	long long int lhsStride = (long long int)lhsRows * (long long int)lhsCols;
	long long int rhsStride = (long long int)rhsRows * (long long int)rhsCols;
	long long int outStride = (long long int)rows * (long long int)cols;
	cublasStatus_t blasRes = cublasSgemmStridedBatched(
		rt->blas,
		transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N,
		transposeLeft ? CUBLAS_OP_T : CUBLAS_OP_N,
		cols,
		rows,
		inner,
		&alpha,
		rhsPtr,
		rhsCols,
		rhsStride,
		lhsPtr,
		lhsCols,
		lhsStride,
		&beta,
		outPtr,
		cols,
		outStride,
		batches
	);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		*err = barr_dup_cublas_error("cublasSgemmStridedBatched", blasRes);
		return 1;
	}
	cuRes = cuCtxSynchronize();
	if (cuRes != CUDA_SUCCESS) {
		*err = barr_dup_cu_error("cuCtxSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static void barrCudaFreeCString(char* s) {
	if (s != NULL) {
		free(s);
	}
}
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"time"
	"unsafe"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

const forwardQuantizeKernelSource = `
extern "C" __global__ void barr_forward_quantize_in_place(
    float* data,
    int elements,
    float levels,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements || scale == 0.0f || levels <= 0.0f) {
        return;
    }
    float q = roundf(data[idx] / scale);
    if (q > levels) {
        q = levels;
    }
    if (q < -levels) {
        q = -levels;
    }
    data[idx] = q * scale;
}
`

type deviceRuntime struct {
	ptr              *C.BarrCudaRuntime
	residentMatrices map[string]residentMatrix
	matMulScratch    map[string]deviceScratchBuffer
	quantizeKernel   *auxKernel
	matMulStats      backend.MatMulAcceleratorStats
}

type residentMatrix struct {
	ptr      C.CUdeviceptr
	rows     int
	cols     int
	elements int
}

type deviceScratchBuffer struct {
	ptr      C.CUdeviceptr
	elements int
}

type deviceKernel struct {
	ptr       *C.BarrCudaKernel
	shapeKind cudaShapeKind
}

type auxKernel struct {
	ptr *C.BarrCudaKernel
}

type cudaShapeKind int

const (
	cudaShapeUnsupported cudaShapeKind = iota
	cudaShapeRowWise
	cudaShapeElementWiseBinary
	cudaShapeElementWiseUnary
	cudaShapeRowScore
)

func newDeviceRuntime() (*deviceRuntime, error) {
	var rt *C.BarrCudaRuntime
	var errStr *C.char
	if C.barrCudaRuntimeCreate(&rt, &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	return &deviceRuntime{ptr: rt, residentMatrices: map[string]residentMatrix{}, matMulScratch: map[string]deviceScratchBuffer{}}, nil
}

func (rt *deviceRuntime) close() {
	if rt == nil || rt.ptr == nil {
		return
	}
	for name, resident := range rt.residentMatrices {
		_ = rt.freeBuffer(resident.ptr)
		delete(rt.residentMatrices, name)
	}
	for name, scratch := range rt.matMulScratch {
		_ = rt.freeBuffer(scratch.ptr)
		delete(rt.matMulScratch, name)
	}
	rt.destroyAuxKernel(rt.quantizeKernel)
	rt.quantizeKernel = nil
	C.barrCudaRuntimeDestroy(rt.ptr)
	rt.ptr = nil
}

func (rt *deviceRuntime) matMulStatsSnapshot() backend.MatMulAcceleratorStats {
	if rt == nil {
		return backend.MatMulAcceleratorStats{}
	}
	stats := rt.matMulStats
	stats.BoundMatrices = int64(len(rt.residentMatrices))
	return stats
}

func (rt *deviceRuntime) recordMatMulRun(start time.Time, uploadedBytes, downloadedBytes int64, boundLeft, boundRight bool) {
	if rt == nil {
		return
	}
	rt.matMulStats.RunCalls++
	if boundLeft {
		rt.matMulStats.BoundLeftCalls++
	}
	if boundRight {
		rt.matMulStats.BoundRightCalls++
	}
	rt.matMulStats.RunUploadedBytes += uploadedBytes
	rt.matMulStats.RunDownloadedBytes += downloadedBytes
	rt.matMulStats.RunNanos += time.Since(start).Nanoseconds()
}

func (rt *deviceRuntime) attachDeviceExecution(prog *backend.NativeKernelProgram, kernel barr.Kernel) error {
	shapeKind := classifyCUDAKernel(kernel)
	if shapeKind == cudaShapeUnsupported {
		prog.LaunchConfig["device_execution"] = false
		prog.LaunchConfig["execution_mode"] = "host_fallback"
		return nil
	}
	deviceKernel, err := rt.compileKernel(prog.Compiled, shapeKind)
	if err != nil {
		return err
	}
	prog.LaunchConfig["device_execution"] = true
	prog.LaunchConfig["execution_mode"] = "cuda_device"
	prog.LaunchConfig["launch_compiler"] = "nvrtc"
	prog.Run = func(inputs []*backend.Tensor) ([]*backend.Tensor, error) {
		return rt.runKernel(deviceKernel, kernel, prog, inputs)
	}
	return nil
}

func (rt *deviceRuntime) compileKernel(compiled backend.CompiledKernel, shapeKind cudaShapeKind) (*deviceKernel, error) {
	src := C.CString(compiled.Source)
	entry := C.CString(compiled.Entry)
	defer C.free(unsafe.Pointer(src))
	defer C.free(unsafe.Pointer(entry))

	var kernel *C.BarrCudaKernel
	var logStr *C.char
	var errStr *C.char
	rc := C.barrCudaCompileKernel(rt.ptr, src, entry, &kernel, &logStr, &errStr)
	logText := cStringValue(logStr)
	if rc != 0 {
		err := cStringError(errStr)
		if logText != "" {
			return nil, fmt.Errorf("%w\n%s", err, logText)
		}
		return nil, err
	}
	return &deviceKernel{ptr: kernel, shapeKind: shapeKind}, nil
}

func (rt *deviceRuntime) compileAuxKernel(source, entry string) (*auxKernel, error) {
	src := C.CString(source)
	cEntry := C.CString(entry)
	defer C.free(unsafe.Pointer(src))
	defer C.free(unsafe.Pointer(cEntry))

	var kernel *C.BarrCudaKernel
	var logStr *C.char
	var errStr *C.char
	rc := C.barrCudaCompileKernel(rt.ptr, src, cEntry, &kernel, &logStr, &errStr)
	logText := cStringValue(logStr)
	if rc != 0 {
		err := cStringError(errStr)
		if logText != "" {
			return nil, fmt.Errorf("%w\n%s", err, logText)
		}
		return nil, err
	}
	return &auxKernel{ptr: kernel}, nil
}

func (rt *deviceRuntime) destroyAuxKernel(kernel *auxKernel) {
	if kernel == nil || kernel.ptr == nil {
		return
	}
	C.barrCudaKernelDestroy(kernel.ptr)
}

func (rt *deviceRuntime) runKernel(deviceKernel *deviceKernel, kernel barr.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	switch deviceKernel.shapeKind {
	case cudaShapeRowWise:
		return rt.runRowWiseKernel(deviceKernel, kernel, prog, inputs)
	case cudaShapeElementWiseBinary:
		return rt.runElementWiseKernel(deviceKernel, kernel, prog, inputs)
	case cudaShapeElementWiseUnary:
		return rt.runUnaryKernel(deviceKernel, kernel, prog, inputs)
	case cudaShapeRowScore:
		return rt.runScoreKernel(deviceKernel, kernel, prog, inputs)
	default:
		return prog.Run(inputs)
	}
}

func (rt *deviceRuntime) runRowWiseKernel(deviceKernel *deviceKernel, kernel barr.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("kernel %q expected 1 input for row-wise launch, got %d", kernel.Name, len(inputs))
	}
	in := inputs[0]
	if in == nil || (len(in.Shape) != 2 && len(in.Shape) != 3) {
		return nil, fmt.Errorf("kernel %q expected rank-2 or rank-3 input", kernel.Name)
	}
	outShape := append([]int(nil), in.Shape...)
	rows := in.Shape[0]
	cols := in.Shape[len(in.Shape)-1]
	if len(in.Shape) == 3 {
		rows = in.Shape[0] * in.Shape[1]
	}
	if rows == 0 || cols == 0 {
		return []*backend.Tensor{newOutputTensor(kernel, outShape, make([]float32, rows*cols))}, nil
	}
	inBuf, err := rt.uploadFloat32(in.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(inBuf)
	outHost := make([]float32, rows*cols)
	outBuf, err := rt.allocFloat32(len(outHost))
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	rowsArg := C.int(rows)
	colsArg := C.int(cols)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((rows + int(block) - 1) / int(block))
	if err := rt.launchRowWise(deviceKernel.ptr, grid, block, inBuf, outBuf, rowsArg, colsArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, outShape, outHost)}, nil
}

func (rt *deviceRuntime) runElementWiseKernel(deviceKernel *deviceKernel, kernel barr.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("kernel %q expected 2 inputs for element-wise launch, got %d", kernel.Name, len(inputs))
	}
	lhs := inputs[0]
	rhs := inputs[1]
	if lhs == nil || rhs == nil || !lhs.EqualShape(rhs) {
		return nil, fmt.Errorf("kernel %q expected matching input shapes", kernel.Name)
	}
	elements := lhs.Elements()
	lhsBuf, err := rt.uploadFloat32(lhs.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(lhsBuf)
	rhsBuf, err := rt.uploadFloat32(rhs.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(rhsBuf)
	outHost := make([]float32, elements)
	outBuf, err := rt.allocFloat32(elements)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	elementsArg := C.int(elements)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((elements + int(block) - 1) / int(block))
	if err := rt.launchElementWise(deviceKernel.ptr, grid, block, lhsBuf, rhsBuf, outBuf, elementsArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, append([]int(nil), lhs.Shape...), outHost)}, nil
}

func (rt *deviceRuntime) runUnaryKernel(deviceKernel *deviceKernel, kernel barr.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("kernel %q expected 1 input for unary launch, got %d", kernel.Name, len(inputs))
	}
	in := inputs[0]
	if in == nil {
		return nil, fmt.Errorf("kernel %q expected non-nil input", kernel.Name)
	}
	elements := in.Elements()
	inBuf, err := rt.uploadFloat32(in.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(inBuf)
	outHost := make([]float32, elements)
	outBuf, err := rt.allocFloat32(elements)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	elementsArg := C.int(elements)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((elements + int(block) - 1) / int(block))
	if err := rt.launchUnary(deviceKernel.ptr, grid, block, inBuf, outBuf, elementsArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, append([]int(nil), in.Shape...), outHost)}, nil
}

func (rt *deviceRuntime) runScoreKernel(deviceKernel *deviceKernel, kernel barr.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("kernel %q expected 2 inputs for score launch, got %d", kernel.Name, len(inputs))
	}
	query := inputs[0]
	docs := inputs[1]
	if query == nil || docs == nil || len(query.Shape) != 1 || len(docs.Shape) != 2 || query.Shape[0] != docs.Shape[1] {
		return nil, fmt.Errorf("kernel %q expected query rank-1 and docs rank-2 inputs", kernel.Name)
	}
	rows := docs.Shape[0]
	cols := docs.Shape[1]
	queryBuf, err := rt.uploadFloat32(query.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(queryBuf)
	docsBuf, err := rt.uploadFloat32(docs.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(docsBuf)
	outHost := make([]float32, rows)
	outBuf, err := rt.allocFloat32(rows)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	rowsArg := C.int(rows)
	colsArg := C.int(cols)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((rows + int(block) - 1) / int(block))
	if err := rt.launchScore(deviceKernel.ptr, grid, block, queryBuf, docsBuf, outBuf, rowsArg, colsArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, []int{rows}, outHost)}, nil
}

func classifyCUDAKernel(kernel barr.Kernel) cudaShapeKind {
	if len(kernel.Body) < 2 {
		return cudaShapeUnsupported
	}
	switch kernel.Body[0].Op {
	case "normalize", "rmsnorm", "layernorm", "softmax", "rope":
		return cudaShapeRowWise
	case "binary_add", "binary_sub", "binary_mul", "binary_div":
		return cudaShapeElementWiseBinary
	case "dequant", "gelu":
		return cudaShapeElementWiseUnary
	case "dot", "cosine", "l2_distance":
		return cudaShapeRowScore
	default:
		return cudaShapeUnsupported
	}
}

func newOutputTensor(kernel barr.Kernel, shape []int, data []float32) *backend.Tensor {
	dtype := "f16"
	if len(kernel.Outputs) > 0 && kernel.Outputs[0].Type.Tensor != nil && kernel.Outputs[0].Type.Tensor.DType != "" {
		dtype = kernel.Outputs[0].Type.Tensor.DType
	}
	switch dtype {
	case "f32":
		return &backend.Tensor{DType: "f32", Shape: append([]int(nil), shape...), F32: data}
	default:
		return &backend.Tensor{DType: "f16", Shape: append([]int(nil), shape...), F32: data}
	}
}

func (rt *deviceRuntime) allocFloat32(elements int) (C.CUdeviceptr, error) {
	var ptr C.CUdeviceptr
	var errStr *C.char
	bytes := C.size_t(elements * 4)
	if C.barrCudaMemAlloc(rt.ptr, &ptr, bytes, &errStr) != 0 {
		return 0, cStringError(errStr)
	}
	return ptr, nil
}

func (rt *deviceRuntime) uploadFloat32(data []float32) (C.CUdeviceptr, error) {
	ptr, err := rt.allocFloat32(len(data))
	if err != nil {
		return 0, err
	}
	if err := rt.copyFloat32ToBuffer(ptr, data); err != nil {
		_ = rt.freeBuffer(ptr)
		return 0, err
	}
	return ptr, nil
}

func (rt *deviceRuntime) matMulScratchFloat32(name string, elements int) (C.CUdeviceptr, error) {
	if elements == 0 {
		return 0, nil
	}
	if rt.matMulScratch == nil {
		rt.matMulScratch = map[string]deviceScratchBuffer{}
	}
	if scratch, ok := rt.matMulScratch[name]; ok && scratch.elements >= elements {
		return scratch.ptr, nil
	}
	if scratch, ok := rt.matMulScratch[name]; ok {
		_ = rt.freeBuffer(scratch.ptr)
		delete(rt.matMulScratch, name)
	}
	ptr, err := rt.allocFloat32(elements)
	if err != nil {
		return 0, err
	}
	rt.matMulScratch[name] = deviceScratchBuffer{ptr: ptr, elements: elements}
	return ptr, nil
}

func (rt *deviceRuntime) uploadMatMulScratchFloat32(name string, data []float32) (C.CUdeviceptr, error) {
	ptr, err := rt.matMulScratchFloat32(name, len(data))
	if err != nil {
		return 0, err
	}
	if len(data) == 0 {
		return ptr, nil
	}
	if err := rt.copyFloat32ToBuffer(ptr, data); err != nil {
		return 0, err
	}
	return ptr, nil
}

func (rt *deviceRuntime) copyFloat32ToBuffer(ptr C.CUdeviceptr, data []float32) error {
	var src unsafe.Pointer
	if len(data) > 0 {
		src = unsafe.Pointer(&data[0])
	}
	var errStr *C.char
	if C.barrCudaMemcpyHtoD(rt.ptr, ptr, src, C.size_t(len(data)*4), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) downloadFloat32(dst []float32, src C.CUdeviceptr) error {
	if len(dst) == 0 {
		return nil
	}
	var errStr *C.char
	if C.barrCudaMemcpyDtoH(rt.ptr, unsafe.Pointer(&dst[0]), src, C.size_t(len(dst)*4), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) readFloat32At(src C.CUdeviceptr, elementIndex int) (float32, error) {
	var out C.float
	var errStr *C.char
	if C.barrCudaReadFloat32(rt.ptr, src, C.int(elementIndex), &out, &errStr) != 0 {
		return 0, cStringError(errStr)
	}
	return float32(out), nil
}

func (rt *deviceRuntime) maxAbsFloat32(src C.CUdeviceptr, elements int) (float32, error) {
	if elements <= 0 {
		return 0, nil
	}
	var index C.int
	var errStr *C.char
	if C.barrCudaBlasIsamax(rt.ptr, src, C.int(elements), &index, &errStr) != 0 {
		return 0, cStringError(errStr)
	}
	if index <= 0 {
		return 0, nil
	}
	value, err := rt.readFloat32At(src, int(index)-1)
	if err != nil {
		return 0, err
	}
	return float32(math.Abs(float64(value))), nil
}

func (rt *deviceRuntime) ensureQuantizeKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.quantizeKernel != nil {
		return rt.quantizeKernel, nil
	}
	kernel, err := rt.compileAuxKernel(forwardQuantizeKernelSource, "barr_forward_quantize_in_place")
	if err != nil {
		return nil, err
	}
	rt.quantizeKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) quantizeBufferInPlace(ptr C.CUdeviceptr, elements, bits int) (bool, error) {
	if rt == nil || rt.ptr == nil || elements == 0 || bits <= 0 {
		return false, nil
	}
	maxAbs, err := rt.maxAbsFloat32(ptr, elements)
	if err != nil {
		return false, err
	}
	if maxAbs == 0 {
		return false, nil
	}
	levelsInt := (1 << uint(bits-1)) - 1
	if levelsInt <= 0 {
		return false, nil
	}
	levels := float32(levelsInt)
	scale := maxAbs / levels
	if scale == 0 {
		return false, nil
	}
	kernel, err := rt.ensureQuantizeKernel()
	if err != nil {
		return false, err
	}
	block := uint(128)
	grid := uint((elements + int(block) - 1) / int(block))
	var errStr *C.char
	if C.barrCudaLaunchQuantizeInPlace(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), ptr, C.int(elements), C.float(levels), C.float(scale), &errStr) != 0 {
		return false, cStringError(errStr)
	}
	return true, nil
}

func quantBitsForPreparedTensor(tensor *backend.Tensor) int {
	if tensor == nil {
		return 0
	}
	switch tensor.DType {
	case "q4":
		return 4
	case "q8":
		return 8
	default:
		return 0
	}
}

func (rt *deviceRuntime) freeBuffer(ptr C.CUdeviceptr) error {
	if ptr == 0 {
		return nil
	}
	var errStr *C.char
	if C.barrCudaMemFree(rt.ptr, ptr, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchRowWise(kernel *C.BarrCudaKernel, grid, block C.uint, in0, out0 C.CUdeviceptr, rows, cols C.int) error {
	var errStr *C.char
	if C.barrCudaLaunchRowWise(rt.ptr, kernel, grid, block, in0, out0, rows, cols, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchElementWise(kernel *C.BarrCudaKernel, grid, block C.uint, lhs, rhs, out0 C.CUdeviceptr, elements C.int) error {
	var errStr *C.char
	if C.barrCudaLaunchElementWise(rt.ptr, kernel, grid, block, lhs, rhs, out0, elements, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxElementWise(kernel *auxKernel, grid, block uint, lhs, rhs, out0 C.CUdeviceptr, elements int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary elementwise kernel is not initialized")
	}
	return rt.launchElementWise(kernel.ptr, C.uint(grid), C.uint(block), lhs, rhs, out0, C.int(elements))
}

func (rt *deviceRuntime) launchUnary(kernel *C.BarrCudaKernel, grid, block C.uint, in0, out0 C.CUdeviceptr, elements C.int) error {
	var errStr *C.char
	if C.barrCudaLaunchUnary(rt.ptr, kernel, grid, block, in0, out0, elements, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchScore(kernel *C.BarrCudaKernel, grid, block C.uint, query, docs, out0 C.CUdeviceptr, rows, cols C.int) error {
	var errStr *C.char
	if C.barrCudaLaunchScore(rt.ptr, kernel, grid, block, query, docs, out0, rows, cols, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchOptimizerUpdate(kernel *auxKernel, grid, block uint, param, mom1, mom2, grad C.CUdeviceptr, elements, mode int, learningRate, weightDecay, beta1, beta2, corr1, corr2, epsilon, scale float32) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda optimizer kernel is not initialized")
	}
	var errStr *C.char
	if C.barrCudaLaunchOptimizerUpdate(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		param,
		mom1,
		mom2,
		grad,
		C.int(elements),
		C.int(mode),
		C.float(learningRate),
		C.float(weightDecay),
		C.float(beta1),
		C.float(beta2),
		C.float(corr1),
		C.float(corr2),
		C.float(epsilon),
		C.float(scale),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxSoftmaxBackwardRows(kernel *auxKernel, grid, block uint, gradOut, probs, out0 C.CUdeviceptr, rows, cols int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary softmax backward kernel is not initialized")
	}
	var errStr *C.char
	if C.barrCudaLaunchSoftmaxBackwardRows(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), gradOut, probs, out0, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxLayerNormBackwardRows(kernel *auxKernel, grid, block uint, gradOut, normalized, pre, out0 C.CUdeviceptr, rows, cols int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary layernorm backward kernel is not initialized")
	}
	var errStr *C.char
	if C.barrCudaLaunchLayerNormBackwardRows(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), gradOut, normalized, pre, out0, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxContrastiveScores(kernel *auxKernel, grid, block uint, query, positive, queryNorms, positiveNorms, scores C.CUdeviceptr, rows, width int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary contrastive score kernel is not initialized")
	}
	var errStr *C.char
	if C.barrCudaLaunchContrastiveScores(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), query, positive, queryNorms, positiveNorms, scores, C.int(rows), C.int(width), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxInfoNCEScales(kernel *auxKernel, grid, block uint, scores, scales, rowLoss, rowScore C.CUdeviceptr, rows int, temperature float32) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary infonce scale kernel is not initialized")
	}
	var errStr *C.char
	if C.barrCudaLaunchInfoNCEScales(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), scores, scales, rowLoss, rowScore, C.int(rows), C.float(temperature), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxContrastiveGrad(kernel *auxKernel, grid, block uint, query, positive, queryNorms, positiveNorms, scores, scales, queryGrads, positiveGrads C.CUdeviceptr, rows, width int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary contrastive grad kernel is not initialized")
	}
	var errStr *C.char
	if C.barrCudaLaunchContrastiveGrad(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), query, positive, queryNorms, positiveNorms, scores, scales, queryGrads, positiveGrads, C.int(rows), C.int(width), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) matMulCublas(lhs, rhs, out0 C.CUdeviceptr, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool) error {
	var errStr *C.char
	var left C.int
	var right C.int
	if transposeLeft {
		left = 1
	}
	if transposeRight {
		right = 1
	}
	if C.barrCudaMatMulCublas(rt.ptr, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, left, right, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) matMulCublasStridedBatched(lhs, rhs, out0 C.CUdeviceptr, batches, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool) error {
	var errStr *C.char
	var left C.int
	var right C.int
	if transposeLeft {
		left = 1
	}
	if transposeRight {
		right = 1
	}
	if C.barrCudaMatMulCublasStridedBatched(rt.ptr, lhs, rhs, out0, batches, lhsRows, lhsCols, rhsRows, rhsCols, left, right, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func cStringValue(value *C.char) string {
	if value == nil {
		return ""
	}
	out := C.GoString(value)
	C.barrCudaFreeCString(value)
	return out
}

func cStringError(value *C.char) error {
	if value == nil {
		return fmt.Errorf("cuda error")
	}
	defer C.barrCudaFreeCString(value)
	return errors.New(C.GoString(value))
}

func (rt *deviceRuntime) runMatMul(inputs []*backend.Tensor, outputType barr.ValueType) (backend.StepDispatchResult, error) {
	return rt.runMatMulWithTranspose(inputs, outputType, false, false)
}

func (rt *deviceRuntime) bindMatMulRight(name string, tensor *backend.Tensor) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	if name == "" {
		return fmt.Errorf("cuda matmul binding name is required")
	}
	if tensor == nil || len(tensor.Shape) != 2 {
		return fmt.Errorf("cuda matmul binding %q must be a rank-2 tensor", name)
	}
	elements := len(tensor.F32)
	quantBits := quantBitsForPreparedTensor(tensor)
	start := time.Now()
	uploadedBytes := int64(elements * 4)
	if resident, ok := rt.residentMatrices[name]; ok {
		if resident.elements == elements {
			if err := rt.copyFloat32ToBuffer(resident.ptr, tensor.F32); err != nil {
				return err
			}
			quantizeStart := time.Now()
			ran, err := rt.quantizeBufferInPlace(resident.ptr, elements, quantBits)
			if err != nil {
				return err
			}
			if ran {
				rt.matMulStats.QuantizePasses++
				rt.matMulStats.QuantizedBytes += uploadedBytes
				rt.matMulStats.QuantizeNanos += time.Since(quantizeStart).Nanoseconds()
			}
			resident.rows = tensor.Shape[0]
			resident.cols = tensor.Shape[1]
			rt.residentMatrices[name] = resident
			rt.matMulStats.BindCalls++
			rt.matMulStats.UploadedBytes += uploadedBytes
			rt.matMulStats.BindNanos += time.Since(start).Nanoseconds()
			rt.matMulStats.BoundMatrices = int64(len(rt.residentMatrices))
			return nil
		}
		_ = rt.freeBuffer(resident.ptr)
		delete(rt.residentMatrices, name)
	}
	ptr, err := rt.uploadFloat32(tensor.F32)
	if err != nil {
		return err
	}
	quantizeStart := time.Now()
	ran, err := rt.quantizeBufferInPlace(ptr, elements, quantBits)
	if err != nil {
		_ = rt.freeBuffer(ptr)
		return err
	}
	if ran {
		rt.matMulStats.QuantizePasses++
		rt.matMulStats.QuantizedBytes += uploadedBytes
		rt.matMulStats.QuantizeNanos += time.Since(quantizeStart).Nanoseconds()
	}
	rt.residentMatrices[name] = residentMatrix{
		ptr:      ptr,
		rows:     tensor.Shape[0],
		cols:     tensor.Shape[1],
		elements: elements,
	}
	rt.matMulStats.BindCalls++
	rt.matMulStats.UploadedBytes += uploadedBytes
	rt.matMulStats.BindNanos += time.Since(start).Nanoseconds()
	rt.matMulStats.BoundMatrices = int64(len(rt.residentMatrices))
	return nil
}

func (rt *deviceRuntime) unbindMatMulRight(name string) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	resident, ok := rt.residentMatrices[name]
	if !ok {
		return nil
	}
	if err := rt.freeBuffer(resident.ptr); err != nil {
		return err
	}
	delete(rt.residentMatrices, name)
	rt.matMulStats.BoundMatrices = int64(len(rt.residentMatrices))
	return nil
}

func (rt *deviceRuntime) runMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if lhs == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul lhs is nil")
	}
	resident, ok := rt.residentMatrices[rightName]
	if !ok {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul binding %q is not resident", rightName)
	}
	rhsTensor := &backend.Tensor{DType: "f32", Shape: []int{resident.rows, resident.cols}}
	batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(lhs, rhsTensor)
	if transposeLeft || transposeRight {
		batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(lhs, rhsTensor, transposeLeft, transposeRight)
	}
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	if rhsBatched {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda cached rhs matmul does not support batched rhs")
	}
	runStart := time.Now()
	runUploadedBytes := int64(len(lhs.F32) * 4)
	runDownloadedBytes := int64(batches * rows * cols * 4)
	compiled := cudaBuiltinMatMulCompiledKernel()
	outHost := make([]float32, batches*rows*cols)
	lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", lhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	lhsRows, lhsCols := batches*rows, inner
	rhsRows, rhsCols := inner, cols
	if transposeLeft || transposeRight {
		lhsRows, lhsCols = matrixShape(lhs)
		rhsRows, rhsCols = resident.rows, resident.cols
	}
	if err := rt.matMulCublas(lhsBuf, resident.ptr, outBuf, C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), transposeLeft, transposeRight); err != nil {
		return backend.StepDispatchResult{}, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	rt.recordMatMulRun(runStart, runUploadedBytes, runDownloadedBytes, false, true)
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_native",
			"execution_mode":   "cuda_device",
			"device_execution": true,
			"launch_api":       "cublasSgemm",
			"launch_compiler":  "cublas",
			"backend_library":  "cublas",
			"transpose_left":   transposeLeft,
			"transpose_right":  transposeRight,
			"rhs_binding":      rightName,
			"rhs_residency":    "device_resident",
		},
	}, nil
}

func (rt *deviceRuntime) runMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if rhs == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul rhs is nil")
	}
	resident, ok := rt.residentMatrices[leftName]
	if !ok {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul binding %q is not resident", leftName)
	}
	lhsTensor := &backend.Tensor{DType: "f32", Shape: []int{resident.rows, resident.cols}}
	batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(lhsTensor, rhs)
	if transposeLeft || transposeRight {
		batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(lhsTensor, rhs, transposeLeft, transposeRight)
	}
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	if rhsBatched || batches != 1 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda cached lhs matmul does not support batched inputs")
	}
	runStart := time.Now()
	runUploadedBytes := int64(len(rhs.F32) * 4)
	runDownloadedBytes := int64(batches * rows * cols * 4)
	compiled := cudaBuiltinMatMulCompiledKernel()
	outHost := make([]float32, batches*rows*cols)
	rhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_rhs", rhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	lhsRows, lhsCols := resident.rows, resident.cols
	rhsRows, rhsCols := batches*inner, cols
	if transposeLeft || transposeRight {
		lhsRows, lhsCols = resident.rows, resident.cols
		rhsRows, rhsCols = matrixShape(rhs)
	}
	if err := rt.matMulCublas(resident.ptr, rhsBuf, outBuf, C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), transposeLeft, transposeRight); err != nil {
		return backend.StepDispatchResult{}, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	rt.recordMatMulRun(runStart, runUploadedBytes, runDownloadedBytes, true, false)
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_native",
			"execution_mode":   "cuda_device",
			"device_execution": true,
			"launch_api":       "cublasSgemm",
			"launch_compiler":  "cublas",
			"backend_library":  "cublas",
			"transpose_left":   transposeLeft,
			"transpose_right":  transposeRight,
			"lhs_binding":      leftName,
			"lhs_residency":    "device_resident",
		},
	}, nil
}

func (rt *deviceRuntime) runMatMulWithTranspose(inputs []*backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if len(inputs) != 2 {
		return backend.StepDispatchResult{}, fmt.Errorf("matmul expects 2 inputs, got %d", len(inputs))
	}
	batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(inputs[0], inputs[1])
	if transposeLeft || transposeRight {
		batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(inputs[0], inputs[1], transposeLeft, transposeRight)
	}
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	runStart := time.Now()
	runUploadedBytes := int64((len(inputs[0].F32) + len(inputs[1].F32)) * 4)
	runDownloadedBytes := int64(batches * rows * cols * 4)
	compiled := cudaBuiltinMatMulCompiledKernel()
	launchAPI := "cublasSgemm"
	outHost := make([]float32, batches*rows*cols)
	if !rhsBatched {
		lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", inputs[0].F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		rhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_rhs", inputs[1].F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(inputs[0])
			rhsRows, rhsCols = matrixShape(inputs[1])
		}
		if err := rt.matMulCublas(lhsBuf, rhsBuf, outBuf, C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), transposeLeft, transposeRight); err != nil {
			return backend.StepDispatchResult{}, err
		}
		if err := rt.downloadFloat32(outHost, outBuf); err != nil {
			return backend.StepDispatchResult{}, err
		}
	} else {
		lhsRows, lhsCols := rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(inputs[0])
			rhsRows, rhsCols = matrixShape(inputs[1])
		}
		lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", inputs[0].F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		rhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_rhs", inputs[1].F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		if err := rt.matMulCublasStridedBatched(lhsBuf, rhsBuf, outBuf, C.int(batches), C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), transposeLeft, transposeRight); err != nil {
			return backend.StepDispatchResult{}, err
		}
		if err := rt.downloadFloat32(outHost, outBuf); err != nil {
			return backend.StepDispatchResult{}, err
		}
		launchAPI = "cublasSgemmStridedBatched"
	}
	rt.recordMatMulRun(runStart, runUploadedBytes, runDownloadedBytes, false, false)
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_native",
			"execution_mode":   "cuda_device",
			"device_execution": true,
			"launch_api":       launchAPI,
			"launch_compiler":  "cublas",
			"backend_library":  "cublas",
			"transpose_left":   transposeLeft,
			"transpose_right":  transposeRight,
		},
	}, nil
}

func cudaBuiltinMatMulCompiledKernel() backend.CompiledKernel {
	return backend.CompiledKernel{
		Name:       "__builtin_matmul",
		Backend:    barr.BackendCUDA,
		Entry:      "cublas_sgemm",
		Source:     "library:cublas_sgemm",
		SourceHash: "library:cublas_sgemm",
		Meta: map[string]string{
			"library":      "cublas",
			"vector_width": "backend_selected",
			"memory":       "device_local",
		},
	}
}

func matmulLayout(lhs, rhs *backend.Tensor) (batches, rows, inner, cols int, rhsBatched bool, outShape []int, err error) {
	if lhs == nil || rhs == nil {
		return 0, 0, 0, 0, false, nil, fmt.Errorf("nil matmul input")
	}
	switch len(lhs.Shape) {
	case 2:
		if len(rhs.Shape) != 2 {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul rank-2 lhs requires rank-2 rhs tensor")
		}
		cols = rhs.Shape[1]
		if lhs.Shape[1] != rhs.Shape[0] {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
		}
		return 1, lhs.Shape[0], lhs.Shape[1], cols, false, []int{lhs.Shape[0], rhs.Shape[1]}, nil
	case 3:
		switch len(rhs.Shape) {
		case 2:
			cols = rhs.Shape[1]
			if lhs.Shape[2] != rhs.Shape[0] {
				return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
			}
			return lhs.Shape[0], lhs.Shape[1], lhs.Shape[2], cols, false, []int{lhs.Shape[0], lhs.Shape[1], rhs.Shape[1]}, nil
		case 3:
			cols = rhs.Shape[2]
			if lhs.Shape[0] != rhs.Shape[0] || lhs.Shape[2] != rhs.Shape[1] {
				return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
			}
			return lhs.Shape[0], lhs.Shape[1], lhs.Shape[2], cols, true, []int{lhs.Shape[0], lhs.Shape[1], rhs.Shape[2]}, nil
		default:
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul rank-3 lhs requires rank-2 or rank-3 rhs tensor")
		}
	default:
		return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul expects rank-2 or rank-3 lhs tensor")
	}
}

func matmulLayoutWithTranspose(lhs, rhs *backend.Tensor, transposeLeft, transposeRight bool) (batches, rows, inner, cols int, rhsBatched bool, outShape []int, err error) {
	if lhs == nil || rhs == nil {
		return 0, 0, 0, 0, false, nil, fmt.Errorf("nil matmul input")
	}
	switch len(lhs.Shape) {
	case 2:
		if len(rhs.Shape) != 2 {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("transposed rank-2 lhs requires rank-2 rhs tensor")
		}
		rows = lhs.Shape[0]
		inner = lhs.Shape[1]
		if transposeLeft {
			rows, inner = inner, rows
		}
		rhsInner := rhs.Shape[0]
		cols = rhs.Shape[1]
		if transposeRight {
			rhsInner, cols = cols, rhsInner
		}
		if inner != rhsInner {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v with transpose_left=%t transpose_right=%t", lhs.Shape, rhs.Shape, transposeLeft, transposeRight)
		}
		return 1, rows, inner, cols, false, []int{rows, cols}, nil
	case 3:
		if len(rhs.Shape) != 3 {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("transposed rank-3 lhs requires rank-3 rhs tensor")
		}
		if lhs.Shape[0] != rhs.Shape[0] {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul batch mismatch %v x %v", lhs.Shape, rhs.Shape)
		}
		rows = lhs.Shape[1]
		inner = lhs.Shape[2]
		if transposeLeft {
			rows, inner = inner, rows
		}
		rhsInner := rhs.Shape[1]
		cols = rhs.Shape[2]
		if transposeRight {
			rhsInner, cols = cols, rhsInner
		}
		if inner != rhsInner {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v with transpose_left=%t transpose_right=%t", lhs.Shape, rhs.Shape, transposeLeft, transposeRight)
		}
		return lhs.Shape[0], rows, inner, cols, true, []int{lhs.Shape[0], rows, cols}, nil
	default:
		return 0, 0, 0, 0, false, nil, fmt.Errorf("transposed matmul expects rank-2 or rank-3 lhs tensor")
	}
}

func matrixShape(t *backend.Tensor) (rows, cols int) {
	if t == nil || len(t.Shape) < 2 {
		return 0, 0
	}
	return t.Shape[len(t.Shape)-2], t.Shape[len(t.Shape)-1]
}

func newStepOutputTensor(outputType barr.ValueType, shape []int, data []float32) *backend.Tensor {
	dtype := "f16"
	if outputType.Tensor != nil && outputType.Tensor.DType != "" {
		dtype = outputType.Tensor.DType
	}
	switch dtype {
	case "f32":
		return &backend.Tensor{DType: "f32", Shape: append([]int(nil), shape...), F32: data}
	default:
		return &backend.Tensor{DType: "f16", Shape: append([]int(nil), shape...), F32: data}
	}
}

func firstIntValue(value any, fallback int) int {
	switch v := value.(type) {
	case int:
		return v
	case int32:
		return int(v)
	case int64:
		return int(v)
	case float64:
		return int(v)
	case string:
		var n int
		_, err := fmt.Sscanf(v, "%d", &n)
		if err == nil && n > 0 {
			return n
		}
	}
	return fallback
}
