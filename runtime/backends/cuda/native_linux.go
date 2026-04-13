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
	int primary_ctx;
	cublasHandle_t blas;
} MantaCudaRuntime;

typedef struct {
	CUmodule module;
	CUfunction function;
} MantaCudaKernel;

static char* manta_dup_cstr(const char* s) {
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

static char* manta_dup_format(const char* prefix, const char* value) {
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

static char* manta_dup_cu_error(const char* prefix, CUresult res) {
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

static char* manta_dup_nvrtc_error(const char* prefix, nvrtcResult res) {
	const char* detail = nvrtcGetErrorString(res);
	if (detail == NULL) detail = "unknown";
	return manta_dup_format(prefix, detail);
}

static const char* manta_cublas_status_name(cublasStatus_t status) {
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

static char* manta_dup_cublas_error(const char* prefix, cublasStatus_t status) {
	return manta_dup_format(prefix, manta_cublas_status_name(status));
}

static int mantaCudaRuntimeCreate(MantaCudaRuntime** out, char** err) {
	MantaCudaRuntime* rt = NULL;
	CUdevice device = 0;
	CUcontext ctx = NULL;
	int major = 0;
	int minor = 0;
	cublasHandle_t blas = NULL;
	CUresult cuRes = cuInit(0);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuInit", cuRes);
		return 1;
	}
	cuRes = cuDeviceGet(&device, 0);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuDeviceGet", cuRes);
		return 1;
	}
	cuRes = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)", cuRes);
		return 1;
	}
	cuRes = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)", cuRes);
		return 1;
	}
	// Repeated runtime loads must not allocate independent CUDA contexts.
	// The primary context is shared by the process and avoids suite-wide VRAM exhaustion.
	cuRes = cuDevicePrimaryCtxRetain(&ctx, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuDevicePrimaryCtxRetain", cuRes);
		return 1;
	}
	cuRes = cuCtxSetCurrent(ctx);
	if (cuRes != CUDA_SUCCESS) {
		cuDevicePrimaryCtxRelease(device);
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cublasStatus_t blasRes = cublasCreate(&blas);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		cuDevicePrimaryCtxRelease(device);
		*err = manta_dup_cublas_error("cublasCreate", blasRes);
		return 1;
	}
	rt = (MantaCudaRuntime*)malloc(sizeof(MantaCudaRuntime));
	if (rt == NULL) {
		cublasDestroy(blas);
		cuDevicePrimaryCtxRelease(device);
		*err = manta_dup_format("malloc", "failed to allocate runtime");
		return 1;
	}
	rt->ctx = ctx;
	rt->device = device;
	rt->major = major;
	rt->minor = minor;
	rt->primary_ctx = 1;
	rt->blas = blas;
	*out = rt;
	return 0;
}

static void mantaCudaRuntimeDestroy(MantaCudaRuntime* rt) {
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
		if (rt->primary_ctx) {
			cuDevicePrimaryCtxRelease(rt->device);
		} else {
			cuCtxDestroy(rt->ctx);
		}
	}
	free(rt);
}

static int mantaCudaCompileKernel(MantaCudaRuntime* rt, const char* src, const char* entry, MantaCudaKernel** out, char** log, char** err) {
	nvrtcProgram program;
	nvrtcResult nvRes;
	size_t logSize = 0;
	size_t ptxSize = 0;
	char arch[64];
	char* ptx = NULL;
	MantaCudaKernel* kernel = NULL;
	CUmodule module = NULL;
	CUfunction function = NULL;

	nvRes = nvrtcCreateProgram(&program, src, entry, 0, NULL, NULL);
	if (nvRes != NVRTC_SUCCESS) {
		*err = manta_dup_nvrtc_error("nvrtcCreateProgram", nvRes);
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
		*err = manta_dup_nvrtc_error("nvrtcCompileProgram", nvRes);
		nvrtcDestroyProgram(&program);
		return 1;
	}

	nvRes = nvrtcGetPTXSize(program, &ptxSize);
	if (nvRes != NVRTC_SUCCESS) {
		*err = manta_dup_nvrtc_error("nvrtcGetPTXSize", nvRes);
		nvrtcDestroyProgram(&program);
		return 1;
	}
	ptx = (char*)malloc(ptxSize);
	if (ptx == NULL) {
		*err = manta_dup_format("malloc", "failed to allocate PTX buffer");
		nvrtcDestroyProgram(&program);
		return 1;
	}
	nvRes = nvrtcGetPTX(program, ptx);
	nvrtcDestroyProgram(&program);
	if (nvRes != NVRTC_SUCCESS) {
		free(ptx);
		*err = manta_dup_nvrtc_error("nvrtcGetPTX", nvRes);
		return 1;
	}

	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		free(ptx);
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
	free(ptx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuModuleLoadDataEx", cuRes);
		return 1;
	}
	cuRes = cuModuleGetFunction(&function, module, entry);
	if (cuRes != CUDA_SUCCESS) {
		cuModuleUnload(module);
		*err = manta_dup_cu_error("cuModuleGetFunction", cuRes);
		return 1;
	}

	kernel = (MantaCudaKernel*)malloc(sizeof(MantaCudaKernel));
	if (kernel == NULL) {
		cuModuleUnload(module);
		*err = manta_dup_format("malloc", "failed to allocate kernel");
		return 1;
	}
	kernel->module = module;
	kernel->function = function;
	*out = kernel;
	return 0;
}

static void mantaCudaKernelDestroy(MantaCudaKernel* kernel) {
	if (kernel == NULL) {
		return;
	}
	if (kernel->module != NULL) {
		cuModuleUnload(kernel->module);
	}
	free(kernel);
}

static int mantaCudaMemAlloc(MantaCudaRuntime* rt, CUdeviceptr* out, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemAlloc(out, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemAlloc", cuRes);
		return 1;
	}
	return 0;
}

static int mantaCudaMemFree(MantaCudaRuntime* rt, CUdeviceptr ptr, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemFree(ptr);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemFree", cuRes);
		return 1;
	}
	return 0;
}

static int mantaCudaMemcpyHtoD(MantaCudaRuntime* rt, CUdeviceptr dst, const void* src, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemcpyHtoD(dst, src, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemcpyHtoD", cuRes);
		return 1;
	}
	return 0;
}

static int mantaCudaMemcpyDtoH(MantaCudaRuntime* rt, void* dst, CUdeviceptr src, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemcpyDtoH(dst, src, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemcpyDtoH", cuRes);
		return 1;
	}
	return 0;
}

static int mantaCudaSynchronize(MantaCudaRuntime* rt, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuCtxSynchronize();
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static int mantaCudaReadFloat32(MantaCudaRuntime* rt, CUdeviceptr src, int elementIndex, float* out, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	CUdeviceptr addr = src + ((CUdeviceptr)elementIndex * sizeof(float));
	cuRes = cuMemcpyDtoH(out, addr, sizeof(float));
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemcpyDtoH", cuRes);
		return 1;
	}
	return 0;
}

static int mantaCudaBlasIsamax(MantaCudaRuntime* rt, CUdeviceptr src, int elements, int* outIndex, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	const float* ptr = (const float*)(uintptr_t)src;
	cublasStatus_t blasRes = cublasIsamax(rt->blas, elements, ptr, 1, outIndex);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		*err = manta_dup_cublas_error("cublasIsamax", blasRes);
		return 1;
	}
	return 0;
}

static int mantaCudaLaunch1D(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, void** args, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuLaunchKernel(kernel->function, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuLaunchKernel", cuRes);
		return 1;
	}
	cuRes = cuCtxSynchronize();
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static int mantaCudaLaunchRowWise(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr in0, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&in0, &out0, &rows, &cols};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchElementWise(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int elements, char** err) {
	void* args[] = {&lhs, &rhs, &out0, &elements};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchUnary(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr in0, CUdeviceptr out0, int elements, char** err) {
	void* args[] = {&in0, &out0, &elements};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchScore(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr docs, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&query, &docs, &out0, &rows, &cols};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchOptimizerUpdate(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr param, CUdeviceptr mom1, CUdeviceptr mom2, CUdeviceptr grad, int elements, int mode, float learningRate, float weightDecay, float beta1, float beta2, float corr1, float corr2, float epsilon, float scale, char** err) {
	void* args[] = {&param, &mom1, &mom2, &grad, &elements, &mode, &learningRate, &weightDecay, &beta1, &beta2, &corr1, &corr2, &epsilon, &scale};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchSoftmaxBackwardRows(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr probs, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&gradOut, &probs, &out0, &rows, &cols};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchLayerNormBackwardRows(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr normalized, CUdeviceptr pre, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&gradOut, &normalized, &pre, &out0, &rows, &cols};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchQuantizeInPlace(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr data, int elements, float levels, float scale, char** err) {
	void* args[] = {&data, &elements, &levels, &scale};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchGDN(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr input, CUdeviceptr beta, CUdeviceptr gamma, CUdeviceptr out0, int elements, int channels, int height, int width, int inverse, char** err) {
	void* args[] = {&input, &beta, &gamma, &out0, &elements, &channels, &height, &width, &inverse};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchConv2D(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr input, CUdeviceptr weight, CUdeviceptr bias, CUdeviceptr out0, int elements, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int inPerGroup, int outPerGroup, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int hasBias, char** err) {
	void* args[] = {&input, &weight, &bias, &out0, &elements, &inChannels, &inHeight, &inWidth, &outChannels, &outHeight, &outWidth, &inPerGroup, &outPerGroup, &kernelH, &kernelW, &strideH, &strideW, &padH, &padW, &dilationH, &dilationW, &hasBias};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchConv2DTranspose(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr input, CUdeviceptr weight, CUdeviceptr bias, CUdeviceptr out0, int elements, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int inPerGroup, int outPerGroup, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int hasBias, char** err) {
	void* args[] = {&input, &weight, &bias, &out0, &elements, &inChannels, &inHeight, &inWidth, &outChannels, &outHeight, &outWidth, &inPerGroup, &outPerGroup, &kernelH, &kernelW, &strideH, &strideW, &padH, &padW, &dilationH, &dilationW, &hasBias};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchMSEPartials(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr partials, int elements, char** err) {
	void* args[] = {&lhs, &rhs, &partials, &elements};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchMSSSIMPartials(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr partials, int elements, char** err) {
	void* args[] = {&lhs, &rhs, &partials, &elements};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchScalarSum(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr values, CUdeviceptr out0, int count, char** err) {
	void* args[] = {&values, &out0, &count};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchRDLoss(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr distortion, CUdeviceptr rate, CUdeviceptr out0, float lambda, char** err) {
	void* args[] = {&distortion, &rate, &out0, &lambda};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchCrossEntropyPartials(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr codes, CUdeviceptr logits, CUdeviceptr partials, int elements, int mode, int layout, int levels, int bits, int codeRank, int codeN, int codeC, int codeH, int codeW, int logitsLen, int logitsN, int logitsC, int logitsH, int logitsW, int sigmaMode, char** err) {
	void* args[] = {&codes, &logits, &partials, &elements, &mode, &layout, &levels, &bits, &codeRank, &codeN, &codeC, &codeH, &codeW, &logitsLen, &logitsN, &logitsC, &logitsH, &logitsW, &sigmaMode};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchContrastiveScores(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr positive, CUdeviceptr queryNorms, CUdeviceptr positiveNorms, CUdeviceptr scores, int rows, int width, char** err) {
	void* args[] = {&query, &positive, &queryNorms, &positiveNorms, &scores, &rows, &width};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchInfoNCEScales(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr scores, CUdeviceptr scales, CUdeviceptr rowLoss, CUdeviceptr rowScore, int rows, float temperature, char** err) {
	void* args[] = {&scores, &scales, &rowLoss, &rowScore, &rows, &temperature};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaLaunchContrastiveGrad(MantaCudaRuntime* rt, MantaCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr positive, CUdeviceptr queryNorms, CUdeviceptr positiveNorms, CUdeviceptr scores, CUdeviceptr scales, CUdeviceptr queryGrads, CUdeviceptr positiveGrads, int rows, int width, char** err) {
	void* args[] = {&query, &positive, &queryNorms, &positiveNorms, &scores, &scales, &queryGrads, &positiveGrads, &rows, &width};
	return mantaCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int mantaCudaMatMulCublasWithBetaNoSync(MantaCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, float betaValue, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	int rows = transposeLeft ? lhsCols : lhsRows;
	int inner = transposeLeft ? lhsRows : lhsCols;
	int rhsInner = transposeRight ? rhsCols : rhsRows;
	int cols = transposeRight ? rhsRows : rhsCols;
	if (inner != rhsInner) {
		*err = manta_dup_format("cublasSgemm", "shape mismatch");
		return 1;
	}
	const float alpha = 1.0f;
	const float beta = betaValue;
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
		*err = manta_dup_cublas_error("cublasSgemm", blasRes);
		return 1;
	}
	return 0;
}

static int mantaCudaMatMulCublasWithBeta(MantaCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, float betaValue, char** err) {
	if (mantaCudaMatMulCublasWithBetaNoSync(rt, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight, betaValue, err) != 0) {
		return 1;
	}
	return mantaCudaSynchronize(rt, err);
}

static int mantaCudaMatMulCublas(MantaCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, char** err) {
	return mantaCudaMatMulCublasWithBeta(rt, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight, 0.0f, err);
}

static int mantaCudaMatMulCublasStridedBatched(MantaCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int batches, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	if (batches <= 0 || lhsRows <= 0 || lhsCols <= 0 || rhsRows <= 0 || rhsCols <= 0) {
		*err = manta_dup_format("cublasSgemmStridedBatched", "invalid shape");
		return 1;
	}
	int rows = transposeLeft ? lhsCols : lhsRows;
	int inner = transposeLeft ? lhsRows : lhsCols;
	int rhsInner = transposeRight ? rhsCols : rhsRows;
	int cols = transposeRight ? rhsRows : rhsCols;
	if (inner != rhsInner) {
		*err = manta_dup_format("cublasSgemmStridedBatched", "shape mismatch");
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
		*err = manta_dup_cublas_error("cublasSgemmStridedBatched", blasRes);
		return 1;
	}
	cuRes = cuCtxSynchronize();
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static void mantaCudaFreeCString(char* s) {
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
	"os"
	"time"
	"unsafe"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

const forwardQuantizeKernelSource = `
extern "C" __global__ void manta_forward_quantize_in_place(
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

const gdnKernelSource = `
extern "C" __global__ void manta_gdn_forward(
    const float* input,
    const float* beta,
    const float* gamma,
    float* out,
    int elements,
    int channels,
    int height,
    int width,
    int inverse
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    int spatial = height * width;
    int c = (idx / spatial) % channels;
    int n = idx / (channels * spatial);
    int hw = idx % spatial;
    float sum = beta[c];
    for (int j = 0; j < channels; ++j) {
        int input_idx = ((n * channels + j) * spatial) + hw;
        float v = input[input_idx];
        sum += gamma[c * channels + j] * v * v;
    }
    if (sum < 1.0e-12f) {
        sum = 1.0e-12f;
    }
    float scale = sqrtf(sum);
    if (inverse != 0) {
        out[idx] = input[idx] * scale;
    } else {
        out[idx] = input[idx] / scale;
    }
}
`

const convKernelSource = `
extern "C" __global__ void manta_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* out,
    int elements,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int inPerGroup,
    int outPerGroup,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int hasBias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    int ox = idx % outWidth;
    int rem = idx / outWidth;
    int oy = rem % outHeight;
    rem /= outHeight;
    int oc = rem % outChannels;
    int n = rem / outChannels;
    int group = oc / outPerGroup;
    float sum = hasBias ? bias[oc] : 0.0f;
    for (int icg = 0; icg < inPerGroup; ++icg) {
        int ic = group * inPerGroup + icg;
        for (int ky = 0; ky < kernelH; ++ky) {
            int iy = oy * strideH + ky * dilationH - padH;
            if (iy < 0 || iy >= inHeight) {
                continue;
            }
            for (int kx = 0; kx < kernelW; ++kx) {
                int ix = ox * strideW + kx * dilationW - padW;
                if (ix < 0 || ix >= inWidth) {
                    continue;
                }
                int inputIdx = ((n * inChannels + ic) * inHeight + iy) * inWidth + ix;
                int weightIdx = ((oc * inPerGroup + icg) * kernelH + ky) * kernelW + kx;
                sum += input[inputIdx] * weight[weightIdx];
            }
        }
    }
    out[idx] = sum;
}

extern "C" __global__ void manta_conv2d_transpose_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* out,
    int elements,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int inPerGroup,
    int outPerGroup,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int hasBias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    int ox = idx % outWidth;
    int rem = idx / outWidth;
    int oy = rem % outHeight;
    rem /= outHeight;
    int oc = rem % outChannels;
    int n = rem / outChannels;
    int group = oc / outPerGroup;
    int ocg = oc - group * outPerGroup;
    float sum = hasBias ? bias[oc] : 0.0f;
    for (int icg = 0; icg < inPerGroup; ++icg) {
        int ic = group * inPerGroup + icg;
        for (int ky = 0; ky < kernelH; ++ky) {
            int yNumerator = oy + padH - ky * dilationH;
            if (yNumerator % strideH != 0) {
                continue;
            }
            int iy = yNumerator / strideH;
            if (iy < 0 || iy >= inHeight) {
                continue;
            }
            for (int kx = 0; kx < kernelW; ++kx) {
                int xNumerator = ox + padW - kx * dilationW;
                if (xNumerator % strideW != 0) {
                    continue;
                }
                int ix = xNumerator / strideW;
                if (ix < 0 || ix >= inWidth) {
                    continue;
                }
                int inputIdx = ((n * inChannels + ic) * inHeight + iy) * inWidth + ix;
                int weightIdx = ((ic * outPerGroup + ocg) * kernelH + ky) * kernelW + kx;
                sum += input[inputIdx] * weight[weightIdx];
            }
        }
    }
    out[idx] = sum;
}
`

const mseLossKernelSource = `
extern "C" __global__ void manta_mse_partials(
    const float* lhs,
    const float* rhs,
    float* partials,
    int elements
) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = idx; i < elements; i += stride) {
        float diff = lhs[i] - rhs[i];
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}
`

const msssimLossKernelSource = `
extern "C" __global__ void manta_msssim_partials(
    const float* lhs,
    const float* rhs,
    float* partials,
    int elements
) {
    __shared__ float sumA[256];
    __shared__ float sumB[256];
    __shared__ float sumAA[256];
    __shared__ float sumBB[256];
    __shared__ float sumAB[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float a = 0.0f;
    float b = 0.0f;
    float aa = 0.0f;
    float bb = 0.0f;
    float ab = 0.0f;
    for (int i = idx; i < elements; i += stride) {
        float x = lhs[i];
        float y = rhs[i];
        a += x;
        b += y;
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    sumA[tid] = a;
    sumB[tid] = b;
    sumAA[tid] = aa;
    sumBB[tid] = bb;
    sumAB[tid] = ab;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sumA[tid] += sumA[tid + offset];
            sumB[tid] += sumB[tid + offset];
            sumAA[tid] += sumAA[tid + offset];
            sumBB[tid] += sumBB[tid + offset];
            sumAB[tid] += sumAB[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int base = blockIdx.x * 5;
        partials[base + 0] = sumA[0];
        partials[base + 1] = sumB[0];
        partials[base + 2] = sumAA[0];
        partials[base + 3] = sumBB[0];
        partials[base + 4] = sumAB[0];
    }
}
`

const scalarLossKernelSource = `
extern "C" __global__ void manta_scalar_sum(
    const float* values,
    float* out,
    int count
) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < count; i += blockDim.x) {
        sum += values[i];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = shared[0];
    }
}

extern "C" __global__ void manta_rd_loss(
    const float* distortion,
    const float* rate,
    float* out,
    float lambda
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = distortion[0] + lambda * rate[0];
    }
}
`

const crossEntropyKernelSource = `
static __device__ int manta_clamp_int(int value, int lo, int hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static __device__ float manta_neg_log2(float p) {
    if (!(p >= 1.0e-12f)) {
        p = 1.0e-12f;
    }
    return -logf(p) * 1.4426950408889634f;
}

static __device__ float manta_softmax_probability_strided(const float* values, int base, int step, int count, int idx) {
    float maxv = values[base];
    for (int i = 1; i < count; ++i) {
        float v = values[base + i * step];
        if (v > maxv) {
            maxv = v;
        }
    }
    float sum = 0.0f;
    float target = 0.0f;
    for (int i = 0; i < count; ++i) {
        float v = expf(values[base + i * step] - maxv);
        if (i == idx) {
            target = v;
        }
        sum += v;
    }
    if (sum == 0.0f) {
        return 1.0f / (float)count;
    }
    return target / sum;
}

static __device__ void manta_unpack_offset4(int offset, int channels, int height, int width, int* n, int* c, int* h, int* w) {
    int spatial = height * width;
    *w = offset % width;
    int rem = offset / width;
    *h = rem % height;
    rem /= height;
    *c = rem % channels;
    *n = rem / channels;
}

static __device__ float manta_categorical_probability(
    const float* logits,
    int layout,
    int offset,
    int idx,
    int levels,
    int codeC,
    int codeH,
    int codeW,
    int logitsLen,
    int logitsC,
    int logitsH,
    int logitsW
) {
    if (layout == 0 || logits == 0 || logitsLen == 0) {
        return 1.0f / (float)levels;
    }
    if (layout == 1) {
        return manta_softmax_probability_strided(logits, 0, 1, levels, idx);
    }
    if (layout == 2) {
        return manta_softmax_probability_strided(logits, offset * levels, 1, levels, idx);
    }
    if (layout == 3) {
        int n, c, h, w;
        manta_unpack_offset4(offset, codeC, codeH, codeW, &n, &c, &h, &w);
        int baseChannel = c * levels;
        int base = ((n * logitsC + baseChannel) * logitsH + h) * logitsW + w;
        return manta_softmax_probability_strided(logits, base, logitsH * logitsW, levels, idx);
    }
    float p = 1.0f / (1.0f + expf(-logits[offset % logitsLen]));
    if (idx == 0) {
        return 1.0f - p;
    }
    int denom = levels - 1;
    if (denom < 1) {
        denom = 1;
    }
    return p / (float)denom;
}

static __device__ float manta_bit_probability(
    const float* logits,
    int layout,
    int offset,
    int bit,
    int bitValue,
    int bits,
    int codeC,
    int codeH,
    int codeW,
    int logitsLen,
    int logitsC,
    int logitsH,
    int logitsW
) {
    if (layout == 0 || logits == 0 || logitsLen == 0) {
        return 0.5f;
    }
    if (layout == 1) {
        return manta_softmax_probability_strided(logits, bit * 2, 1, 2, bitValue);
    }
    if (layout == 2) {
        return manta_softmax_probability_strided(logits, (offset * bits + bit) * 2, 1, 2, bitValue);
    }
    if (layout == 3) {
        int n, c, h, w;
        manta_unpack_offset4(offset, codeC, codeH, codeW, &n, &c, &h, &w);
        int ch = c * bits * 2 + bit * 2;
        int base = ((n * logitsC + ch) * logitsH + h) * logitsW + w;
        return manta_softmax_probability_strided(logits, base, logitsH * logitsW, 2, bitValue);
    }
    float p = 1.0f / (1.0f + expf(-logits[(offset * bits + bit) % logitsLen]));
    if (bitValue == 1) {
        return p;
    }
    return 1.0f - p;
}

static __device__ float manta_normal_cdf(float x) {
    return 0.5f * (1.0f + erff(x * 0.7071067811865475f));
}

static __device__ float manta_norm_sigma(float raw, int sigmaMode) {
    if (sigmaMode == 1) {
        if (raw > 32.0f) {
            return raw + 1.0e-6f;
        }
        return log1pf(expf(raw)) + 1.0e-6f;
    }
    if (sigmaMode == 2) {
        return expf(raw);
    }
    return raw;
}

static __device__ float manta_log_normal_loss(
    const float* codes,
    const float* params,
    int offset,
    int codeH,
    int codeW,
    int paramsC,
    int paramsH,
    int paramsW,
    int sigmaMode
) {
    int x = offset % codeW;
    int rem = offset / codeW;
    int y = rem % codeH;
    int n = rem / codeH;
    float mu = params[((n * paramsC + 0) * paramsH + y) * paramsW + x];
    float sigma = manta_norm_sigma(params[((n * paramsC + 1) * paramsH + y) * paramsW + x], sigmaMode);
    int sym = manta_clamp_int((int)roundf(codes[offset]), 0, 255);
    float span = 32.0f;
    float p;
    if (sym <= 0) {
        float hi = -16.0f + (0.5f / 255.0f) * span;
        p = manta_normal_cdf((hi - mu) / sigma);
    } else if (sym >= 255) {
        float lo = -16.0f + (254.5f / 255.0f) * span;
        p = 1.0f - manta_normal_cdf((lo - mu) / sigma);
    } else {
        float lo = -16.0f + (((float)sym - 0.5f) / 255.0f) * span;
        float hi = -16.0f + (((float)sym + 0.5f) / 255.0f) * span;
        p = manta_normal_cdf((hi - mu) / sigma) - manta_normal_cdf((lo - mu) / sigma);
    }
    return manta_neg_log2(p);
}

extern "C" __global__ void manta_cross_entropy_partials(
    const float* codes,
    const float* logits,
    float* partials,
    int elements,
    int mode,
    int layout,
    int levels,
    int bits,
    int codeRank,
    int codeN,
    int codeC,
    int codeH,
    int codeW,
    int logitsLen,
    int logitsN,
    int logitsC,
    int logitsH,
    int logitsW,
    int sigmaMode
) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = idx; i < elements; i += stride) {
        if (mode == 2) {
            sum += manta_log_normal_loss(codes, logits, i, codeH, codeW, logitsC, logitsH, logitsW, sigmaMode);
        } else if (mode == 1) {
            int maxSymbol = (1 << bits) - 1;
            int symbol = manta_clamp_int((int)roundf(codes[i]), 0, maxSymbol);
            for (int bit = 0; bit < bits; ++bit) {
                int shift = bits - 1 - bit;
                int bitValue = (symbol >> shift) & 1;
                float p = manta_bit_probability(logits, layout, i, bit, bitValue, bits, codeC, codeH, codeW, logitsLen, logitsC, logitsH, logitsW);
                sum += manta_neg_log2(p);
            }
        } else {
            int symbol = manta_clamp_int((int)roundf(codes[i]), 0, levels - 1);
            float p = manta_categorical_probability(logits, layout, i, symbol, levels, codeC, codeH, codeW, logitsLen, logitsC, logitsH, logitsW);
            sum += manta_neg_log2(p);
        }
    }
    shared[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}
`

type deviceRuntime struct {
	ptr                *C.MantaCudaRuntime
	residentMatrices   map[string]residentMatrix
	matMulScratch      map[string]deviceScratchBuffer
	quantizeKernel     *auxKernel
	gdnKernel          *auxKernel
	conv2DKernel       *auxKernel
	conv2DTransKernel  *auxKernel
	mseLossKernel      *auxKernel
	msssimLossKernel   *auxKernel
	scalarAddKernel    *auxKernel
	rdLossKernel       *auxKernel
	crossEntropyKernel *auxKernel
	matMulStats        backend.MatMulAcceleratorStats
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
	ptr       *C.MantaCudaKernel
	shapeKind cudaShapeKind
}

type auxKernel struct {
	ptr *C.MantaCudaKernel
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
	var rt *C.MantaCudaRuntime
	var errStr *C.char
	if C.mantaCudaRuntimeCreate(&rt, &errStr) != 0 {
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
	rt.destroyAuxKernel(rt.gdnKernel)
	rt.gdnKernel = nil
	rt.destroyAuxKernel(rt.conv2DKernel)
	rt.conv2DKernel = nil
	rt.destroyAuxKernel(rt.conv2DTransKernel)
	rt.conv2DTransKernel = nil
	rt.destroyAuxKernel(rt.mseLossKernel)
	rt.mseLossKernel = nil
	rt.destroyAuxKernel(rt.msssimLossKernel)
	rt.msssimLossKernel = nil
	rt.destroyAuxKernel(rt.scalarAddKernel)
	rt.scalarAddKernel = nil
	rt.destroyAuxKernel(rt.rdLossKernel)
	rt.rdLossKernel = nil
	rt.destroyAuxKernel(rt.crossEntropyKernel)
	rt.crossEntropyKernel = nil
	C.mantaCudaRuntimeDestroy(rt.ptr)
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

func (rt *deviceRuntime) attachDeviceExecution(prog *backend.NativeKernelProgram, kernel mantaartifact.Kernel) error {
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

	var kernel *C.MantaCudaKernel
	var logStr *C.char
	var errStr *C.char
	rc := C.mantaCudaCompileKernel(rt.ptr, src, entry, &kernel, &logStr, &errStr)
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

	var kernel *C.MantaCudaKernel
	var logStr *C.char
	var errStr *C.char
	rc := C.mantaCudaCompileKernel(rt.ptr, src, cEntry, &kernel, &logStr, &errStr)
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
	C.mantaCudaKernelDestroy(kernel.ptr)
}

func (rt *deviceRuntime) runKernel(deviceKernel *deviceKernel, kernel mantaartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
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

func (rt *deviceRuntime) runRowWiseKernel(deviceKernel *deviceKernel, kernel mantaartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
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

func (rt *deviceRuntime) runElementWiseKernel(deviceKernel *deviceKernel, kernel mantaartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
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

func (rt *deviceRuntime) runUnaryKernel(deviceKernel *deviceKernel, kernel mantaartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
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

func (rt *deviceRuntime) runScoreKernel(deviceKernel *deviceKernel, kernel mantaartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
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

func classifyCUDAKernel(kernel mantaartifact.Kernel) cudaShapeKind {
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

func newOutputTensor(kernel mantaartifact.Kernel, shape []int, data []float32) *backend.Tensor {
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
	if C.mantaCudaMemAlloc(rt.ptr, &ptr, bytes, &errStr) != 0 {
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
	if C.mantaCudaMemcpyHtoD(rt.ptr, ptr, src, C.size_t(len(data)*4), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) downloadFloat32(dst []float32, src C.CUdeviceptr) error {
	if len(dst) == 0 {
		return nil
	}
	var errStr *C.char
	if C.mantaCudaMemcpyDtoH(rt.ptr, unsafe.Pointer(&dst[0]), src, C.size_t(len(dst)*4), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) readFloat32At(src C.CUdeviceptr, elementIndex int) (float32, error) {
	var out C.float
	var errStr *C.char
	if C.mantaCudaReadFloat32(rt.ptr, src, C.int(elementIndex), &out, &errStr) != 0 {
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
	if C.mantaCudaBlasIsamax(rt.ptr, src, C.int(elements), &index, &errStr) != 0 {
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
	kernel, err := rt.compileAuxKernel(forwardQuantizeKernelSource, "manta_forward_quantize_in_place")
	if err != nil {
		return nil, err
	}
	rt.quantizeKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureGDNKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.gdnKernel != nil {
		return rt.gdnKernel, nil
	}
	kernel, err := rt.compileAuxKernel(gdnKernelSource, "manta_gdn_forward")
	if err != nil {
		return nil, err
	}
	rt.gdnKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureConv2DKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.conv2DKernel != nil {
		return rt.conv2DKernel, nil
	}
	kernel, err := rt.compileAuxKernel(convKernelSource, "manta_conv2d_forward")
	if err != nil {
		return nil, err
	}
	rt.conv2DKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureConv2DTransposeKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.conv2DTransKernel != nil {
		return rt.conv2DTransKernel, nil
	}
	kernel, err := rt.compileAuxKernel(convKernelSource, "manta_conv2d_transpose_forward")
	if err != nil {
		return nil, err
	}
	rt.conv2DTransKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureMSELossKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.mseLossKernel != nil {
		return rt.mseLossKernel, nil
	}
	kernel, err := rt.compileAuxKernel(mseLossKernelSource, "manta_mse_partials")
	if err != nil {
		return nil, err
	}
	rt.mseLossKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureMSSSIMLossKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.msssimLossKernel != nil {
		return rt.msssimLossKernel, nil
	}
	kernel, err := rt.compileAuxKernel(msssimLossKernelSource, "manta_msssim_partials")
	if err != nil {
		return nil, err
	}
	rt.msssimLossKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureScalarAddKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.scalarAddKernel != nil {
		return rt.scalarAddKernel, nil
	}
	kernel, err := rt.compileAuxKernel(scalarLossKernelSource, "manta_scalar_sum")
	if err != nil {
		return nil, err
	}
	rt.scalarAddKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureRDLossKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.rdLossKernel != nil {
		return rt.rdLossKernel, nil
	}
	kernel, err := rt.compileAuxKernel(scalarLossKernelSource, "manta_rd_loss")
	if err != nil {
		return nil, err
	}
	rt.rdLossKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureCrossEntropyKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.crossEntropyKernel != nil {
		return rt.crossEntropyKernel, nil
	}
	kernel, err := rt.compileAuxKernel(crossEntropyKernelSource, "manta_cross_entropy_partials")
	if err != nil {
		return nil, err
	}
	rt.crossEntropyKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) runGDNStep(inputs []*backend.Tensor, outputType mantaartifact.ValueType, inverse bool) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) < 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda gdn expects input, beta, and gamma")
	}
	input, beta, gamma := inputs[0], inputs[1], inputs[2]
	if len(input.Shape) != 4 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda gdn expects NCHW input")
	}
	elements := input.Elements()
	if elements == 0 {
		return backend.StepDispatchResult{Outputs: []*backend.Tensor{newStepOutputTensor(outputType, input.Shape, nil)}}, nil
	}
	channels, height, width := input.Shape[1], input.Shape[2], input.Shape[3]
	if len(beta.F32) < channels || len(gamma.F32) < channels*channels {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda gdn beta/gamma shape mismatch")
	}
	kernel, err := rt.ensureGDNKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	inputBuf, err := rt.uploadFloat32(input.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(inputBuf)
	betaBuf, err := rt.uploadFloat32(beta.F32[:channels])
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(betaBuf)
	gammaBuf, err := rt.uploadFloat32(gamma.F32[:channels*channels])
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(gammaBuf)
	outBuf, err := rt.allocFloat32(elements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((elements + int(block) - 1) / int(block))
	inverseFlag := 0
	entry := "__builtin_cuda_gdn"
	op := "gdn"
	if inverse {
		inverseFlag = 1
		entry = "__builtin_cuda_igdn"
		op = "igdn"
	}
	if err := rt.launchGDN(kernel, grid, block, inputBuf, betaBuf, gammaBuf, outBuf, elements, channels, height, width, inverseFlag); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, elements)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, input.Shape, outHost)},
		VariantEntry: entry,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_step",
			"device_execution": true,
			"execution_mode":   "cuda_device",
			"launch_api":       "cuda_driver",
			"op":               op,
		},
	}, nil
}

func (rt *deviceRuntime) runConv2DStep(inputs []*backend.Tensor, outputType mantaartifact.ValueType, cfg cudaConv2DConfig) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) < 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda conv2d expects input and weight")
	}
	kernel, err := rt.ensureConv2DKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outShape := []int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}
	outputElements := cfg.batches * cfg.outChannels * cfg.outHeight * cfg.outWidth
	if outputElements == 0 {
		return cudaConvStepResult(outputType, "__builtin_cuda_conv2d", "conv2d", outShape, nil, cfgMetadata(cfg)), nil
	}
	inputBuf, err := rt.uploadFloat32(inputs[0].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(inputBuf)
	weightBuf, err := rt.uploadFloat32(inputs[1].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(weightBuf)
	var biasBuf C.CUdeviceptr
	if cfg.hasBias {
		biasBuf, err = rt.uploadFloat32(inputs[2].F32[:cfg.outChannels])
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		defer rt.freeBuffer(biasBuf)
	}
	outBuf, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((outputElements + int(block) - 1) / int(block))
	if err := rt.launchConv2D(kernel, grid, block, inputBuf, weightBuf, biasBuf, outBuf, outputElements, cfg); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, outputElements)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return cudaConvStepResult(outputType, "__builtin_cuda_conv2d", "conv2d", outShape, outHost, cfgMetadata(cfg)), nil
}

func (rt *deviceRuntime) runConv2DTransposeStep(inputs []*backend.Tensor, outputType mantaartifact.ValueType, cfg cudaConv2DTransposeConfig) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) < 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda conv2d_transpose expects input and weight")
	}
	kernel, err := rt.ensureConv2DTransposeKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outShape := []int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}
	outputElements := cfg.batches * cfg.outChannels * cfg.outHeight * cfg.outWidth
	if outputElements == 0 {
		return cudaConvStepResult(outputType, "__builtin_cuda_conv2d_transpose", "conv2d_transpose", outShape, nil, cfgTransposeMetadata(cfg)), nil
	}
	inputBuf, err := rt.uploadFloat32(inputs[0].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(inputBuf)
	weightBuf, err := rt.uploadFloat32(inputs[1].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(weightBuf)
	var biasBuf C.CUdeviceptr
	if cfg.hasBias {
		biasBuf, err = rt.uploadFloat32(inputs[2].F32[:cfg.outChannels])
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		defer rt.freeBuffer(biasBuf)
	}
	outBuf, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((outputElements + int(block) - 1) / int(block))
	if err := rt.launchConv2DTranspose(kernel, grid, block, inputBuf, weightBuf, biasBuf, outBuf, outputElements, cfg); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, outputElements)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return cudaConvStepResult(outputType, "__builtin_cuda_conv2d_transpose", "conv2d_transpose", outShape, outHost, cfgTransposeMetadata(cfg)), nil
}

func (rt *deviceRuntime) runMSELossStep(inputs []*backend.Tensor, outputType mantaartifact.ValueType) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda mse_loss expects two inputs")
	}
	lhs, rhs := inputs[0], inputs[1]
	if !lhs.EqualShape(rhs) || len(lhs.F32) != len(rhs.F32) {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda mse_loss shape mismatch")
	}
	elements := len(lhs.F32)
	if elements == 0 {
		return cudaScalarStepResult(outputType, "__builtin_cuda_mse_loss", "mse_loss", "cuda_reduction", 0), nil
	}
	kernel, err := rt.ensureMSELossKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	lhsBuf, err := rt.uploadFloat32(lhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(lhsBuf)
	rhsBuf, err := rt.uploadFloat32(rhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(rhsBuf)
	grid, block := reductionLaunchConfig(elements)
	partialsBuf, err := rt.allocFloat32(int(grid))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(partialsBuf)
	if err := rt.launchMSEPartials(kernel, grid, block, lhsBuf, rhsBuf, partialsBuf, elements); err != nil {
		return backend.StepDispatchResult{}, err
	}
	partials := make([]float32, int(grid))
	if err := rt.downloadFloat32(partials, partialsBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	value := sumFloat32(partials) / float32(elements)
	return cudaScalarStepResult(outputType, "__builtin_cuda_mse_loss", "mse_loss", "cuda_reduction", value), nil
}

func (rt *deviceRuntime) runMSSSIMLossStep(inputs []*backend.Tensor, outputType mantaartifact.ValueType) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda ms_ssim_loss expects two inputs")
	}
	lhs, rhs := inputs[0], inputs[1]
	if !lhs.EqualShape(rhs) || len(lhs.F32) != len(rhs.F32) {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda ms_ssim_loss shape mismatch")
	}
	elements := len(lhs.F32)
	if elements == 0 {
		return cudaScalarStepResult(outputType, "__builtin_cuda_ms_ssim_loss", "ms_ssim_loss", "cuda_reduction", 0), nil
	}
	kernel, err := rt.ensureMSSSIMLossKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	lhsBuf, err := rt.uploadFloat32(lhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(lhsBuf)
	rhsBuf, err := rt.uploadFloat32(rhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(rhsBuf)
	grid, block := reductionLaunchConfig(elements)
	partialsBuf, err := rt.allocFloat32(int(grid) * 5)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(partialsBuf)
	if err := rt.launchMSSSIMPartials(kernel, grid, block, lhsBuf, rhsBuf, partialsBuf, elements); err != nil {
		return backend.StepDispatchResult{}, err
	}
	partials := make([]float32, int(grid)*5)
	if err := rt.downloadFloat32(partials, partialsBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	sumA, sumB, sumAA, sumBB, sumAB := 0.0, 0.0, 0.0, 0.0, 0.0
	for i := 0; i < len(partials); i += 5 {
		sumA += float64(partials[i])
		sumB += float64(partials[i+1])
		sumAA += float64(partials[i+2])
		sumBB += float64(partials[i+3])
		sumAB += float64(partials[i+4])
	}
	value := float32(msSSIMLossFromMoments(sumA, sumB, sumAA, sumBB, sumAB, elements))
	return cudaScalarStepResult(outputType, "__builtin_cuda_ms_ssim_loss", "ms_ssim_loss", "cuda_reduction", value), nil
}

func (rt *deviceRuntime) runScalarAddStep(inputs []*backend.Tensor, outputType mantaartifact.ValueType) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) == 0 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda scalar_add expects at least one input")
	}
	values := make([]float32, len(inputs))
	for i, input := range inputs {
		if input == nil || len(input.F32) != 1 {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda scalar_add input %d must be scalar", i)
		}
		values[i] = input.F32[0]
	}
	kernel, err := rt.ensureScalarAddKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	valuesBuf, err := rt.uploadFloat32(values)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(valuesBuf)
	outBuf, err := rt.allocFloat32(1)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	if err := rt.launchScalarSum(kernel, 1, cudaReductionBlockSize, valuesBuf, outBuf, len(values)); err != nil {
		return backend.StepDispatchResult{}, err
	}
	out := []float32{0}
	if err := rt.downloadFloat32(out, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return cudaScalarStepResult(outputType, "__builtin_cuda_scalar_add", "scalar_add", "cuda_scalar", out[0]), nil
}

func (rt *deviceRuntime) runRDLossStep(inputs []*backend.Tensor, outputType mantaartifact.ValueType, lambda float32) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil || len(inputs[0].F32) != 1 || len(inputs[1].F32) != 1 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda rate_distortion_loss expects scalar distortion and rate")
	}
	if math.IsNaN(float64(lambda)) || math.IsInf(float64(lambda), 0) || lambda < 0 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda rate_distortion_loss lambda must be finite and non-negative")
	}
	kernel, err := rt.ensureRDLossKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	distortionBuf, err := rt.uploadFloat32(inputs[0].F32[:1])
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(distortionBuf)
	rateBuf, err := rt.uploadFloat32(inputs[1].F32[:1])
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(rateBuf)
	outBuf, err := rt.allocFloat32(1)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	if err := rt.launchRDLoss(kernel, 1, 1, distortionBuf, rateBuf, outBuf, lambda); err != nil {
		return backend.StepDispatchResult{}, err
	}
	out := []float32{0}
	if err := rt.downloadFloat32(out, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	result := cudaScalarStepResult(outputType, "__builtin_cuda_rate_distortion_loss", "rate_distortion_loss", "cuda_scalar", out[0])
	result.Metadata["lambda"] = lambda
	return result, nil
}

func (rt *deviceRuntime) runCrossEntropyStep(inputs []*backend.Tensor, outputType mantaartifact.ValueType, plan cudaCrossEntropyPlan) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) < 1 || inputs[0] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda cross_entropy expects codes")
	}
	codes := inputs[0]
	var logits *backend.Tensor
	if len(inputs) > 1 {
		logits = inputs[1]
	}
	if len(codes.F32) != codes.Elements() {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda cross_entropy codes must be dense")
	}
	if err := validateCrossEntropyPlanInputs(codes, logits, plan); err != nil {
		return backend.StepDispatchResult{}, err
	}
	elements := len(codes.F32)
	if elements == 0 {
		return cudaScalarStepResult(outputType, "__builtin_cuda_cross_entropy", "cross_entropy_factorized", "cuda_reduction", 0), nil
	}
	kernel, err := rt.ensureCrossEntropyKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	codesBuf, err := rt.uploadFloat32(codes.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(codesBuf)
	var logitsBuf C.CUdeviceptr
	logitsLen := 0
	if logits != nil && len(logits.F32) > 0 {
		logitsLen = len(logits.F32)
		logitsBuf, err = rt.uploadFloat32(logits.F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		defer rt.freeBuffer(logitsBuf)
	}
	grid, block := reductionLaunchConfig(elements)
	partialsBuf, err := rt.allocFloat32(int(grid))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(partialsBuf)
	codeRank, codeN, codeC, codeH, codeW := cudaTensorShapeArgs(codes)
	logitsN, logitsC, logitsH, logitsW := 0, 0, 0, 0
	if logits != nil {
		_, logitsN, logitsC, logitsH, logitsW = cudaTensorShapeArgs(logits)
	}
	if err := rt.launchCrossEntropyPartials(kernel, grid, block, codesBuf, logitsBuf, partialsBuf, elements, int(plan.mode), int(plan.layout), plan.levels, plan.bits, codeRank, codeN, codeC, codeH, codeW, logitsLen, logitsN, logitsC, logitsH, logitsW, int(plan.sigmaMode)); err != nil {
		return backend.StepDispatchResult{}, err
	}
	partials := make([]float32, int(grid))
	if err := rt.downloadFloat32(partials, partialsBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	value := sumFloat32(partials)
	result := cudaScalarStepResult(outputType, "__builtin_cuda_cross_entropy", "cross_entropy_factorized", "cuda_reduction", value)
	result.Metadata["cross_entropy_mode"] = cudaCrossEntropyModeName(plan.mode)
	result.Metadata["logits_layout"] = cudaCrossEntropyLayoutName(plan.layout)
	result.Metadata["levels"] = plan.levels
	result.Metadata["bits"] = plan.bits
	if plan.mode == cudaCrossEntropyLogNormal {
		result.Metadata["sigma_parameter"] = cudaSigmaModeName(plan.sigmaMode)
	}
	return result, nil
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
	if C.mantaCudaLaunchQuantizeInPlace(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), ptr, C.int(elements), C.float(levels), C.float(scale), &errStr) != 0 {
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
	if C.mantaCudaMemFree(rt.ptr, ptr, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchRowWise(kernel *C.MantaCudaKernel, grid, block C.uint, in0, out0 C.CUdeviceptr, rows, cols C.int) error {
	var errStr *C.char
	if C.mantaCudaLaunchRowWise(rt.ptr, kernel, grid, block, in0, out0, rows, cols, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchElementWise(kernel *C.MantaCudaKernel, grid, block C.uint, lhs, rhs, out0 C.CUdeviceptr, elements C.int) error {
	var errStr *C.char
	if C.mantaCudaLaunchElementWise(rt.ptr, kernel, grid, block, lhs, rhs, out0, elements, &errStr) != 0 {
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

func (rt *deviceRuntime) launchGDN(kernel *auxKernel, grid, block uint, input, beta, gamma, out0 C.CUdeviceptr, elements, channels, height, width, inverse int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda gdn kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchGDN(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), input, beta, gamma, out0, C.int(elements), C.int(channels), C.int(height), C.int(width), C.int(inverse), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchConv2D(kernel *auxKernel, grid, block uint, input, weight, bias, out0 C.CUdeviceptr, elements int, cfg cudaConv2DConfig) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda conv2d kernel is not initialized")
	}
	hasBias := 0
	if cfg.hasBias {
		hasBias = 1
	}
	var errStr *C.char
	if C.mantaCudaLaunchConv2D(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		input,
		weight,
		bias,
		out0,
		C.int(elements),
		C.int(cfg.inChannels),
		C.int(cfg.inHeight),
		C.int(cfg.inWidth),
		C.int(cfg.outChannels),
		C.int(cfg.outHeight),
		C.int(cfg.outWidth),
		C.int(cfg.inPerGroup),
		C.int(cfg.outPerGroup),
		C.int(cfg.kernelH),
		C.int(cfg.kernelW),
		C.int(cfg.strideH),
		C.int(cfg.strideW),
		C.int(cfg.padH),
		C.int(cfg.padW),
		C.int(cfg.dilationH),
		C.int(cfg.dilationW),
		C.int(hasBias),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchConv2DTranspose(kernel *auxKernel, grid, block uint, input, weight, bias, out0 C.CUdeviceptr, elements int, cfg cudaConv2DTransposeConfig) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda conv2d_transpose kernel is not initialized")
	}
	hasBias := 0
	if cfg.hasBias {
		hasBias = 1
	}
	var errStr *C.char
	if C.mantaCudaLaunchConv2DTranspose(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		input,
		weight,
		bias,
		out0,
		C.int(elements),
		C.int(cfg.inChannels),
		C.int(cfg.inHeight),
		C.int(cfg.inWidth),
		C.int(cfg.outChannels),
		C.int(cfg.outHeight),
		C.int(cfg.outWidth),
		C.int(cfg.inPerGroup),
		C.int(cfg.outPerGroup),
		C.int(cfg.kernelH),
		C.int(cfg.kernelW),
		C.int(cfg.strideH),
		C.int(cfg.strideW),
		C.int(cfg.padH),
		C.int(cfg.padW),
		C.int(cfg.dilationH),
		C.int(cfg.dilationW),
		C.int(hasBias),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchMSEPartials(kernel *auxKernel, grid, block uint, lhs, rhs, partials C.CUdeviceptr, elements int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda mse_loss kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchMSEPartials(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), lhs, rhs, partials, C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchMSSSIMPartials(kernel *auxKernel, grid, block uint, lhs, rhs, partials C.CUdeviceptr, elements int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda ms_ssim_loss kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchMSSSIMPartials(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), lhs, rhs, partials, C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchScalarSum(kernel *auxKernel, grid, block uint, values, out0 C.CUdeviceptr, count int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda scalar_add kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchScalarSum(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), values, out0, C.int(count), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchRDLoss(kernel *auxKernel, grid, block uint, distortion, rate, out0 C.CUdeviceptr, lambda float32) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda rate_distortion_loss kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchRDLoss(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), distortion, rate, out0, C.float(lambda), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchCrossEntropyPartials(kernel *auxKernel, grid, block uint, codes, logits, partials C.CUdeviceptr, elements, mode, layout, levels, bits, codeRank, codeN, codeC, codeH, codeW, logitsLen, logitsN, logitsC, logitsH, logitsW, sigmaMode int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda cross_entropy kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchCrossEntropyPartials(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		codes,
		logits,
		partials,
		C.int(elements),
		C.int(mode),
		C.int(layout),
		C.int(levels),
		C.int(bits),
		C.int(codeRank),
		C.int(codeN),
		C.int(codeC),
		C.int(codeH),
		C.int(codeW),
		C.int(logitsLen),
		C.int(logitsN),
		C.int(logitsC),
		C.int(logitsH),
		C.int(logitsW),
		C.int(sigmaMode),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchUnary(kernel *C.MantaCudaKernel, grid, block C.uint, in0, out0 C.CUdeviceptr, elements C.int) error {
	var errStr *C.char
	if C.mantaCudaLaunchUnary(rt.ptr, kernel, grid, block, in0, out0, elements, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchScore(kernel *C.MantaCudaKernel, grid, block C.uint, query, docs, out0 C.CUdeviceptr, rows, cols C.int) error {
	var errStr *C.char
	if C.mantaCudaLaunchScore(rt.ptr, kernel, grid, block, query, docs, out0, rows, cols, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchOptimizerUpdate(kernel *auxKernel, grid, block uint, param, mom1, mom2, grad C.CUdeviceptr, elements, mode int, learningRate, weightDecay, beta1, beta2, corr1, corr2, epsilon, scale float32) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda optimizer kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchOptimizerUpdate(
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
	if C.mantaCudaLaunchSoftmaxBackwardRows(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), gradOut, probs, out0, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxLayerNormBackwardRows(kernel *auxKernel, grid, block uint, gradOut, normalized, pre, out0 C.CUdeviceptr, rows, cols int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary layernorm backward kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchLayerNormBackwardRows(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), gradOut, normalized, pre, out0, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxContrastiveScores(kernel *auxKernel, grid, block uint, query, positive, queryNorms, positiveNorms, scores C.CUdeviceptr, rows, width int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary contrastive score kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchContrastiveScores(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), query, positive, queryNorms, positiveNorms, scores, C.int(rows), C.int(width), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxInfoNCEScales(kernel *auxKernel, grid, block uint, scores, scales, rowLoss, rowScore C.CUdeviceptr, rows int, temperature float32) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary infonce scale kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchInfoNCEScales(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), scores, scales, rowLoss, rowScore, C.int(rows), C.float(temperature), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxContrastiveGrad(kernel *auxKernel, grid, block uint, query, positive, queryNorms, positiveNorms, scores, scales, queryGrads, positiveGrads C.CUdeviceptr, rows, width int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary contrastive grad kernel is not initialized")
	}
	var errStr *C.char
	if C.mantaCudaLaunchContrastiveGrad(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), query, positive, queryNorms, positiveNorms, scores, scales, queryGrads, positiveGrads, C.int(rows), C.int(width), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) matMulCublas(lhs, rhs, out0 C.CUdeviceptr, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool) error {
	return rt.matMulCublasWithBeta(lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight, 0)
}

func (rt *deviceRuntime) matMulCublasWithBeta(lhs, rhs, out0 C.CUdeviceptr, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool, beta float32) error {
	var errStr *C.char
	var left C.int
	var right C.int
	if transposeLeft {
		left = 1
	}
	if transposeRight {
		right = 1
	}
	if C.mantaCudaMatMulCublasWithBeta(rt.ptr, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, left, right, C.float(beta), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) matMulCublasWithBetaNoSync(lhs, rhs, out0 C.CUdeviceptr, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool, beta float32) error {
	var errStr *C.char
	var left C.int
	var right C.int
	if transposeLeft {
		left = 1
	}
	if transposeRight {
		right = 1
	}
	if C.mantaCudaMatMulCublasWithBetaNoSync(rt.ptr, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, left, right, C.float(beta), &errStr) != 0 {
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
	if C.mantaCudaMatMulCublasStridedBatched(rt.ptr, lhs, rhs, out0, batches, lhsRows, lhsCols, rhsRows, rhsCols, left, right, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func cStringValue(value *C.char) string {
	if value == nil {
		return ""
	}
	out := C.GoString(value)
	C.mantaCudaFreeCString(value)
	return out
}

func cStringError(value *C.char) error {
	if value == nil {
		return fmt.Errorf("cuda error")
	}
	defer C.mantaCudaFreeCString(value)
	return errors.New(C.GoString(value))
}

func (rt *deviceRuntime) synchronize() error {
	var errStr *C.char
	if C.mantaCudaSynchronize(rt.ptr, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func cudaEnvFlagEnabled(name string) bool {
	switch cudaEnv(name) {
	case "1", "true", "TRUE", "yes", "YES":
		return true
	default:
		return false
	}
}

func cudaEnv(name string) string {
	if value, ok := os.LookupEnv(name); ok {
		return value
	}
	return ""
}

func (rt *deviceRuntime) runMatMul(inputs []*backend.Tensor, outputType mantaartifact.ValueType) (backend.StepDispatchResult, error) {
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

func (rt *deviceRuntime) runMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	if lhs == nil {
		return nil, fmt.Errorf("cuda matmul lhs is nil")
	}
	if len(rightNames) == 0 {
		return nil, fmt.Errorf("cuda matmul requires at least one rhs binding")
	}
	type boundRightPlan struct {
		name            string
		resident        residentMatrix
		batches         int
		rows            int
		inner           int
		cols            int
		outShape        []int
		lhsRows         int
		lhsCols         int
		rhsRows         int
		rhsCols         int
		downloadedBytes int64
	}
	plans := make([]boundRightPlan, len(rightNames))
	for i, name := range rightNames {
		resident, ok := rt.residentMatrices[name]
		if !ok {
			return nil, fmt.Errorf("cuda matmul binding %q is not resident", name)
		}
		rhsTensor := &backend.Tensor{DType: "f32", Shape: []int{resident.rows, resident.cols}}
		batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(lhs, rhsTensor)
		if transposeLeft || transposeRight {
			batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(lhs, rhsTensor, transposeLeft, transposeRight)
		}
		if err != nil {
			return nil, err
		}
		if rhsBatched {
			return nil, fmt.Errorf("cuda cached rhs matmul does not support batched rhs")
		}
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(lhs)
			rhsRows, rhsCols = resident.rows, resident.cols
		}
		plans[i] = boundRightPlan{
			name:            name,
			resident:        resident,
			batches:         batches,
			rows:            rows,
			inner:           inner,
			cols:            cols,
			outShape:        outShape,
			lhsRows:         lhsRows,
			lhsCols:         lhsCols,
			rhsRows:         rhsRows,
			rhsCols:         rhsCols,
			downloadedBytes: int64(batches * rows * cols * 4),
		}
	}

	compiled := cudaBuiltinMatMulCompiledKernel()
	runStart := time.Now()
	runUploadedBytes := int64(len(lhs.F32) * 4)
	lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", lhs.F32)
	if err != nil {
		return nil, err
	}
	results := make([]backend.StepDispatchResult, len(plans))
	for i, plan := range plans {
		if i > 0 {
			runStart = time.Now()
			runUploadedBytes = 0
		}
		outHost := make([]float32, plan.batches*plan.rows*plan.cols)
		outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
		if err != nil {
			return nil, err
		}
		if err := rt.matMulCublas(lhsBuf, plan.resident.ptr, outBuf, C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight); err != nil {
			return nil, err
		}
		if err := rt.downloadFloat32(outHost, outBuf); err != nil {
			return nil, err
		}
		rt.recordMatMulRun(runStart, runUploadedBytes, plan.downloadedBytes, false, true)
		results[i] = backend.StepDispatchResult{
			Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, plan.outShape, outHost)},
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
				"rhs_binding":      plan.name,
				"rhs_residency":    "device_resident",
				"coalesced_lhs":    true,
			},
		}
	}
	return results, nil
}

func (rt *deviceRuntime) runAccumulatedMatMulsWithBoundRights(lhsInputs []*backend.Tensor, rightNames []string, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if len(lhsInputs) == 0 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda accumulated matmul requires at least one lhs input")
	}
	if len(lhsInputs) != len(rightNames) {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda accumulated matmul lhs/right count mismatch: %d != %d", len(lhsInputs), len(rightNames))
	}
	type accumulatedBoundRightPlan struct {
		lhs             *backend.Tensor
		name            string
		resident        residentMatrix
		batches         int
		rows            int
		inner           int
		cols            int
		outShape        []int
		lhsRows         int
		lhsCols         int
		rhsRows         int
		rhsCols         int
		uploadedBytes   int64
		downloadedBytes int64
	}
	plans := make([]accumulatedBoundRightPlan, len(lhsInputs))
	var outShape []int
	var outElements int
	for i, lhs := range lhsInputs {
		if lhs == nil {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda accumulated matmul lhs %d is nil", i)
		}
		name := rightNames[i]
		resident, ok := rt.residentMatrices[name]
		if !ok {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul binding %q is not resident", name)
		}
		rhsTensor := &backend.Tensor{DType: "f32", Shape: []int{resident.rows, resident.cols}}
		batches, rows, inner, cols, rhsBatched, planOutShape, err := matmulLayout(lhs, rhsTensor)
		if transposeLeft || transposeRight {
			batches, rows, inner, cols, rhsBatched, planOutShape, err = matmulLayoutWithTranspose(lhs, rhsTensor, transposeLeft, transposeRight)
		}
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		if rhsBatched {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda cached rhs matmul does not support batched rhs")
		}
		if i == 0 {
			outShape = append([]int(nil), planOutShape...)
			outElements = batches * rows * cols
		} else if !sameIntSlice(outShape, planOutShape) {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda accumulated matmul output shape mismatch: %v != %v", planOutShape, outShape)
		}
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(lhs)
			rhsRows, rhsCols = resident.rows, resident.cols
		}
		plans[i] = accumulatedBoundRightPlan{
			lhs:             lhs,
			name:            name,
			resident:        resident,
			batches:         batches,
			rows:            rows,
			inner:           inner,
			cols:            cols,
			outShape:        planOutShape,
			lhsRows:         lhsRows,
			lhsCols:         lhsCols,
			rhsRows:         rhsRows,
			rhsCols:         rhsCols,
			uploadedBytes:   int64(len(lhs.F32) * 4),
			downloadedBytes: int64(batches * rows * cols * 4),
		}
	}

	compiled := cudaBuiltinMatMulCompiledKernel()
	runStart := time.Now()
	var runUploadedBytes int64
	outHost := make([]float32, outElements)
	outBuf, err := rt.matMulScratchFloat32("matmul_accum_out", len(outHost))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	singleSync := !cudaEnvFlagEnabled("MANTA_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC")
	bindings := make([]string, len(plans))
	for i, plan := range plans {
		runUploadedBytes += plan.uploadedBytes
		bindings[i] = plan.name
		lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_accum_lhs", plan.lhs.F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		beta := float32(0)
		if i > 0 {
			beta = 1
		}
		if singleSync {
			err = rt.matMulCublasWithBetaNoSync(lhsBuf, plan.resident.ptr, outBuf, C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight, beta)
		} else {
			err = rt.matMulCublasWithBeta(lhsBuf, plan.resident.ptr, outBuf, C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight, beta)
		}
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
	}
	if singleSync {
		err = rt.synchronize()
	}
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	rt.recordMatMulRun(runStart, runUploadedBytes, int64(len(outHost)*4), false, true)
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":             "backend_native",
			"execution_mode":            "cuda_device",
			"device_execution":          true,
			"launch_api":                "cublasSgemmAccumulated",
			"launch_compiler":           "cublas",
			"backend_library":           "cublas",
			"transpose_left":            transposeLeft,
			"transpose_right":           transposeRight,
			"rhs_bindings":              bindings,
			"rhs_residency":             "device_resident",
			"accumulated_bound_rights":  true,
			"accumulated_cublas_calls":  len(plans),
			"accumulated_download_once": true,
			"accumulated_single_sync":   singleSync,
		},
	}, nil
}

func (rt *deviceRuntime) runMatMulsWithSharedLeft(lhs *backend.Tensor, rhsInputs []*backend.Tensor, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	if lhs == nil {
		return nil, fmt.Errorf("cuda matmul lhs is nil")
	}
	if len(rhsInputs) == 0 {
		return nil, fmt.Errorf("cuda matmul requires at least one rhs input")
	}
	type sharedLeftPlan struct {
		rhs             *backend.Tensor
		batches         int
		rows            int
		inner           int
		cols            int
		rhsBatched      bool
		outShape        []int
		lhsRows         int
		lhsCols         int
		rhsRows         int
		rhsCols         int
		downloadedBytes int64
	}
	plans := make([]sharedLeftPlan, len(rhsInputs))
	for i, rhs := range rhsInputs {
		if rhs == nil {
			return nil, fmt.Errorf("cuda matmul rhs %d is nil", i)
		}
		batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(lhs, rhs)
		if transposeLeft || transposeRight {
			batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(lhs, rhs, transposeLeft, transposeRight)
		}
		if err != nil {
			return nil, err
		}
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight || rhsBatched {
			lhsRows, lhsCols = matrixShape(lhs)
			rhsRows, rhsCols = matrixShape(rhs)
		}
		plans[i] = sharedLeftPlan{
			rhs:             rhs,
			batches:         batches,
			rows:            rows,
			inner:           inner,
			cols:            cols,
			rhsBatched:      rhsBatched,
			outShape:        outShape,
			lhsRows:         lhsRows,
			lhsCols:         lhsCols,
			rhsRows:         rhsRows,
			rhsCols:         rhsCols,
			downloadedBytes: int64(batches * rows * cols * 4),
		}
	}

	compiled := cudaBuiltinMatMulCompiledKernel()
	lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", lhs.F32)
	if err != nil {
		return nil, err
	}
	results := make([]backend.StepDispatchResult, len(plans))
	for i, plan := range plans {
		runStart := time.Now()
		runUploadedBytes := int64(len(plan.rhs.F32) * 4)
		if i == 0 {
			runUploadedBytes += int64(len(lhs.F32) * 4)
		}
		rhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_rhs", plan.rhs.F32)
		if err != nil {
			return nil, err
		}
		outHost := make([]float32, plan.batches*plan.rows*plan.cols)
		outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
		if err != nil {
			return nil, err
		}
		launchAPI := "cublasSgemm"
		if plan.rhsBatched {
			if err := rt.matMulCublasStridedBatched(lhsBuf, rhsBuf, outBuf, C.int(plan.batches), C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight); err != nil {
				return nil, err
			}
			launchAPI = "cublasSgemmStridedBatched"
		} else if err := rt.matMulCublas(lhsBuf, rhsBuf, outBuf, C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight); err != nil {
			return nil, err
		}
		if err := rt.downloadFloat32(outHost, outBuf); err != nil {
			return nil, err
		}
		rt.recordMatMulRun(runStart, runUploadedBytes, plan.downloadedBytes, false, false)
		results[i] = backend.StepDispatchResult{
			Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, plan.outShape, outHost)},
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
				"shared_lhs":       true,
			},
		}
	}
	return results, nil
}

func (rt *deviceRuntime) runMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
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

func (rt *deviceRuntime) runMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
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

func (rt *deviceRuntime) runMatMulWithTranspose(inputs []*backend.Tensor, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
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
		Backend:    mantaartifact.BackendCUDA,
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

func sameIntSlice(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func newStepOutputTensor(outputType mantaartifact.ValueType, shape []int, data []float32) *backend.Tensor {
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

const (
	cudaReductionBlockSize = 256
	cudaMaxReductionBlocks = 1024
)

func reductionLaunchConfig(elements int) (grid, block uint) {
	if elements <= 0 {
		return 1, cudaReductionBlockSize
	}
	blocks := (elements + cudaReductionBlockSize - 1) / cudaReductionBlockSize
	if blocks < 1 {
		blocks = 1
	}
	if blocks > cudaMaxReductionBlocks {
		blocks = cudaMaxReductionBlocks
	}
	return uint(blocks), cudaReductionBlockSize
}

func cudaConvStepResult(outputType mantaartifact.ValueType, variant, op string, shape []int, data []float32, extra map[string]any) backend.StepDispatchResult {
	metadata := map[string]any{
		"dispatch_mode":    "backend_step",
		"device_execution": true,
		"execution_mode":   "cuda_device",
		"launch_api":       "cuda_driver",
		"launch_compiler":  "nvrtc",
		"op":               op,
	}
	for key, value := range extra {
		metadata[key] = value
	}
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, shape, data)},
		VariantEntry: variant,
		Metadata:     metadata,
	}
}

func cfgMetadata(cfg cudaConv2DConfig) map[string]any {
	return map[string]any{
		"groups":     cfg.groups,
		"stride_h":   cfg.strideH,
		"stride_w":   cfg.strideW,
		"pad_h":      cfg.padH,
		"pad_w":      cfg.padW,
		"dilation_h": cfg.dilationH,
		"dilation_w": cfg.dilationW,
		"has_bias":   cfg.hasBias,
	}
}

func cfgTransposeMetadata(cfg cudaConv2DTransposeConfig) map[string]any {
	return map[string]any{
		"groups":           cfg.groups,
		"stride_h":         cfg.strideH,
		"stride_w":         cfg.strideW,
		"pad_h":            cfg.padH,
		"pad_w":            cfg.padW,
		"dilation_h":       cfg.dilationH,
		"dilation_w":       cfg.dilationW,
		"output_padding_h": cfg.outPadH,
		"output_padding_w": cfg.outPadW,
		"has_bias":         cfg.hasBias,
	}
}

func msSSIMLossFromMoments(sumA, sumB, sumAA, sumBB, sumAB float64, elements int) float64 {
	if elements <= 0 {
		return 0
	}
	denom := float64(elements)
	meanA := sumA / denom
	meanB := sumB / denom
	varA := sumAA/denom - meanA*meanA
	varB := sumBB/denom - meanB*meanB
	cov := sumAB/denom - meanA*meanB
	const c1 = 0.01 * 0.01
	const c2 = 0.03 * 0.03
	ssim := ((2*meanA*meanB + c1) * (2*cov + c2)) / ((meanA*meanA + meanB*meanB + c1) * (varA + varB + c2))
	if math.IsNaN(ssim) || math.IsInf(ssim, 0) {
		ssim = 0
	}
	loss := 1 - ssim
	if loss < 0 {
		return 0
	}
	if loss > 1 {
		return 1
	}
	return loss
}

func cudaScalarStepResult(outputType mantaartifact.ValueType, variant, op, launchAPI string, value float32) backend.StepDispatchResult {
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, []int{1}, []float32{value})},
		VariantEntry: variant,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_step",
			"device_execution": true,
			"execution_mode":   "cuda_device",
			"launch_api":       launchAPI,
			"launch_compiler":  "nvrtc",
			"op":               op,
		},
	}
}

func validateCrossEntropyPlanInputs(codes, logits *backend.Tensor, plan cudaCrossEntropyPlan) error {
	if codes == nil {
		return fmt.Errorf("cuda cross_entropy expects codes")
	}
	switch plan.mode {
	case cudaCrossEntropyLogNormal:
		if logits == nil {
			return fmt.Errorf("cuda cross_entropy log_normal expects norm params")
		}
		if len(codes.Shape) != 3 || len(logits.Shape) != 4 {
			return fmt.Errorf("cuda cross_entropy log_normal expects codes NHW and params N2HW")
		}
		if logits.Shape[0] != codes.Shape[0] || logits.Shape[1] < 2 || logits.Shape[2] != codes.Shape[1] || logits.Shape[3] != codes.Shape[2] {
			return fmt.Errorf("cuda cross_entropy norm param shape %v does not match codes %v", logits.Shape, codes.Shape)
		}
		if len(logits.F32) < logits.Elements() {
			return fmt.Errorf("cuda cross_entropy norm params must be dense")
		}
		if err := validateLogNormalParams(logits, plan.sigmaMode); err != nil {
			return err
		}
	case cudaCrossEntropyBitPlane:
		if plan.bits <= 0 {
			return fmt.Errorf("cuda cross_entropy bit-plane mode requires bits")
		}
		if plan.layout == cudaCrossEntropyLayoutNCHW && !supportsNCHWBitPair(codes, logits, plan.bits) {
			return fmt.Errorf("cuda cross_entropy bit-plane NCHW logits shape %v does not match codes %v", logitsShape(logits), codes.Shape)
		}
	case cudaCrossEntropyCategorical:
		if plan.levels <= 0 {
			return fmt.Errorf("cuda cross_entropy levels must be positive")
		}
		if plan.layout == cudaCrossEntropyLayoutNCHW && !supportsNCHWAlphabet(codes, logits, plan.levels) {
			return fmt.Errorf("cuda cross_entropy NCHW logits shape %v does not match codes %v", logitsShape(logits), codes.Shape)
		}
	default:
		return fmt.Errorf("cuda cross_entropy unsupported mode %d", plan.mode)
	}
	return nil
}

func validateLogNormalParams(params *backend.Tensor, sigmaMode cudaSigmaMode) error {
	channels, height, width := params.Shape[1], params.Shape[2], params.Shape[3]
	for n := 0; n < params.Shape[0]; n++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				muOffset := ((n*channels+0)*height+y)*width + x
				sigmaOffset := ((n*channels+1)*height+y)*width + x
				mu := params.F32[muOffset]
				rawSigma := params.F32[sigmaOffset]
				if math.IsNaN(float64(mu)) || math.IsInf(float64(mu), 0) {
					return fmt.Errorf("cuda cross_entropy invalid norm params at offset %d", muOffset)
				}
				if math.IsNaN(float64(rawSigma)) || math.IsInf(float64(rawSigma), 0) || (sigmaMode == cudaSigmaRaw && rawSigma <= 0) {
					return fmt.Errorf("cuda cross_entropy invalid norm params at offset %d", sigmaOffset)
				}
			}
		}
	}
	return nil
}

func cudaTensorShapeArgs(tensor *backend.Tensor) (rank, n, c, h, w int) {
	if tensor == nil {
		return 0, 0, 0, 0, 0
	}
	switch len(tensor.Shape) {
	case 4:
		return 4, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2], tensor.Shape[3]
	case 3:
		return 3, tensor.Shape[0], 1, tensor.Shape[1], tensor.Shape[2]
	case 2:
		return 2, 1, tensor.Shape[0], 1, tensor.Shape[1]
	case 1:
		return 1, 1, 1, 1, tensor.Shape[0]
	default:
		width := tensor.Elements()
		if width < 1 {
			width = 1
		}
		return len(tensor.Shape), 1, 1, 1, width
	}
}

func logitsShape(tensor *backend.Tensor) []int {
	if tensor == nil {
		return nil
	}
	return tensor.Shape
}

func cudaCrossEntropyModeName(mode cudaCrossEntropyMode) string {
	switch mode {
	case cudaCrossEntropyCategorical:
		return "categorical"
	case cudaCrossEntropyBitPlane:
		return "bit-plane"
	case cudaCrossEntropyLogNormal:
		return "log_normal"
	default:
		return "unknown"
	}
}

func cudaCrossEntropyLayoutName(layout cudaCrossEntropyLayout) string {
	switch layout {
	case cudaCrossEntropyLayoutUniform:
		return "uniform"
	case cudaCrossEntropyLayoutGlobal:
		return "global"
	case cudaCrossEntropyLayoutFlat:
		return "flat"
	case cudaCrossEntropyLayoutNCHW:
		return "nchw"
	case cudaCrossEntropyLayoutSigmoidFallback:
		return "sigmoid_fallback"
	default:
		return "unknown"
	}
}

func cudaSigmaModeName(mode cudaSigmaMode) string {
	switch mode {
	case cudaSigmaRaw:
		return "raw"
	case cudaSigmaSoftplus:
		return "softplus"
	case cudaSigmaExp:
		return "exp"
	default:
		return "unknown"
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
