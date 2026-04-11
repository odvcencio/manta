//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	void* device;
	void* queue;
} BarrMetalRuntime;

typedef struct {
	void* pipeline;
} BarrMetalKernel;

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

static char* barr_dup_ns_error(const char* prefix, NSError* error) {
	if (error == nil) {
		return barr_dup_format(prefix, "unknown");
	}
	NSString* desc = [error localizedDescription];
	return barr_dup_format(prefix, [desc UTF8String]);
}

static int barrMetalRuntimeCreate(BarrMetalRuntime** out, char** err) {
	@autoreleasepool {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			*err = barr_dup_format("MTLCreateSystemDefaultDevice", "no Metal device");
			return 1;
		}
		id<MTLCommandQueue> queue = [device newCommandQueue];
		if (queue == nil) {
			*err = barr_dup_format("newCommandQueue", "failed");
			return 1;
		}
		BarrMetalRuntime* rt = (BarrMetalRuntime*)malloc(sizeof(BarrMetalRuntime));
		if (rt == NULL) {
			*err = barr_dup_format("malloc", "failed to allocate runtime");
			return 1;
		}
		rt->device = (__bridge_retained void*)device;
		rt->queue = (__bridge_retained void*)queue;
		*out = rt;
		return 0;
	}
}

static void barrMetalRuntimeDestroy(BarrMetalRuntime* rt) {
	if (rt == NULL) {
		return;
	}
	if (rt->device != NULL) {
		(void)(__bridge_transfer id)rt->device;
	}
	if (rt->queue != NULL) {
		(void)(__bridge_transfer id)rt->queue;
	}
	free(rt);
}

static int barrMetalCompileKernel(BarrMetalRuntime* rt, const char* src, const char* entry, BarrMetalKernel** out, char** err) {
	@autoreleasepool {
		id<MTLDevice> device = (__bridge id<MTLDevice>)rt->device;
		NSString* source = [NSString stringWithUTF8String:src];
		NSString* entryName = [NSString stringWithUTF8String:entry];
		if (source == nil || entryName == nil) {
			*err = barr_dup_format("NSString", "invalid UTF-8");
			return 1;
		}
		NSError* nsErr = nil;
		id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&nsErr];
		if (library == nil) {
			*err = barr_dup_ns_error("newLibraryWithSource", nsErr);
			return 1;
		}
		id<MTLFunction> function = [library newFunctionWithName:entryName];
		if (function == nil) {
			*err = barr_dup_format("newFunctionWithName", "entry not found");
			return 1;
		}
		id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&nsErr];
		if (pipeline == nil) {
			*err = barr_dup_ns_error("newComputePipelineStateWithFunction", nsErr);
			return 1;
		}
		BarrMetalKernel* kernel = (BarrMetalKernel*)malloc(sizeof(BarrMetalKernel));
		if (kernel == NULL) {
			*err = barr_dup_format("malloc", "failed to allocate kernel");
			return 1;
		}
		kernel->pipeline = (__bridge_retained void*)pipeline;
		*out = kernel;
		return 0;
	}
}

static void barrMetalKernelDestroy(BarrMetalKernel* kernel) {
	if (kernel == NULL) {
		return;
	}
	if (kernel->pipeline != NULL) {
		(void)(__bridge_transfer id)kernel->pipeline;
	}
	free(kernel);
}

static int barrMetalLaunchRowWise(BarrMetalRuntime* rt, BarrMetalKernel* kernel, const float* in0, float* out0, int rows, int cols, int threadgroupSize, char** err) {
	@autoreleasepool {
		id<MTLDevice> device = (__bridge id<MTLDevice>)rt->device;
		id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)rt->queue;
		id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)kernel->pipeline;
		NSUInteger elements = (NSUInteger)rows * (NSUInteger)cols;
		NSUInteger bytes = elements * sizeof(float);
		id<MTLBuffer> inBuf = [device newBufferWithBytes:in0 length:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> outBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> rowsBuf = [device newBufferWithBytes:&rows length:sizeof(int) options:MTLResourceStorageModeShared];
		id<MTLBuffer> colsBuf = [device newBufferWithBytes:&cols length:sizeof(int) options:MTLResourceStorageModeShared];
		if (inBuf == nil || outBuf == nil || rowsBuf == nil || colsBuf == nil) {
			*err = barr_dup_format("newBuffer", "failed");
			return 1;
		}
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if (commandBuffer == nil || encoder == nil) {
			*err = barr_dup_format("commandBuffer", "failed");
			return 1;
		}
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:inBuf offset:0 atIndex:0];
		[encoder setBuffer:outBuf offset:0 atIndex:1];
		[encoder setBuffer:rowsBuf offset:0 atIndex:2];
		[encoder setBuffer:colsBuf offset:0 atIndex:3];
		NSUInteger tg = threadgroupSize > 0 ? (NSUInteger)threadgroupSize : 1;
		if (tg > pipeline.maxTotalThreadsPerThreadgroup) {
			tg = pipeline.maxTotalThreadsPerThreadgroup;
		}
		MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
		MTLSize group = MTLSizeMake(tg, 1, 1);
		[encoder dispatchThreads:grid threadsPerThreadgroup:group];
		[encoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if (commandBuffer.error != nil) {
			*err = barr_dup_ns_error("commandBuffer", commandBuffer.error);
			return 1;
		}
		memcpy(out0, [outBuf contents], bytes);
		return 0;
	}
}

static int barrMetalLaunchElementWise(BarrMetalRuntime* rt, BarrMetalKernel* kernel, const float* lhs, const float* rhs, float* out0, int elements, int threadgroupSize, char** err) {
	@autoreleasepool {
		id<MTLDevice> device = (__bridge id<MTLDevice>)rt->device;
		id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)rt->queue;
		id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)kernel->pipeline;
		NSUInteger bytes = (NSUInteger)elements * sizeof(float);
		id<MTLBuffer> lhsBuf = [device newBufferWithBytes:lhs length:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> rhsBuf = [device newBufferWithBytes:rhs length:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> outBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> elementsBuf = [device newBufferWithBytes:&elements length:sizeof(int) options:MTLResourceStorageModeShared];
		if (lhsBuf == nil || rhsBuf == nil || outBuf == nil || elementsBuf == nil) {
			*err = barr_dup_format("newBuffer", "failed");
			return 1;
		}
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if (commandBuffer == nil || encoder == nil) {
			*err = barr_dup_format("commandBuffer", "failed");
			return 1;
		}
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:lhsBuf offset:0 atIndex:0];
		[encoder setBuffer:rhsBuf offset:0 atIndex:1];
		[encoder setBuffer:outBuf offset:0 atIndex:2];
		[encoder setBuffer:elementsBuf offset:0 atIndex:3];
		NSUInteger tg = threadgroupSize > 0 ? (NSUInteger)threadgroupSize : 1;
		if (tg > pipeline.maxTotalThreadsPerThreadgroup) {
			tg = pipeline.maxTotalThreadsPerThreadgroup;
		}
		MTLSize grid = MTLSizeMake((NSUInteger)elements, 1, 1);
		MTLSize group = MTLSizeMake(tg, 1, 1);
		[encoder dispatchThreads:grid threadsPerThreadgroup:group];
		[encoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if (commandBuffer.error != nil) {
			*err = barr_dup_ns_error("commandBuffer", commandBuffer.error);
			return 1;
		}
		memcpy(out0, [outBuf contents], bytes);
		return 0;
	}
}

static int barrMetalLaunchUnary(BarrMetalRuntime* rt, BarrMetalKernel* kernel, const float* in0, float* out0, int elements, int threadgroupSize, char** err) {
	@autoreleasepool {
		id<MTLDevice> device = (__bridge id<MTLDevice>)rt->device;
		id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)rt->queue;
		id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)kernel->pipeline;
		NSUInteger bytes = (NSUInteger)elements * sizeof(float);
		id<MTLBuffer> inBuf = [device newBufferWithBytes:in0 length:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> outBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> elementsBuf = [device newBufferWithBytes:&elements length:sizeof(int) options:MTLResourceStorageModeShared];
		if (inBuf == nil || outBuf == nil || elementsBuf == nil) {
			*err = barr_dup_format("newBuffer", "failed");
			return 1;
		}
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if (commandBuffer == nil || encoder == nil) {
			*err = barr_dup_format("commandBuffer", "failed");
			return 1;
		}
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:inBuf offset:0 atIndex:0];
		[encoder setBuffer:outBuf offset:0 atIndex:1];
		[encoder setBuffer:elementsBuf offset:0 atIndex:2];
		NSUInteger tg = threadgroupSize > 0 ? (NSUInteger)threadgroupSize : 1;
		if (tg > pipeline.maxTotalThreadsPerThreadgroup) {
			tg = pipeline.maxTotalThreadsPerThreadgroup;
		}
		MTLSize grid = MTLSizeMake((NSUInteger)elements, 1, 1);
		MTLSize group = MTLSizeMake(tg, 1, 1);
		[encoder dispatchThreads:grid threadsPerThreadgroup:group];
		[encoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if (commandBuffer.error != nil) {
			*err = barr_dup_ns_error("commandBuffer", commandBuffer.error);
			return 1;
		}
		memcpy(out0, [outBuf contents], bytes);
		return 0;
	}
}

static int barrMetalLaunchScore(BarrMetalRuntime* rt, BarrMetalKernel* kernel, const float* query, const float* docs, float* out0, int rows, int cols, int threadgroupSize, char** err) {
	@autoreleasepool {
		id<MTLDevice> device = (__bridge id<MTLDevice>)rt->device;
		id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)rt->queue;
		id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)kernel->pipeline;
		NSUInteger queryBytes = (NSUInteger)cols * sizeof(float);
		NSUInteger docsBytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
		NSUInteger outBytes = (NSUInteger)rows * sizeof(float);
		id<MTLBuffer> queryBuf = [device newBufferWithBytes:query length:queryBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> docsBuf = [device newBufferWithBytes:docs length:docsBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> outBuf = [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> rowsBuf = [device newBufferWithBytes:&rows length:sizeof(int) options:MTLResourceStorageModeShared];
		id<MTLBuffer> colsBuf = [device newBufferWithBytes:&cols length:sizeof(int) options:MTLResourceStorageModeShared];
		if (queryBuf == nil || docsBuf == nil || outBuf == nil || rowsBuf == nil || colsBuf == nil) {
			*err = barr_dup_format("newBuffer", "failed");
			return 1;
		}
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if (commandBuffer == nil || encoder == nil) {
			*err = barr_dup_format("commandBuffer", "failed");
			return 1;
		}
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:queryBuf offset:0 atIndex:0];
		[encoder setBuffer:docsBuf offset:0 atIndex:1];
		[encoder setBuffer:outBuf offset:0 atIndex:2];
		[encoder setBuffer:rowsBuf offset:0 atIndex:3];
		[encoder setBuffer:colsBuf offset:0 atIndex:4];
		NSUInteger tg = threadgroupSize > 0 ? (NSUInteger)threadgroupSize : 1;
		if (tg > pipeline.maxTotalThreadsPerThreadgroup) {
			tg = pipeline.maxTotalThreadsPerThreadgroup;
		}
		MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
		MTLSize group = MTLSizeMake(tg, 1, 1);
		[encoder dispatchThreads:grid threadsPerThreadgroup:group];
		[encoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if (commandBuffer.error != nil) {
			*err = barr_dup_ns_error("commandBuffer", commandBuffer.error);
			return 1;
		}
		memcpy(out0, [outBuf contents], outBytes);
		return 0;
	}
}

static int barrMetalMatMulMPS(BarrMetalRuntime* rt, const float* lhs, const float* rhs, float* out0, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, char** err) {
	@autoreleasepool {
		id<MTLDevice> device = (__bridge id<MTLDevice>)rt->device;
		id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)rt->queue;
		NSUInteger lhsBytes = (NSUInteger)lhsRows * (NSUInteger)lhsCols * sizeof(float);
		NSUInteger rhsBytes = (NSUInteger)rhsRows * (NSUInteger)rhsCols * sizeof(float);
		NSUInteger rows = transposeLeft ? (NSUInteger)lhsCols : (NSUInteger)lhsRows;
		NSUInteger inner = transposeLeft ? (NSUInteger)lhsRows : (NSUInteger)lhsCols;
		NSUInteger rhsInner = transposeRight ? (NSUInteger)rhsCols : (NSUInteger)rhsRows;
		NSUInteger cols = transposeRight ? (NSUInteger)rhsRows : (NSUInteger)rhsCols;
		if (inner != rhsInner) {
			*err = barr_dup_format("MPSMatrixMultiplication", "shape mismatch");
			return 1;
		}
		NSUInteger outBytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
		id<MTLBuffer> lhsBuf = [device newBufferWithBytes:lhs length:lhsBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> rhsBuf = [device newBufferWithBytes:rhs length:rhsBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> outBuf = [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
		if (lhsBuf == nil || rhsBuf == nil || outBuf == nil) {
			*err = barr_dup_format("newBuffer", "failed");
			return 1;
		}
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		if (commandBuffer == nil) {
			*err = barr_dup_format("commandBuffer", "failed");
			return 1;
		}
		MPSMatrixDescriptor* lhsDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)lhsRows
		                                                                     columns:(NSUInteger)lhsCols
		                                                                    rowBytes:(NSUInteger)lhsCols * sizeof(float)
		                                                                    dataType:MPSDataTypeFloat32];
		MPSMatrixDescriptor* rhsDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rhsRows
		                                                                     columns:(NSUInteger)rhsCols
		                                                                    rowBytes:(NSUInteger)rhsCols * sizeof(float)
		                                                                    dataType:MPSDataTypeFloat32];
		MPSMatrixDescriptor* outDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
		                                                                     columns:cols
		                                                                    rowBytes:cols * sizeof(float)
		                                                                    dataType:MPSDataTypeFloat32];
		if (lhsDesc == nil || rhsDesc == nil || outDesc == nil) {
			*err = barr_dup_format("MPSMatrixDescriptor", "failed");
			return 1;
		}
		MPSMatrix* lhsMatrix = [[MPSMatrix alloc] initWithBuffer:lhsBuf descriptor:lhsDesc];
		MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:rhsBuf descriptor:rhsDesc];
		MPSMatrix* outMatrix = [[MPSMatrix alloc] initWithBuffer:outBuf descriptor:outDesc];
		if (lhsMatrix == nil || rhsMatrix == nil || outMatrix == nil) {
			*err = barr_dup_format("MPSMatrix", "failed");
			return 1;
		}
		MPSMatrixMultiplication* op = [[MPSMatrixMultiplication alloc] initWithDevice:device
		                                                                        transposeLeft:transposeLeft ? YES : NO
		                                                                       transposeRight:transposeRight ? YES : NO
		                                                                           resultRows:rows
		                                                                        resultColumns:cols
		                                                                      interiorColumns:inner
		                                                                                  alpha:1.0
		                                                                                   beta:0.0];
		if (op == nil) {
			*err = barr_dup_format("MPSMatrixMultiplication", "failed");
			return 1;
		}
		[op encodeToCommandBuffer:commandBuffer leftMatrix:lhsMatrix rightMatrix:rhsMatrix resultMatrix:outMatrix];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if (commandBuffer.error != nil) {
			*err = barr_dup_ns_error("commandBuffer", commandBuffer.error);
			return 1;
		}
		memcpy(out0, [outBuf contents], outBytes);
		return 0;
	}
}

static void barrMetalFreeCString(char* s) {
	if (s != NULL) {
		free(s);
	}
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

type deviceRuntime struct {
	ptr *C.BarrMetalRuntime
}

type deviceKernel struct {
	ptr       *C.BarrMetalKernel
	shapeKind metalShapeKind
}

type metalShapeKind int

const (
	metalShapeUnsupported metalShapeKind = iota
	metalShapeRowWise
	metalShapeElementWiseBinary
	metalShapeElementWiseUnary
	metalShapeRowScore
)

func newDeviceRuntime() (*deviceRuntime, error) {
	var rt *C.BarrMetalRuntime
	var errStr *C.char
	if C.barrMetalRuntimeCreate(&rt, &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	return &deviceRuntime{ptr: rt}, nil
}

func (rt *deviceRuntime) close() {
	if rt == nil || rt.ptr == nil {
		return
	}
	C.barrMetalRuntimeDestroy(rt.ptr)
	rt.ptr = nil
}

func (rt *deviceRuntime) attachDeviceExecution(prog *backend.NativeKernelProgram, kernel barr.Kernel) error {
	shapeKind := classifyMetalKernel(kernel)
	if shapeKind == metalShapeUnsupported {
		prog.LaunchConfig["device_execution"] = false
		prog.LaunchConfig["execution_mode"] = "host_fallback"
		return nil
	}
	deviceKernel, err := rt.compileKernel(prog.Compiled, shapeKind)
	if err != nil {
		return err
	}
	prog.LaunchConfig["device_execution"] = true
	prog.LaunchConfig["execution_mode"] = "metal_device"
	prog.LaunchConfig["launch_compiler"] = "metal_runtime"
	prog.Run = func(inputs []*backend.Tensor) ([]*backend.Tensor, error) {
		return rt.runKernel(deviceKernel, kernel, prog, inputs)
	}
	return nil
}

func (rt *deviceRuntime) compileKernel(compiled backend.CompiledKernel, shapeKind metalShapeKind) (*deviceKernel, error) {
	src := C.CString(compiled.Source)
	entry := C.CString(compiled.Entry)
	defer C.free(unsafe.Pointer(src))
	defer C.free(unsafe.Pointer(entry))

	var kernel *C.BarrMetalKernel
	var errStr *C.char
	if C.barrMetalCompileKernel(rt.ptr, src, entry, &kernel, &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	return &deviceKernel{ptr: kernel, shapeKind: shapeKind}, nil
}

func (rt *deviceRuntime) runKernel(deviceKernel *deviceKernel, kernel barr.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	switch deviceKernel.shapeKind {
	case metalShapeRowWise:
		return rt.runRowWiseKernel(deviceKernel, kernel, prog, inputs)
	case metalShapeElementWiseBinary:
		return rt.runElementWiseKernel(deviceKernel, kernel, prog, inputs)
	case metalShapeElementWiseUnary:
		return rt.runUnaryKernel(deviceKernel, kernel, prog, inputs)
	case metalShapeRowScore:
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
	outHost := make([]float32, rows*cols)
	tg := C.int(firstIntValue(prog.LaunchConfig["launch_threadgroup_size"], 128))
	var inPtr *C.float
	var outPtr *C.float
	if len(in.F32) > 0 {
		inPtr = (*C.float)(unsafe.Pointer(&in.F32[0]))
	}
	if len(outHost) > 0 {
		outPtr = (*C.float)(unsafe.Pointer(&outHost[0]))
	}
	var errStr *C.char
	if C.barrMetalLaunchRowWise(rt.ptr, deviceKernel.ptr, inPtr, outPtr, C.int(rows), C.int(cols), tg, &errStr) != 0 {
		return nil, cStringError(errStr)
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
	outHost := make([]float32, elements)
	tg := C.int(firstIntValue(prog.LaunchConfig["launch_threadgroup_size"], 128))
	var lhsPtr, rhsPtr, outPtr *C.float
	if len(lhs.F32) > 0 {
		lhsPtr = (*C.float)(unsafe.Pointer(&lhs.F32[0]))
	}
	if len(rhs.F32) > 0 {
		rhsPtr = (*C.float)(unsafe.Pointer(&rhs.F32[0]))
	}
	if len(outHost) > 0 {
		outPtr = (*C.float)(unsafe.Pointer(&outHost[0]))
	}
	var errStr *C.char
	if C.barrMetalLaunchElementWise(rt.ptr, deviceKernel.ptr, lhsPtr, rhsPtr, outPtr, C.int(elements), tg, &errStr) != 0 {
		return nil, cStringError(errStr)
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
	outHost := make([]float32, elements)
	tg := C.int(firstIntValue(prog.LaunchConfig["launch_threadgroup_size"], 128))
	var inPtr, outPtr *C.float
	if len(in.F32) > 0 {
		inPtr = (*C.float)(unsafe.Pointer(&in.F32[0]))
	}
	if len(outHost) > 0 {
		outPtr = (*C.float)(unsafe.Pointer(&outHost[0]))
	}
	var errStr *C.char
	if C.barrMetalLaunchUnary(rt.ptr, deviceKernel.ptr, inPtr, outPtr, C.int(elements), tg, &errStr) != 0 {
		return nil, cStringError(errStr)
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
	outHost := make([]float32, rows)
	tg := C.int(firstIntValue(prog.LaunchConfig["launch_threadgroup_size"], 128))
	var queryPtr, docsPtr, outPtr *C.float
	if len(query.F32) > 0 {
		queryPtr = (*C.float)(unsafe.Pointer(&query.F32[0]))
	}
	if len(docs.F32) > 0 {
		docsPtr = (*C.float)(unsafe.Pointer(&docs.F32[0]))
	}
	if len(outHost) > 0 {
		outPtr = (*C.float)(unsafe.Pointer(&outHost[0]))
	}
	var errStr *C.char
	if C.barrMetalLaunchScore(rt.ptr, deviceKernel.ptr, queryPtr, docsPtr, outPtr, C.int(rows), C.int(cols), tg, &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	return []*backend.Tensor{newOutputTensor(kernel, []int{rows}, outHost)}, nil
}

func classifyMetalKernel(kernel barr.Kernel) metalShapeKind {
	if len(kernel.Body) < 2 {
		return metalShapeUnsupported
	}
	switch kernel.Body[0].Op {
	case "normalize", "rmsnorm", "layernorm", "softmax", "rope":
		return metalShapeRowWise
	case "binary_add", "binary_sub", "binary_mul", "binary_div":
		return metalShapeElementWiseBinary
	case "dequant", "gelu":
		return metalShapeElementWiseUnary
	case "dot", "cosine", "l2_distance":
		return metalShapeRowScore
	default:
		return metalShapeUnsupported
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

func cStringError(value *C.char) error {
	if value == nil {
		return fmt.Errorf("metal error")
	}
	defer C.barrMetalFreeCString(value)
	return errors.New(C.GoString(value))
}

func (rt *deviceRuntime) runMatMul(inputs []*backend.Tensor, outputType barr.ValueType) (backend.StepDispatchResult, error) {
	return rt.runMatMulWithTranspose(inputs, outputType, false, false)
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
	compiled := metalBuiltinMatMulCompiledKernel()
	outHost := make([]float32, batches*rows*cols)
	if !rhsBatched {
		var lhsPtr, rhsPtr, outPtr *C.float
		if len(inputs[0].F32) > 0 {
			lhsPtr = (*C.float)(unsafe.Pointer(&inputs[0].F32[0]))
		}
		if len(inputs[1].F32) > 0 {
			rhsPtr = (*C.float)(unsafe.Pointer(&inputs[1].F32[0]))
		}
		if len(outHost) > 0 {
			outPtr = (*C.float)(unsafe.Pointer(&outHost[0]))
		}
		var errStr *C.char
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(inputs[0])
			rhsRows, rhsCols = matrixShape(inputs[1])
		}
		if C.barrMetalMatMulMPS(rt.ptr, lhsPtr, rhsPtr, outPtr, C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), boolToCInt(transposeLeft), boolToCInt(transposeRight), &errStr) != 0 {
			return backend.StepDispatchResult{}, cStringError(errStr)
		}
	} else {
		lhsRows, lhsCols := rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(inputs[0])
			rhsRows, rhsCols = matrixShape(inputs[1])
		}
		lhsStride := lhsRows * lhsCols
		rhsStride := rhsRows * rhsCols
		outStride := rows * cols
		for b := 0; b < batches; b++ {
			lhsSlice := inputs[0].F32[b*lhsStride : (b+1)*lhsStride]
			rhsSlice := inputs[1].F32[b*rhsStride : (b+1)*rhsStride]
			outSlice := outHost[b*outStride : (b+1)*outStride]
			var lhsPtr, rhsPtr, outPtr *C.float
			if len(lhsSlice) > 0 {
				lhsPtr = (*C.float)(unsafe.Pointer(&lhsSlice[0]))
			}
			if len(rhsSlice) > 0 {
				rhsPtr = (*C.float)(unsafe.Pointer(&rhsSlice[0]))
			}
			if len(outSlice) > 0 {
				outPtr = (*C.float)(unsafe.Pointer(&outSlice[0]))
			}
			var errStr *C.char
			if C.barrMetalMatMulMPS(rt.ptr, lhsPtr, rhsPtr, outPtr, C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), boolToCInt(transposeLeft), boolToCInt(transposeRight), &errStr) != 0 {
				return backend.StepDispatchResult{}, cStringError(errStr)
			}
		}
	}
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_native",
			"execution_mode":   "metal_device",
			"device_execution": true,
			"launch_api":       "MPSMatrixMultiplication",
			"launch_compiler":  "metal_performance_shaders",
			"backend_library":  "MetalPerformanceShaders",
			"transpose_left":   transposeLeft,
			"transpose_right":  transposeRight,
		},
	}, nil
}

func metalBuiltinMatMulCompiledKernel() backend.CompiledKernel {
	return backend.CompiledKernel{
		Name:       "__builtin_matmul",
		Backend:    barr.BackendMetal,
		Entry:      "mps_matrix_multiplication",
		Source:     "library:mps_matrix_multiplication",
		SourceHash: "library:mps_matrix_multiplication",
		Meta: map[string]string{
			"library":      "MetalPerformanceShaders",
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
	if len(lhs.Shape) != 2 || len(rhs.Shape) != 2 {
		return 0, 0, 0, 0, false, nil, fmt.Errorf("transposed matmul expects rank-2 tensors")
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
}

func matrixShape(t *backend.Tensor) (rows, cols int) {
	if t == nil || len(t.Shape) < 2 {
		return 0, 0
	}
	return t.Shape[len(t.Shape)-2], t.Shape[len(t.Shape)-1]
}

func boolToCInt(v bool) C.int {
	if v {
		return 1
	}
	return 0
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
	default:
		return fallback
	}
}
