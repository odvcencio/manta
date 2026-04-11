package metal

import (
	"fmt"
	"math"
	"time"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

type matMulAccelerator struct {
	device *deviceRuntime
	bound  map[string]*backend.Tensor
	stats  backend.MatMulAcceleratorStats
}

func init() {
	backend.RegisterMatMulAccelerator(barr.BackendMetal, NewMatMulAccelerator)
}

// NewMatMulAccelerator exposes the Metal backend's library-backed matmul fast path.
func NewMatMulAccelerator() (backend.MatMulAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	return &matMulAccelerator{device: device, bound: map[string]*backend.Tensor{}}, nil
}

func (a *matMulAccelerator) Backend() barr.BackendKind {
	return barr.BackendMetal
}

func (a *matMulAccelerator) RunMatMul(inputs []*backend.Tensor, outputType barr.ValueType) (backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("metal matmul accelerator is not initialized")
	}
	return a.RunMatMulWithTranspose(inputs, outputType, false, false)
}

func (a *matMulAccelerator) RunMatMulWithTranspose(inputs []*backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("metal matmul accelerator is not initialized")
	}
	start := time.Now()
	result, err := a.device.runMatMulWithTranspose(inputs, outputType, transposeLeft, transposeRight)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	a.recordMatMulRun(start, inputs, result, false, false)
	return result, nil
}

func (a *matMulAccelerator) BindMatrix(name string, tensor *backend.Tensor) error {
	if a == nil || a.device == nil {
		return fmt.Errorf("metal matmul accelerator is not initialized")
	}
	if name == "" {
		return fmt.Errorf("metal matmul accelerator binding name is required")
	}
	if tensor == nil || len(tensor.Shape) != 2 {
		return fmt.Errorf("metal matmul accelerator binding %q must be a rank-2 tensor", name)
	}
	if a.bound == nil {
		a.bound = map[string]*backend.Tensor{}
	}
	start := time.Now()
	clone := tensor.Clone()
	switch clone.DType {
	case "q4":
		fakeQuantizeInPlace(clone.F32, 4)
		a.stats.QuantizePasses++
		a.stats.QuantizedBytes += int64(len(clone.F32) * 4)
	case "q8":
		fakeQuantizeInPlace(clone.F32, 8)
		a.stats.QuantizePasses++
		a.stats.QuantizedBytes += int64(len(clone.F32) * 4)
	}
	a.bound[name] = clone
	a.stats.BindCalls++
	a.stats.UploadedBytes += int64(len(clone.F32) * 4)
	a.stats.BindNanos += time.Since(start).Nanoseconds()
	a.stats.BoundMatrices = int64(len(a.bound))
	return nil
}

func (a *matMulAccelerator) UnbindMatrix(name string) error {
	if a == nil || a.device == nil {
		return fmt.Errorf("metal matmul accelerator is not initialized")
	}
	delete(a.bound, name)
	a.stats.BoundMatrices = int64(len(a.bound))
	return nil
}

func (a *matMulAccelerator) RunMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("metal matmul accelerator is not initialized")
	}
	lhs := a.bound[leftName]
	if lhs == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("metal matmul accelerator binding %q is not resident", leftName)
	}
	start := time.Now()
	result, err := a.device.runMatMulWithTranspose([]*backend.Tensor{lhs, rhs}, outputType, transposeLeft, transposeRight)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	a.recordMatMulRun(start, []*backend.Tensor{lhs, rhs}, result, true, false)
	if result.Metadata == nil {
		result.Metadata = map[string]any{}
	}
	result.Metadata["lhs_binding"] = leftName
	result.Metadata["lhs_residency"] = "host_bound"
	return result, nil
}

func (a *matMulAccelerator) RunMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("metal matmul accelerator is not initialized")
	}
	rhs := a.bound[rightName]
	if rhs == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("metal matmul accelerator binding %q is not resident", rightName)
	}
	start := time.Now()
	result, err := a.device.runMatMulWithTranspose([]*backend.Tensor{lhs, rhs}, outputType, transposeLeft, transposeRight)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	a.recordMatMulRun(start, []*backend.Tensor{lhs, rhs}, result, false, true)
	if result.Metadata == nil {
		result.Metadata = map[string]any{}
	}
	result.Metadata["rhs_binding"] = rightName
	result.Metadata["rhs_residency"] = "host_bound"
	return result, nil
}

func (a *matMulAccelerator) Stats() backend.MatMulAcceleratorStats {
	if a == nil {
		return backend.MatMulAcceleratorStats{}
	}
	return a.stats
}

func (a *matMulAccelerator) recordMatMulRun(start time.Time, inputs []*backend.Tensor, result backend.StepDispatchResult, boundLeft, boundRight bool) {
	if a == nil {
		return
	}
	a.stats.RunCalls++
	if boundLeft {
		a.stats.BoundLeftCalls++
	}
	if boundRight {
		a.stats.BoundRightCalls++
	}
	for _, input := range inputs {
		a.stats.RunUploadedBytes += tensorFloat32Bytes(input)
	}
	for _, output := range result.Outputs {
		a.stats.RunDownloadedBytes += tensorFloat32Bytes(output)
	}
	a.stats.RunNanos += time.Since(start).Nanoseconds()
}

func tensorFloat32Bytes(tensor *backend.Tensor) int64 {
	if tensor == nil {
		return 0
	}
	return int64(len(tensor.F32) * 4)
}

func (a *matMulAccelerator) Close() {
	if a == nil || a.device == nil {
		return
	}
	a.device.close()
	a.device = nil
	a.bound = nil
}

func fakeQuantizeInPlace(data []float32, bits int) {
	if bits <= 0 || len(data) == 0 {
		return
	}
	maxAbs := float32(0)
	for _, v := range data {
		abs := float32(math.Abs(float64(v)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if maxAbs == 0 {
		return
	}
	levelsInt := (1 << uint(bits-1)) - 1
	levels := float32(levelsInt)
	if levels <= 0 {
		return
	}
	scale := maxAbs / levels
	if scale == 0 {
		return
	}
	for i, v := range data {
		q := float32(math.Round(float64(v / scale)))
		if q > levels {
			q = levels
		}
		if q < -levels {
			q = -levels
		}
		data[i] = q * scale
	}
}
