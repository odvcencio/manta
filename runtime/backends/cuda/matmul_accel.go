package cuda

import (
	"fmt"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

type matMulAccelerator struct {
	device *deviceRuntime
}

func init() {
	backend.RegisterMatMulAccelerator(barr.BackendCUDA, NewMatMulAccelerator)
}

// NewMatMulAccelerator exposes the CUDA backend's library-backed matmul fast path.
func NewMatMulAccelerator() (backend.MatMulAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	return &matMulAccelerator{device: device}, nil
}

func (a *matMulAccelerator) Backend() barr.BackendKind {
	return barr.BackendCUDA
}

func (a *matMulAccelerator) RunMatMul(inputs []*backend.Tensor, outputType barr.ValueType) (backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runMatMul(inputs, outputType)
}

func (a *matMulAccelerator) RunMatMulWithTranspose(inputs []*backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runMatMulWithTranspose(inputs, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) BindMatrix(name string, tensor *backend.Tensor) error {
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.bindMatMulRight(name, tensor)
}

func (a *matMulAccelerator) UnbindMatrix(name string) error {
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.unbindMatMulRight(name)
}

func (a *matMulAccelerator) RunMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runMatMulWithBoundLeft(leftName, rhs, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) RunMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runMatMulWithBoundRight(lhs, rightName, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) RunMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType barr.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	if a == nil || a.device == nil {
		return nil, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runMatMulWithBoundRights(lhs, rightNames, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) Stats() backend.MatMulAcceleratorStats {
	if a == nil || a.device == nil {
		return backend.MatMulAcceleratorStats{}
	}
	return a.device.matMulStatsSnapshot()
}

func (a *matMulAccelerator) Close() {
	if a == nil || a.device == nil {
		return
	}
	a.device.close()
	a.device = nil
}
