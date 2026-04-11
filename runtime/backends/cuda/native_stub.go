//go:build !linux || !cgo

package cuda

import (
	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

type deviceRuntime struct{}

func newDeviceRuntime() (*deviceRuntime, error) {
	return nil, nil
}

func (rt *deviceRuntime) close() {}

func (rt *deviceRuntime) matMulStatsSnapshot() backend.MatMulAcceleratorStats {
	return backend.MatMulAcceleratorStats{}
}

func (rt *deviceRuntime) attachDeviceExecution(prog *backend.NativeKernelProgram, kernel barr.Kernel) error {
	if prog.LaunchConfig == nil {
		prog.LaunchConfig = map[string]any{}
	}
	prog.LaunchConfig["device_execution"] = false
	prog.LaunchConfig["execution_mode"] = "host_fallback"
	return nil
}

func (rt *deviceRuntime) runMatMul(inputs []*backend.Tensor, outputType barr.ValueType) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runMatMulWithTranspose(inputs []*backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) bindMatMulRight(name string, tensor *backend.Tensor) error {
	return nil
}

func (rt *deviceRuntime) unbindMatMulRight(name string) error {
	return nil
}

func (rt *deviceRuntime) runMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType barr.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	return nil, nil
}

func (rt *deviceRuntime) runMatMulsWithSharedLeft(lhs *backend.Tensor, rhs []*backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	return nil, nil
}

func (rt *deviceRuntime) runAccumulatedMatMulsWithBoundRights(lhs []*backend.Tensor, rightNames []string, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}
