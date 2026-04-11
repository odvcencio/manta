//go:build !darwin || !cgo

package metal

import (
	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

type deviceRuntime struct{}

func newDeviceRuntime() (*deviceRuntime, error) {
	return nil, nil
}

func (rt *deviceRuntime) close() {}

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
