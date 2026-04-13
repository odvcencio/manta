//go:build !js || !wasm

package webgpu

import (
	"context"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

type deviceRuntime struct{}

func newDeviceRuntime(context.Context) (*deviceRuntime, error) {
	return nil, nil
}

func (*deviceRuntime) dispatchStep(context.Context, mantaartifact.Step, []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	return backend.StepDispatchResult{}, false, nil
}
