//go:build linux && cgo

package cuda

import (
	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

type imageGradAccelerator struct {
	device *deviceRuntime
}

func init() {
	backend.RegisterImageGradAccelerator(mantaartifact.BackendCUDA, NewImageGradAccelerator)
}

func NewImageGradAccelerator() (backend.ImageGradAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	return &imageGradAccelerator{device: device}, nil
}

func (a *imageGradAccelerator) Backend() mantaartifact.BackendKind {
	return mantaartifact.BackendCUDA
}

func (a *imageGradAccelerator) RunConv2DBackward(input, weight, bias, gradOut *backend.Tensor, attrs map[string]string) (*backend.Tensor, *backend.Tensor, *backend.Tensor, bool, error) {
	if a == nil || a.device == nil {
		return nil, nil, nil, false, nil
	}
	inputs := []*backend.Tensor{input, weight}
	if bias != nil {
		inputs = append(inputs, bias)
	}
	cfg, ok := planBuiltinConv2D(mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: attrs}, inputs)
	if !ok {
		return nil, nil, nil, false, nil
	}
	gradIn, gradW, gradB, err := a.device.runConv2DBackward(input, weight, bias, gradOut, cfg)
	return gradIn, gradW, gradB, true, err
}

func (a *imageGradAccelerator) RunConv2DTransposeBackward(input, weight, bias, gradOut *backend.Tensor, attrs map[string]string) (*backend.Tensor, *backend.Tensor, *backend.Tensor, bool, error) {
	if a == nil || a.device == nil {
		return nil, nil, nil, false, nil
	}
	inputs := []*backend.Tensor{input, weight}
	if bias != nil {
		inputs = append(inputs, bias)
	}
	cfg, ok := planBuiltinConv2DTranspose(mantaartifact.Step{Kind: mantaartifact.StepConv2DTrans, Attributes: attrs}, inputs)
	if !ok {
		return nil, nil, nil, false, nil
	}
	gradIn, gradW, gradB, err := a.device.runConv2DTransposeBackward(input, weight, bias, gradOut, cfg)
	return gradIn, gradW, gradB, true, err
}

func (a *imageGradAccelerator) RunGDNBackward(input, beta, gamma, gradOut *backend.Tensor, inverse bool) (*backend.Tensor, *backend.Tensor, *backend.Tensor, bool, error) {
	if a == nil || a.device == nil {
		return nil, nil, nil, false, nil
	}
	if !supportsBuiltinGDN([]*backend.Tensor{input, beta, gamma}) {
		return nil, nil, nil, false, nil
	}
	gradIn, gradBeta, gradGamma, err := a.device.runGDNBackward(input, beta, gamma, gradOut, inverse)
	return gradIn, gradBeta, gradGamma, true, err
}

func (a *imageGradAccelerator) Close() {
	if a == nil || a.device == nil {
		return
	}
	a.device.close()
	a.device = nil
}
