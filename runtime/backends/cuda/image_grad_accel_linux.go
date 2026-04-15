//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda.h>
*/
import "C"

import (
	"fmt"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

type imageGradAccelerator struct {
	device    *deviceRuntime
	owned     []C.CUdeviceptr
	residency map[*backend.Tensor]C.CUdeviceptr
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

func (a *imageGradAccelerator) RunConv2D(input, weight, bias *backend.Tensor, attrs map[string]string) (*backend.Tensor, bool, error) {
	if a == nil || a.device == nil {
		return nil, false, nil
	}
	inputs := []*backend.Tensor{input, weight}
	if bias != nil {
		inputs = append(inputs, bias)
	}
	cfg, ok := planBuiltinConv2D(mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: attrs}, inputs)
	if !ok {
		return nil, false, nil
	}
	kernel, err := a.device.ensureConv2DKernel()
	if err != nil {
		return nil, true, err
	}
	inputBuf, err := a.bufferForTensor(input)
	if err != nil {
		return nil, true, err
	}
	weightBuf, err := a.bufferForTensor(weight)
	if err != nil {
		return nil, true, err
	}
	var biasBuf C.CUdeviceptr
	if cfg.hasBias {
		biasBuf, err = a.bufferForTensor(bias)
		if err != nil {
			return nil, true, err
		}
	}
	outShape := []int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}
	outputElements := cfg.batches * cfg.outChannels * cfg.outHeight * cfg.outWidth
	if outputElements == 0 {
		return newDeviceTensor(tensorDType(input), outShape, 0), true, nil
	}
	outBuf, err := a.allocStepFloat32(outputElements)
	if err != nil {
		return nil, true, err
	}
	block := uint(128)
	grid := uint((outputElements + int(block) - 1) / int(block))
	if err := a.device.launchConv2D(kernel, grid, block, inputBuf, weightBuf, biasBuf, outBuf, outputElements, cfg); err != nil {
		return nil, true, err
	}
	return newDeviceTensor(tensorDType(input), outShape, outBuf), true, nil
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
	inputKernel, err := a.device.ensureConv2DInputGradKernel()
	if err != nil {
		return nil, nil, nil, true, err
	}
	weightKernel, err := a.device.ensureConv2DWeightGradKernel()
	if err != nil {
		return nil, nil, nil, true, err
	}
	inputBuf, err := a.bufferForTensor(input)
	if err != nil {
		return nil, nil, nil, true, err
	}
	weightBuf, err := a.bufferForTensor(weight)
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradOutBuf, err := a.bufferForTensor(gradOut)
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradInBuf, err := a.allocStepFloat32(input.Elements())
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradWBuf, err := a.device.allocFloat32(weight.Elements())
	if err != nil {
		return nil, nil, nil, true, err
	}
	defer a.device.freeBuffer(gradWBuf)
	block := uint(128)
	if input.Elements() > 0 {
		grid := uint((input.Elements() + int(block) - 1) / int(block))
		if err := a.device.launchConv2DInputGrad(inputKernel, grid, block, gradOutBuf, weightBuf, gradInBuf, input.Elements(), cfg); err != nil {
			return nil, nil, nil, true, err
		}
	}
	if weight.Elements() > 0 {
		grid := uint((weight.Elements() + int(block) - 1) / int(block))
		if err := a.device.launchConv2DWeightGrad(weightKernel, grid, block, inputBuf, gradOutBuf, gradWBuf, weight.Elements(), cfg); err != nil {
			return nil, nil, nil, true, err
		}
	}
	gradWHost := make([]float32, weight.Elements())
	if err := a.device.downloadFloat32(gradWHost, gradWBuf); err != nil {
		return nil, nil, nil, true, err
	}
	var gradB *backend.Tensor
	if bias != nil {
		biasKernel, err := a.device.ensureConv2DBiasGradKernel()
		if err != nil {
			return nil, nil, nil, true, err
		}
		gradBBuf, err := a.device.allocFloat32(bias.Elements())
		if err != nil {
			return nil, nil, nil, true, err
		}
		defer a.device.freeBuffer(gradBBuf)
		if cfg.outChannels > 0 {
			grid := uint((cfg.outChannels + int(block) - 1) / int(block))
			if err := a.device.launchConv2DBiasGrad(biasKernel, grid, block, gradOutBuf, gradBBuf, cfg); err != nil {
				return nil, nil, nil, true, err
			}
		}
		gradBHost := make([]float32, bias.Elements())
		if err := a.device.downloadFloat32(gradBHost, gradBBuf); err != nil {
			return nil, nil, nil, true, err
		}
		gradB = &backend.Tensor{DType: bias.DType, Shape: append([]int(nil), bias.Shape...), F32: gradBHost}
	}
	gradIn := newDeviceTensor(input.DType, input.Shape, gradInBuf)
	gradW := &backend.Tensor{DType: weight.DType, Shape: append([]int(nil), weight.Shape...), F32: gradWHost}
	return gradIn, gradW, gradB, true, nil
}

func (a *imageGradAccelerator) RunConv2DTranspose(input, weight, bias *backend.Tensor, attrs map[string]string) (*backend.Tensor, bool, error) {
	if a == nil || a.device == nil {
		return nil, false, nil
	}
	inputs := []*backend.Tensor{input, weight}
	if bias != nil {
		inputs = append(inputs, bias)
	}
	cfg, ok := planBuiltinConv2DTranspose(mantaartifact.Step{Kind: mantaartifact.StepConv2DTrans, Attributes: attrs}, inputs)
	if !ok {
		return nil, false, nil
	}
	kernel, err := a.device.ensureConv2DTransposeKernel()
	if err != nil {
		return nil, true, err
	}
	inputBuf, err := a.bufferForTensor(input)
	if err != nil {
		return nil, true, err
	}
	weightBuf, err := a.bufferForTensor(weight)
	if err != nil {
		return nil, true, err
	}
	var biasBuf C.CUdeviceptr
	if cfg.hasBias {
		biasBuf, err = a.bufferForTensor(bias)
		if err != nil {
			return nil, true, err
		}
	}
	outShape := []int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}
	outputElements := cfg.batches * cfg.outChannels * cfg.outHeight * cfg.outWidth
	if outputElements == 0 {
		return newDeviceTensor(tensorDType(input), outShape, 0), true, nil
	}
	outBuf, err := a.allocStepFloat32(outputElements)
	if err != nil {
		return nil, true, err
	}
	block := uint(128)
	grid := uint((outputElements + int(block) - 1) / int(block))
	if err := a.device.launchConv2DTranspose(kernel, grid, block, inputBuf, weightBuf, biasBuf, outBuf, outputElements, cfg); err != nil {
		return nil, true, err
	}
	return newDeviceTensor(tensorDType(input), outShape, outBuf), true, nil
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
	inputKernel, err := a.device.ensureConv2DTransposeInputGradKernel()
	if err != nil {
		return nil, nil, nil, true, err
	}
	weightKernel, err := a.device.ensureConv2DTransposeWeightGradKernel()
	if err != nil {
		return nil, nil, nil, true, err
	}
	inputBuf, err := a.bufferForTensor(input)
	if err != nil {
		return nil, nil, nil, true, err
	}
	weightBuf, err := a.bufferForTensor(weight)
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradOutBuf, err := a.bufferForTensor(gradOut)
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradInBuf, err := a.allocStepFloat32(input.Elements())
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradWBuf, err := a.device.allocFloat32(weight.Elements())
	if err != nil {
		return nil, nil, nil, true, err
	}
	defer a.device.freeBuffer(gradWBuf)
	block := uint(128)
	if input.Elements() > 0 {
		grid := uint((input.Elements() + int(block) - 1) / int(block))
		if err := a.device.launchConv2DTransposeInputGrad(inputKernel, grid, block, gradOutBuf, weightBuf, gradInBuf, input.Elements(), cfg); err != nil {
			return nil, nil, nil, true, err
		}
	}
	if weight.Elements() > 0 {
		grid := uint((weight.Elements() + int(block) - 1) / int(block))
		if err := a.device.launchConv2DTransposeWeightGrad(weightKernel, grid, block, inputBuf, gradOutBuf, gradWBuf, weight.Elements(), cfg); err != nil {
			return nil, nil, nil, true, err
		}
	}
	gradWHost := make([]float32, weight.Elements())
	if err := a.device.downloadFloat32(gradWHost, gradWBuf); err != nil {
		return nil, nil, nil, true, err
	}
	var gradB *backend.Tensor
	if bias != nil {
		biasKernel, err := a.device.ensureConv2DBiasGradKernel()
		if err != nil {
			return nil, nil, nil, true, err
		}
		gradBBuf, err := a.device.allocFloat32(bias.Elements())
		if err != nil {
			return nil, nil, nil, true, err
		}
		defer a.device.freeBuffer(gradBBuf)
		if cfg.outChannels > 0 {
			grid := uint((cfg.outChannels + int(block) - 1) / int(block))
			convCfg := cudaConv2DConfig{
				batches:     cfg.batches,
				outChannels: cfg.outChannels,
				outHeight:   cfg.outHeight,
				outWidth:    cfg.outWidth,
			}
			if err := a.device.launchConv2DBiasGrad(biasKernel, grid, block, gradOutBuf, gradBBuf, convCfg); err != nil {
				return nil, nil, nil, true, err
			}
		}
		gradBHost := make([]float32, bias.Elements())
		if err := a.device.downloadFloat32(gradBHost, gradBBuf); err != nil {
			return nil, nil, nil, true, err
		}
		gradB = &backend.Tensor{DType: bias.DType, Shape: append([]int(nil), bias.Shape...), F32: gradBHost}
	}
	gradIn := newDeviceTensor(input.DType, input.Shape, gradInBuf)
	gradW := &backend.Tensor{DType: weight.DType, Shape: append([]int(nil), weight.Shape...), F32: gradWHost}
	return gradIn, gradW, gradB, true, nil
}

func (a *imageGradAccelerator) RunGDN(input, beta, gamma *backend.Tensor, inverse bool) (*backend.Tensor, bool, error) {
	if a == nil || a.device == nil {
		return nil, false, nil
	}
	inputs := []*backend.Tensor{input, beta, gamma}
	if !supportsBuiltinGDN(inputs) {
		return nil, false, nil
	}
	kernel, err := a.device.ensureGDNKernel()
	if err != nil {
		return nil, true, err
	}
	elements := input.Elements()
	if elements == 0 {
		return newDeviceTensor(tensorDType(input), input.Shape, 0), true, nil
	}
	channels, height, width := input.Shape[1], input.Shape[2], input.Shape[3]
	inputBuf, err := a.bufferForTensor(input)
	if err != nil {
		return nil, true, err
	}
	betaBuf, err := a.bufferForTensor(beta)
	if err != nil {
		return nil, true, err
	}
	gammaBuf, err := a.bufferForTensor(gamma)
	if err != nil {
		return nil, true, err
	}
	outBuf, err := a.allocStepFloat32(elements)
	if err != nil {
		return nil, true, err
	}
	block := uint(128)
	grid := uint((elements + int(block) - 1) / int(block))
	inverseFlag := 0
	if inverse {
		inverseFlag = 1
	}
	if err := a.device.launchGDN(kernel, grid, block, inputBuf, betaBuf, gammaBuf, outBuf, elements, channels, height, width, inverseFlag); err != nil {
		return nil, true, err
	}
	return newDeviceTensor(tensorDType(input), input.Shape, outBuf), true, nil
}

func (a *imageGradAccelerator) RunGDNBackward(input, beta, gamma, gradOut *backend.Tensor, inverse bool) (*backend.Tensor, *backend.Tensor, *backend.Tensor, bool, error) {
	if a == nil || a.device == nil {
		return nil, nil, nil, false, nil
	}
	if !supportsBuiltinGDN([]*backend.Tensor{input, beta, gamma}) {
		return nil, nil, nil, false, nil
	}
	inputKernel, err := a.device.ensureGDNInputGradKernel()
	if err != nil {
		return nil, nil, nil, true, err
	}
	betaKernel, err := a.device.ensureGDNBetaGradKernel()
	if err != nil {
		return nil, nil, nil, true, err
	}
	gammaKernel, err := a.device.ensureGDNGammaGradKernel()
	if err != nil {
		return nil, nil, nil, true, err
	}
	elements := input.Elements()
	batches, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	inputBuf, err := a.bufferForTensor(input)
	if err != nil {
		return nil, nil, nil, true, err
	}
	betaBuf, err := a.bufferForTensor(beta)
	if err != nil {
		return nil, nil, nil, true, err
	}
	gammaBuf, err := a.bufferForTensor(gamma)
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradOutBuf, err := a.bufferForTensor(gradOut)
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradInBuf, err := a.allocStepFloat32(elements)
	if err != nil {
		return nil, nil, nil, true, err
	}
	gradBetaBuf, err := a.device.allocFloat32(channels)
	if err != nil {
		return nil, nil, nil, true, err
	}
	defer a.device.freeBuffer(gradBetaBuf)
	gammaElements := channels * channels
	gradGammaBuf, err := a.device.allocFloat32(gammaElements)
	if err != nil {
		return nil, nil, nil, true, err
	}
	defer a.device.freeBuffer(gradGammaBuf)
	block := uint(128)
	inverseFlag := 0
	if inverse {
		inverseFlag = 1
	}
	if elements > 0 {
		grid := uint((elements + int(block) - 1) / int(block))
		if err := a.device.launchGDNInputGrad(inputKernel, grid, block, inputBuf, betaBuf, gammaBuf, gradOutBuf, gradInBuf, elements, channels, height, width, inverseFlag); err != nil {
			return nil, nil, nil, true, err
		}
	}
	if channels > 0 {
		grid := uint((channels + int(block) - 1) / int(block))
		if err := a.device.launchGDNBetaGrad(betaKernel, grid, block, inputBuf, betaBuf, gammaBuf, gradOutBuf, gradBetaBuf, channels, height, width, batches, inverseFlag); err != nil {
			return nil, nil, nil, true, err
		}
	}
	if gammaElements > 0 {
		grid := uint((gammaElements + int(block) - 1) / int(block))
		if err := a.device.launchGDNGammaGrad(gammaKernel, grid, block, inputBuf, betaBuf, gammaBuf, gradOutBuf, gradGammaBuf, gammaElements, channels, height, width, batches, inverseFlag); err != nil {
			return nil, nil, nil, true, err
		}
	}
	activeBeta := make([]float32, channels)
	if err := a.device.downloadFloat32(activeBeta, gradBetaBuf); err != nil {
		return nil, nil, nil, true, err
	}
	activeGamma := make([]float32, gammaElements)
	if err := a.device.downloadFloat32(activeGamma, gradGammaBuf); err != nil {
		return nil, nil, nil, true, err
	}
	gradBetaHost := make([]float32, beta.Elements())
	copy(gradBetaHost, activeBeta)
	gradGammaHost := make([]float32, gamma.Elements())
	copy(gradGammaHost, activeGamma)
	gradIn := newDeviceTensor(input.DType, input.Shape, gradInBuf)
	gradBeta := &backend.Tensor{DType: beta.DType, Shape: append([]int(nil), beta.Shape...), F32: gradBetaHost}
	gradGamma := &backend.Tensor{DType: gamma.DType, Shape: append([]int(nil), gamma.Shape...), F32: gradGammaHost}
	return gradIn, gradBeta, gradGamma, true, nil
}

func (a *imageGradAccelerator) Materialize(tensor *backend.Tensor) error {
	if a == nil || a.device == nil || tensor == nil {
		return nil
	}
	if len(tensor.F32) == tensor.Elements() || tensor.Elements() == 0 {
		return nil
	}
	if tensor.Device != backend.DeviceCUDA || tensor.DevicePtr == 0 {
		return nil
	}
	tensor.F32 = make([]float32, tensor.Elements())
	if err := a.device.downloadFloat32(tensor.F32, C.CUdeviceptr(tensor.DevicePtr)); err != nil {
		return err
	}
	tensor.Device = backend.DeviceCPU
	tensor.DevicePtr = 0
	return nil
}

func (a *imageGradAccelerator) ReleaseStep() {
	if a == nil || a.device == nil {
		return
	}
	for _, ptr := range a.owned {
		_ = a.device.freeBuffer(ptr)
	}
	a.owned = nil
	a.residency = nil
}

func (a *imageGradAccelerator) Close() {
	if a == nil || a.device == nil {
		return
	}
	a.ReleaseStep()
	a.device.close()
	a.device = nil
}

func (a *imageGradAccelerator) bufferForTensor(tensor *backend.Tensor) (C.CUdeviceptr, error) {
	if tensor == nil {
		return 0, nil
	}
	if tensor.Device == backend.DeviceCUDA && tensor.DevicePtr != 0 {
		return C.CUdeviceptr(tensor.DevicePtr), nil
	}
	if a.residency == nil {
		a.residency = map[*backend.Tensor]C.CUdeviceptr{}
	}
	if ptr, ok := a.residency[tensor]; ok {
		return ptr, nil
	}
	if len(tensor.F32) < tensor.Elements() {
		return 0, errMissingTensorHostStorage(tensor)
	}
	ptr, err := a.device.uploadFloat32(tensor.F32[:tensor.Elements()])
	if err != nil {
		return 0, err
	}
	a.owned = append(a.owned, ptr)
	a.residency[tensor] = ptr
	return ptr, nil
}

func (a *imageGradAccelerator) allocStepFloat32(elements int) (C.CUdeviceptr, error) {
	ptr, err := a.device.allocFloat32(elements)
	if err != nil {
		return 0, err
	}
	a.owned = append(a.owned, ptr)
	return ptr, nil
}

func newDeviceTensor(dtype string, shape []int, ptr C.CUdeviceptr) *backend.Tensor {
	if dtype == "" {
		dtype = "f32"
	}
	return &backend.Tensor{
		DType:     dtype,
		Shape:     append([]int(nil), shape...),
		Device:    backend.DeviceCUDA,
		DevicePtr: uintptr(ptr),
	}
}

func tensorDType(tensor *backend.Tensor) string {
	dtype := "f32"
	if tensor != nil && tensor.DType != "" {
		dtype = tensor.DType
	}
	return dtype
}

func errMissingTensorHostStorage(tensor *backend.Tensor) error {
	return fmt.Errorf("tensor shape %v is missing host storage", tensor.Shape)
}
