package cuda

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"sync"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

type cachedLoad struct {
	compiled map[string]backend.CompiledKernel
	native   map[string]backend.NativeKernelProgram
	device   *deviceRuntime
}

// Backend is the CUDA backend stub.
type Backend struct {
	mu          sync.Mutex
	loadCache   map[string]cachedLoad
	cacheHits   int
	cacheMisses int
}

// New returns a CUDA backend.
func New() *Backend {
	return &Backend{loadCache: map[string]cachedLoad{}}
}

// Kind reports the backend identity.
func (b *Backend) Kind() mantaartifact.BackendKind {
	return mantaartifact.BackendCUDA
}

// Capabilities reports the runtime features the CUDA backend supports.
func (b *Backend) Capabilities() []string {
	return []string{
		mantaartifact.CapabilityCandidatePack,
		mantaartifact.CapabilityKVCache,
		mantaartifact.CapabilityMaskedMeanPool,
		mantaartifact.CapabilityHostFallback,
		mantaartifact.CapabilityDeviceExecution,
		mantaartifact.CapabilityImageOps,
		mantaartifact.CapabilityTrainingLosses,
		mantaartifact.CapabilityTurboQuant,
	}
}

// CanLoad reports whether the module allows CUDA execution.
func (b *Backend) CanLoad(mod *mantaartifact.Module) bool {
	return mod != nil && mod.SupportsBackend(mantaartifact.BackendCUDA)
}

// Load prepares a CUDA executor stub.
func (b *Backend) Load(_ context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding) (backend.Executor, error) {
	return b.load(context.Background(), mod, weights, "")
}

func (b *Backend) LoadWithCacheKey(ctx context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	return b.load(ctx, mod, weights, cacheKey)
}

func (b *Backend) load(_ context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	if cacheKey != "" {
		if cached, ok := b.cachedLoad(cacheKey); ok {
			return &executor{module: mod, weights: weights, compiled: cached.compiled, native: cached.native, device: cached.device}, nil
		}
	}
	compiled, err := backend.CompileVariants(mod, mantaartifact.BackendCUDA)
	if err != nil {
		return nil, err
	}
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	native := map[string]backend.NativeKernelProgram{}
	for _, kernel := range mod.Kernels {
		prog, err := backend.CompileNativeKernelProgram(mantaartifact.BackendCUDA, kernel, compiled[kernel.Name])
		if err != nil {
			if device != nil {
				device.close()
			}
			return nil, err
		}
		if device != nil {
			if err := device.attachDeviceExecution(&prog, kernel); err != nil {
				device.close()
				return nil, err
			}
		}
		native[kernel.Name] = prog
	}
	if cacheKey != "" {
		b.storeCachedLoad(cacheKey, cachedLoad{compiled: compiled, native: native, device: device})
	}
	return &executor{module: mod, weights: weights, compiled: compiled, native: native, device: device}, nil
}

func (b *Backend) cachedLoad(cacheKey string) (cachedLoad, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	cached, ok := b.loadCache[cacheKey]
	if ok {
		b.cacheHits++
	} else {
		b.cacheMisses++
	}
	return cached, ok
}

func (b *Backend) storeCachedLoad(cacheKey string, cached cachedLoad) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.loadCache[cacheKey] = cached
}

type executor struct {
	module   *mantaartifact.Module
	weights  map[string]backend.WeightBinding
	compiled map[string]backend.CompiledKernel
	native   map[string]backend.NativeKernelProgram
	device   *deviceRuntime
}

func (e *executor) Backend() mantaartifact.BackendKind {
	return mantaartifact.BackendCUDA
}

func (e *executor) Run(ctx context.Context, req backend.Request) (backend.Result, error) {
	return backend.ExecuteSymbolic(ctx, e.module, e.weights, e.compiled, e.dispatchKernel, e.dispatchStep, mantaartifact.BackendCUDA, req)
}

func (e *executor) dispatchKernel(_ context.Context, kernel mantaartifact.Kernel, inputs []*backend.Tensor) (backend.KernelDispatchResult, error) {
	prog, ok := e.native[kernel.Name]
	if !ok {
		return backend.KernelDispatchResult{}, fmt.Errorf("CUDA kernel %q is not compiled", kernel.Name)
	}
	meta := cloneLaunchConfig(prog.LaunchConfig)
	runner := prog.Run
	if shouldFallbackScoreKernel(kernel, inputs) && prog.Fallback != nil {
		runner = prog.Fallback
		meta["device_execution"] = false
		meta["dispatch_mode"] = "host_fallback"
		meta["execution_mode"] = "host_fallback"
		meta["launch_api"] = "host_reference"
		meta["fallback_reason"] = "unsupported_input_shape"
	}
	out, err := runner(inputs)
	if err != nil {
		return backend.KernelDispatchResult{}, err
	}
	return backend.KernelDispatchResult{
		Outputs:      out,
		VariantEntry: prog.Compiled.Entry,
		SourceHash:   prog.Compiled.SourceHash,
		Metadata:     meta,
	}, nil
}

func (e *executor) dispatchStep(_ context.Context, step mantaartifact.Step, outputType mantaartifact.ValueType, inputs []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	switch step.Kind {
	case mantaartifact.StepMatMul:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinMatMul(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runMatMul(inputs, outputType)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case mantaartifact.StepConv2D:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		cfg, ok := planBuiltinConv2D(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runConv2DStep(inputs, outputType, cfg)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case mantaartifact.StepConv2DTrans:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		cfg, ok := planBuiltinConv2DTranspose(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runConv2DTransposeStep(inputs, outputType, cfg)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case mantaartifact.StepGDN, mantaartifact.StepIGDN:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinGDN(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runGDNStep(inputs, outputType, step.Kind == mantaartifact.StepIGDN)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case mantaartifact.StepMSELoss:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinMSELoss(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runMSELossStep(inputs, outputType)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case mantaartifact.StepScalarAdd:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinScalarAdd(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runScalarAddStep(inputs, outputType)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case mantaartifact.StepRDLoss:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		lambda := stepAttrFloat32(step.Attributes, "lambda", 1)
		if !supportsBuiltinRDLoss(inputs, lambda) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runRDLossStep(inputs, outputType, lambda)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case mantaartifact.StepCrossEntropy:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		plan, ok := planBuiltinCrossEntropy(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runCrossEntropyStep(inputs, outputType, plan)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	default:
		return backend.StepDispatchResult{}, false, nil
	}
}

func supportsBuiltinMatMul(inputs []*backend.Tensor) bool {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return false
	}
	lhs := inputs[0]
	rhs := inputs[1]
	switch len(lhs.Shape) {
	case 2:
		return len(rhs.Shape) == 2 && lhs.Shape[1] == rhs.Shape[0]
	case 3:
		switch len(rhs.Shape) {
		case 2:
			return lhs.Shape[2] == rhs.Shape[0]
		case 3:
			return lhs.Shape[0] == rhs.Shape[0] && lhs.Shape[2] == rhs.Shape[1]
		default:
			return false
		}
	default:
		return false
	}
}

func supportsBuiltinGDN(inputs []*backend.Tensor) bool {
	if len(inputs) < 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return false
	}
	input, beta, gamma := inputs[0], inputs[1], inputs[2]
	if len(input.Shape) != 4 || len(input.F32) != input.Elements() {
		return false
	}
	channels := input.Shape[1]
	if len(beta.F32) < channels {
		return false
	}
	if len(gamma.Shape) != 2 || gamma.Shape[0] != channels || gamma.Shape[1] != channels || len(gamma.F32) < channels*channels {
		return false
	}
	return true
}

type cudaConv2DConfig struct {
	batches     int
	inChannels  int
	inHeight    int
	inWidth     int
	outChannels int
	outHeight   int
	outWidth    int
	inPerGroup  int
	outPerGroup int
	kernelH     int
	kernelW     int
	groups      int
	strideH     int
	strideW     int
	padH        int
	padW        int
	dilationH   int
	dilationW   int
	hasBias     bool
}

type cudaConv2DTransposeConfig struct {
	batches     int
	inChannels  int
	inHeight    int
	inWidth     int
	outChannels int
	outHeight   int
	outWidth    int
	outPerGroup int
	inPerGroup  int
	kernelH     int
	kernelW     int
	groups      int
	strideH     int
	strideW     int
	padH        int
	padW        int
	dilationH   int
	dilationW   int
	outPadH     int
	outPadW     int
	hasBias     bool
}

func planBuiltinConv2D(step mantaartifact.Step, inputs []*backend.Tensor) (cudaConv2DConfig, bool) {
	if len(inputs) < 2 || len(inputs) > 3 || inputs[0] == nil || inputs[1] == nil {
		return cudaConv2DConfig{}, false
	}
	input, weight := inputs[0], inputs[1]
	if len(input.Shape) != 4 || len(weight.Shape) != 4 || len(input.F32) != input.Elements() || len(weight.F32) != weight.Elements() {
		return cudaConv2DConfig{}, false
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	outC, inPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	groups := stepAttrInt(step.Attributes, "groups", 1)
	if groups <= 0 || inPerGroup*groups != inC || outC%groups != 0 {
		return cudaConv2DConfig{}, false
	}
	strideH := stepAttrInt(step.Attributes, "stride_h", stepAttrInt(step.Attributes, "stride", 1))
	strideW := stepAttrInt(step.Attributes, "stride_w", stepAttrInt(step.Attributes, "stride", 1))
	padH := stepAttrInt(step.Attributes, "pad_h", stepAttrInt(step.Attributes, "padding", 0))
	padW := stepAttrInt(step.Attributes, "pad_w", stepAttrInt(step.Attributes, "padding", 0))
	dilationH := stepAttrInt(step.Attributes, "dilation_h", stepAttrInt(step.Attributes, "dilation", 1))
	dilationW := stepAttrInt(step.Attributes, "dilation_w", stepAttrInt(step.Attributes, "dilation", 1))
	if strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 {
		return cudaConv2DConfig{}, false
	}
	outH := (inH+2*padH-dilationH*(kH-1)-1)/strideH + 1
	outW := (inW+2*padW-dilationW*(kW-1)-1)/strideW + 1
	if outH <= 0 || outW <= 0 {
		return cudaConv2DConfig{}, false
	}
	hasBias := len(inputs) == 3 && inputs[2] != nil
	if hasBias && len(inputs[2].F32) < outC {
		return cudaConv2DConfig{}, false
	}
	return cudaConv2DConfig{
		batches:     n,
		inChannels:  inC,
		inHeight:    inH,
		inWidth:     inW,
		outChannels: outC,
		outHeight:   outH,
		outWidth:    outW,
		inPerGroup:  inPerGroup,
		outPerGroup: outC / groups,
		kernelH:     kH,
		kernelW:     kW,
		groups:      groups,
		strideH:     strideH,
		strideW:     strideW,
		padH:        padH,
		padW:        padW,
		dilationH:   dilationH,
		dilationW:   dilationW,
		hasBias:     hasBias,
	}, true
}

func planBuiltinConv2DTranspose(step mantaartifact.Step, inputs []*backend.Tensor) (cudaConv2DTransposeConfig, bool) {
	if len(inputs) < 2 || len(inputs) > 3 || inputs[0] == nil || inputs[1] == nil {
		return cudaConv2DTransposeConfig{}, false
	}
	input, weight := inputs[0], inputs[1]
	if len(input.Shape) != 4 || len(weight.Shape) != 4 || len(input.F32) != input.Elements() || len(weight.F32) != weight.Elements() {
		return cudaConv2DTransposeConfig{}, false
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	weightInC, outPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	if weightInC != inC {
		return cudaConv2DTransposeConfig{}, false
	}
	groups := stepAttrInt(step.Attributes, "groups", 1)
	if groups <= 0 || inC%groups != 0 {
		return cudaConv2DTransposeConfig{}, false
	}
	strideH := stepAttrInt(step.Attributes, "stride_h", stepAttrInt(step.Attributes, "stride", 1))
	strideW := stepAttrInt(step.Attributes, "stride_w", stepAttrInt(step.Attributes, "stride", 1))
	padH := stepAttrInt(step.Attributes, "pad_h", stepAttrInt(step.Attributes, "padding", 0))
	padW := stepAttrInt(step.Attributes, "pad_w", stepAttrInt(step.Attributes, "padding", 0))
	dilationH := stepAttrInt(step.Attributes, "dilation_h", stepAttrInt(step.Attributes, "dilation", 1))
	dilationW := stepAttrInt(step.Attributes, "dilation_w", stepAttrInt(step.Attributes, "dilation", 1))
	outPadH := stepAttrInt(step.Attributes, "output_padding_h", stepAttrInt(step.Attributes, "output_padding", 0))
	outPadW := stepAttrInt(step.Attributes, "output_padding_w", stepAttrInt(step.Attributes, "output_padding", 0))
	if strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 {
		return cudaConv2DTransposeConfig{}, false
	}
	outC := outPerGroup * groups
	outH := (inH-1)*strideH - 2*padH + dilationH*(kH-1) + outPadH + 1
	outW := (inW-1)*strideW - 2*padW + dilationW*(kW-1) + outPadW + 1
	if outH <= 0 || outW <= 0 {
		return cudaConv2DTransposeConfig{}, false
	}
	hasBias := len(inputs) == 3 && inputs[2] != nil
	if hasBias && len(inputs[2].F32) < outC {
		return cudaConv2DTransposeConfig{}, false
	}
	return cudaConv2DTransposeConfig{
		batches:     n,
		inChannels:  inC,
		inHeight:    inH,
		inWidth:     inW,
		outChannels: outC,
		outHeight:   outH,
		outWidth:    outW,
		outPerGroup: outPerGroup,
		inPerGroup:  inC / groups,
		kernelH:     kH,
		kernelW:     kW,
		groups:      groups,
		strideH:     strideH,
		strideW:     strideW,
		padH:        padH,
		padW:        padW,
		dilationH:   dilationH,
		dilationW:   dilationW,
		outPadH:     outPadH,
		outPadW:     outPadW,
		hasBias:     hasBias,
	}, true
}

func supportsBuiltinMSELoss(inputs []*backend.Tensor) bool {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return false
	}
	lhs, rhs := inputs[0], inputs[1]
	return lhs.EqualShape(rhs) && len(lhs.F32) == len(rhs.F32) && len(lhs.F32) == lhs.Elements()
}

func supportsBuiltinScalarAdd(inputs []*backend.Tensor) bool {
	if len(inputs) == 0 {
		return false
	}
	for _, input := range inputs {
		if input == nil || len(input.F32) != 1 {
			return false
		}
	}
	return true
}

func supportsBuiltinRDLoss(inputs []*backend.Tensor, lambda float32) bool {
	return supportsBuiltinScalarAdd(inputs) && len(inputs) == 2 && !math.IsNaN(float64(lambda)) && !math.IsInf(float64(lambda), 0) && lambda >= 0
}

type cudaCrossEntropyMode int

const (
	cudaCrossEntropyCategorical cudaCrossEntropyMode = iota
	cudaCrossEntropyBitPlane
	cudaCrossEntropyLogNormal
)

type cudaCrossEntropyLayout int

const (
	cudaCrossEntropyLayoutUniform cudaCrossEntropyLayout = iota
	cudaCrossEntropyLayoutGlobal
	cudaCrossEntropyLayoutFlat
	cudaCrossEntropyLayoutNCHW
	cudaCrossEntropyLayoutSigmoidFallback
)

type cudaSigmaMode int

const (
	cudaSigmaRaw cudaSigmaMode = iota
	cudaSigmaSoftplus
	cudaSigmaExp
)

type cudaCrossEntropyPlan struct {
	mode      cudaCrossEntropyMode
	layout    cudaCrossEntropyLayout
	levels    int
	bits      int
	sigmaMode cudaSigmaMode
}

func planBuiltinCrossEntropy(step mantaartifact.Step, inputs []*backend.Tensor) (cudaCrossEntropyPlan, bool) {
	if len(inputs) < 1 || inputs[0] == nil {
		return cudaCrossEntropyPlan{}, false
	}
	codes := inputs[0]
	if len(codes.F32) != codes.Elements() {
		return cudaCrossEntropyPlan{}, false
	}
	var logits *backend.Tensor
	if len(inputs) > 1 {
		logits = inputs[1]
	}
	attrs := step.Attributes
	if attrs != nil && attrs["distribution"] == "log_normal" {
		sigmaMode, ok := cudaSigmaModeForAttrs(attrs)
		if !ok || logits == nil || len(codes.Shape) != 3 || len(logits.Shape) != 4 {
			return cudaCrossEntropyPlan{}, false
		}
		if logits.Shape[0] != codes.Shape[0] || logits.Shape[1] < 2 || logits.Shape[2] != codes.Shape[1] || logits.Shape[3] != codes.Shape[2] {
			return cudaCrossEntropyPlan{}, false
		}
		if len(logits.F32) < logits.Elements() {
			return cudaCrossEntropyPlan{}, false
		}
		return cudaCrossEntropyPlan{mode: cudaCrossEntropyLogNormal, layout: cudaCrossEntropyLayoutNCHW, levels: 256, bits: 8, sigmaMode: sigmaMode}, true
	}

	bits := stepAttrInt(attrs, "bits", cudaBitsForQTensor(codes))
	levels := stepAttrInt(attrs, "levels", 0)
	if levels <= 0 {
		if bits > 0 {
			levels = 1 << bits
		} else {
			levels = 256
		}
	}
	if levels <= 0 {
		return cudaCrossEntropyPlan{}, false
	}
	if cudaFactorizationAttr(attrs) == "bit-plane" {
		if bits <= 0 || bits > 16 {
			return cudaCrossEntropyPlan{}, false
		}
		layout, ok := cudaCrossEntropyBitLayout(codes, logits, bits)
		if !ok {
			return cudaCrossEntropyPlan{}, false
		}
		return cudaCrossEntropyPlan{mode: cudaCrossEntropyBitPlane, layout: layout, levels: levels, bits: bits}, true
	}
	layout, ok := cudaCrossEntropyCategoricalLayout(codes, logits, levels, attrs)
	if !ok {
		return cudaCrossEntropyPlan{}, false
	}
	return cudaCrossEntropyPlan{mode: cudaCrossEntropyCategorical, layout: layout, levels: levels, bits: bits}, true
}

func cudaCrossEntropyCategoricalLayout(codes, logits *backend.Tensor, levels int, attrs map[string]string) (cudaCrossEntropyLayout, bool) {
	if logits == nil || len(logits.F32) == 0 {
		return cudaCrossEntropyLayoutUniform, true
	}
	if len(logits.F32) == levels {
		return cudaCrossEntropyLayoutGlobal, true
	}
	if attrs != nil && attrs["logits_layout"] == "nchw_alphabet" && supportsNCHWAlphabet(codes, logits, levels) {
		return cudaCrossEntropyLayoutNCHW, true
	}
	if len(logits.F32) >= codes.Elements()*levels {
		return cudaCrossEntropyLayoutFlat, true
	}
	if len(logits.F32) > 0 {
		return cudaCrossEntropyLayoutSigmoidFallback, true
	}
	return cudaCrossEntropyLayoutUniform, true
}

func cudaCrossEntropyBitLayout(codes, logits *backend.Tensor, bits int) (cudaCrossEntropyLayout, bool) {
	if logits == nil || len(logits.F32) == 0 {
		return cudaCrossEntropyLayoutUniform, true
	}
	if supportsNCHWBitPair(codes, logits, bits) {
		return cudaCrossEntropyLayoutNCHW, true
	}
	if len(logits.F32) == bits*2 {
		return cudaCrossEntropyLayoutGlobal, true
	}
	if len(logits.F32) >= codes.Elements()*bits*2 {
		return cudaCrossEntropyLayoutFlat, true
	}
	if len(logits.F32) > 0 {
		return cudaCrossEntropyLayoutSigmoidFallback, true
	}
	return cudaCrossEntropyLayoutUniform, true
}

func supportsNCHWAlphabet(codes, logits *backend.Tensor, levels int) bool {
	if codes == nil || logits == nil || len(codes.Shape) != 4 || len(logits.Shape) != 4 {
		return false
	}
	return logits.Shape[0] == codes.Shape[0] &&
		logits.Shape[1] >= codes.Shape[1]*levels &&
		logits.Shape[2] == codes.Shape[2] &&
		logits.Shape[3] == codes.Shape[3] &&
		len(logits.F32) >= logits.Elements()
}

func supportsNCHWBitPair(codes, logits *backend.Tensor, bits int) bool {
	if codes == nil || logits == nil || len(codes.Shape) != 4 || len(logits.Shape) != 4 {
		return false
	}
	return logits.Shape[0] == codes.Shape[0] &&
		logits.Shape[1] >= codes.Shape[1]*bits*2 &&
		logits.Shape[2] == codes.Shape[2] &&
		logits.Shape[3] == codes.Shape[3] &&
		len(logits.F32) >= logits.Elements()
}

func cudaSigmaModeForAttrs(attrs map[string]string) (cudaSigmaMode, bool) {
	if attrs == nil {
		return cudaSigmaRaw, true
	}
	switch attrs["sigma_parameter"] {
	case "":
		return cudaSigmaRaw, true
	case "softplus":
		return cudaSigmaSoftplus, true
	case "exp":
		return cudaSigmaExp, true
	default:
		return cudaSigmaRaw, false
	}
}

func cudaFactorizationAttr(attrs map[string]string) string {
	if attrs == nil {
		return "categorical"
	}
	switch attrs["factorization"] {
	case "bit-plane", "bitplane", "bit_plane":
		return "bit-plane"
	default:
		return "categorical"
	}
}

func cudaBitsForQTensor(t *backend.Tensor) int {
	if t == nil {
		return 0
	}
	switch t.DType {
	case "q2":
		return 2
	case "q4":
		return 4
	case "q8", "q_norm":
		return 8
	default:
		return 0
	}
}

func stepAttrInt(attrs map[string]string, key string, fallback int) int {
	if attrs == nil {
		return fallback
	}
	raw := attrs[key]
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func stepAttrFloat32(attrs map[string]string, key string, fallback float32) float32 {
	if attrs == nil {
		return fallback
	}
	raw := attrs[key]
	if raw == "" {
		return fallback
	}
	value, err := strconv.ParseFloat(raw, 32)
	if err != nil {
		return fallback
	}
	return float32(value)
}

func shouldFallbackScoreKernel(kernel mantaartifact.Kernel, inputs []*backend.Tensor) bool {
	if len(kernel.Body) == 0 {
		return false
	}
	switch kernel.Body[0].Op {
	case "dot", "cosine", "l2_distance":
		if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
			return false
		}
		query := inputs[0]
		docs := inputs[1]
		return !(len(query.Shape) == 1 && len(docs.Shape) == 2 && query.Shape[0] == docs.Shape[1])
	default:
		return false
	}
}

func cloneLaunchConfig(in map[string]any) map[string]any {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
