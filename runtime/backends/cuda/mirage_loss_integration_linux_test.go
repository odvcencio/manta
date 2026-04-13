//go:build linux && cgo

package cuda

import (
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDAMirageScalarLossSteps(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	outputType := scalarF32ValueType()
	lhs := backend.NewTensorF32([]int{3}, []float32{1, 2, 3})
	rhs := backend.NewTensorF32([]int{3}, []float32{1, 4, 7})
	mse, err := rt.runMSELossStep([]*backend.Tensor{lhs, rhs}, outputType)
	if err != nil {
		t.Fatalf("run mse loss: %v", err)
	}
	if mse.VariantEntry != "__builtin_cuda_mse_loss" {
		t.Fatalf("mse variant = %q", mse.VariantEntry)
	}
	if mse.Metadata["device_execution"] != true {
		t.Fatalf("mse device_execution = %v, want true", mse.Metadata["device_execution"])
	}
	assertScalarClose(t, mse.Outputs[0], 20.0/3.0, 0.0005)

	msssim, err := rt.runMSSSIMLossStep([]*backend.Tensor{lhs, rhs}, outputType)
	if err != nil {
		t.Fatalf("run ms-ssim loss: %v", err)
	}
	if msssim.VariantEntry != "__builtin_cuda_ms_ssim_loss" {
		t.Fatalf("ms-ssim variant = %q", msssim.VariantEntry)
	}
	assertScalarClose(t, msssim.Outputs[0], expectedMSSSIMLoss(lhs, rhs), 0.0005)

	sum, err := rt.runScalarAddStep([]*backend.Tensor{
		backend.NewTensorF32([]int{1}, []float32{1.5}),
		backend.NewTensorF32([]int{1}, []float32{2}),
		backend.NewTensorF32([]int{1}, []float32{3}),
	}, outputType)
	if err != nil {
		t.Fatalf("run scalar add: %v", err)
	}
	if sum.VariantEntry != "__builtin_cuda_scalar_add" {
		t.Fatalf("scalar add variant = %q", sum.VariantEntry)
	}
	assertScalarClose(t, sum.Outputs[0], 6.5, 0.0005)

	rd, err := rt.runRDLossStep([]*backend.Tensor{
		backend.NewTensorF32([]int{1}, []float32{2}),
		backend.NewTensorF32([]int{1}, []float32{5}),
	}, outputType, 0.1)
	if err != nil {
		t.Fatalf("run rd loss: %v", err)
	}
	if rd.VariantEntry != "__builtin_cuda_rate_distortion_loss" {
		t.Fatalf("rd variant = %q", rd.VariantEntry)
	}
	assertScalarClose(t, rd.Outputs[0], 2.5, 0.0005)
}

func TestCUDACrossEntropyStepMatchesHostMirageLayouts(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	outputType := scalarF32ValueType()
	globalCodes := backend.NewTensorQ2([]int{4}, []float32{0, 1, 2, 3})
	globalLogits := backend.NewTensorF32([]int{4}, []float32{0, 0.5, 1, -0.5})
	globalStep := mantaartifact.Step{Kind: mantaartifact.StepCrossEntropy, Attributes: map[string]string{"bits": "2"}}
	globalPlan, ok := planBuiltinCrossEntropy(globalStep, []*backend.Tensor{globalCodes, globalLogits})
	if !ok {
		t.Fatal("global categorical cross entropy should be planned")
	}
	global, err := rt.runCrossEntropyStep([]*backend.Tensor{globalCodes, globalLogits}, outputType, globalPlan)
	if err != nil {
		t.Fatalf("run global cross entropy: %v", err)
	}
	assertScalarClose(t, global.Outputs[0], expectedCategoricalGlobal(globalCodes, globalLogits, 4), 0.0005)

	alphabetCodes := backend.NewTensorQ2([]int{1, 2, 1, 2}, []float32{0, 1, 2, 3})
	alphabetLogits := backend.NewTensorF32([]int{1, 8, 1, 2}, make([]float32, 16))
	for c := 0; c < 2; c++ {
		for x := 0; x < 2; x++ {
			for level := 0; level < 4; level++ {
				alphabetLogits.F32[offset4(alphabetLogits.Shape, 0, c*4+level, 0, x)] = float32(0.2*float64(c+1) + 0.3*float64(level) - 0.1*float64(x))
			}
		}
	}
	alphabetStep := mantaartifact.Step{Kind: mantaartifact.StepCrossEntropy, Attributes: map[string]string{"bits": "2", "logits_layout": "nchw_alphabet"}}
	alphabetPlan, ok := planBuiltinCrossEntropy(alphabetStep, []*backend.Tensor{alphabetCodes, alphabetLogits})
	if !ok {
		t.Fatal("NCHW alphabet cross entropy should be planned")
	}
	alphabet, err := rt.runCrossEntropyStep([]*backend.Tensor{alphabetCodes, alphabetLogits}, outputType, alphabetPlan)
	if err != nil {
		t.Fatalf("run NCHW alphabet cross entropy: %v", err)
	}
	assertScalarClose(t, alphabet.Outputs[0], expectedCategoricalNCHW(alphabetCodes, alphabetLogits, 4), 0.0005)

	bitCodes := backend.NewTensorQ2([]int{1, 1, 1, 3}, []float32{0, 1, 3})
	bitLogits := backend.NewTensorF32([]int{1, 4, 1, 3}, []float32{
		0.2, -0.1, 0.4,
		-0.2, 0.3, -0.5,
		0.6, -0.4, 0.1,
		-0.3, 0.2, -0.2,
	})
	bitStep := mantaartifact.Step{Kind: mantaartifact.StepCrossEntropy, Attributes: map[string]string{"bits": "2", "factorization": "bit-plane", "logits_layout": "nchw_alphabet"}}
	bitPlan, ok := planBuiltinCrossEntropy(bitStep, []*backend.Tensor{bitCodes, bitLogits})
	if !ok {
		t.Fatal("NCHW bit-plane cross entropy should be planned")
	}
	bit, err := rt.runCrossEntropyStep([]*backend.Tensor{bitCodes, bitLogits}, outputType, bitPlan)
	if err != nil {
		t.Fatalf("run NCHW bit-plane cross entropy: %v", err)
	}
	assertScalarClose(t, bit.Outputs[0], expectedBitPlaneNCHW(bitCodes, bitLogits, 2), 0.0005)

	normCodes := backend.NewTensorQNorm([]int{1, 1, 3}, []float32{0, 128, 255})
	normParams := backend.NewTensorF32([]int{1, 2, 1, 3}, []float32{
		-1, 0, 1,
		0.5, 1, -0.5,
	})
	normStep := mantaartifact.Step{Kind: mantaartifact.StepCrossEntropy, Attributes: map[string]string{"distribution": "log_normal", "sigma_parameter": "softplus"}}
	normPlan, ok := planBuiltinCrossEntropy(normStep, []*backend.Tensor{normCodes, normParams})
	if !ok {
		t.Fatal("log-normal cross entropy should be planned")
	}
	norm, err := rt.runCrossEntropyStep([]*backend.Tensor{normCodes, normParams}, outputType, normPlan)
	if err != nil {
		t.Fatalf("run log-normal cross entropy: %v", err)
	}
	assertScalarClose(t, norm.Outputs[0], expectedLogNormal(normCodes, normParams), 0.002)
}

func scalarF32ValueType() mantaartifact.ValueType {
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f32"},
	}
}

func assertScalarClose(t *testing.T, tensor *backend.Tensor, want float64, tolerance float64) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != 1 || tensor.Shape[0] != 1 || len(tensor.F32) != 1 {
		t.Fatalf("unexpected scalar tensor: %+v", tensor)
	}
	got := float64(tensor.F32[0])
	if math.Abs(got-want) > tolerance {
		t.Fatalf("scalar = %.8f, want %.8f", got, want)
	}
}

func expectedCategoricalGlobal(codes, logits *backend.Tensor, levels int) float64 {
	total := 0.0
	for _, raw := range codes.F32 {
		idx := clampIntForTest(int(math.Round(float64(raw))), 0, levels-1)
		total += -math.Log2(softmaxProbabilityForTest(logits.F32, 0, 1, levels, idx))
	}
	return total
}

func expectedCategoricalNCHW(codes, logits *backend.Tensor, levels int) float64 {
	total := 0.0
	for i, raw := range codes.F32 {
		n, c, h, w := unpackOffset4ForTest(codes.Shape, i)
		idx := clampIntForTest(int(math.Round(float64(raw))), 0, levels-1)
		base := offset4(logits.Shape, n, c*levels, h, w)
		total += -math.Log2(softmaxProbabilityForTest(logits.F32, base, logits.Shape[2]*logits.Shape[3], levels, idx))
	}
	return total
}

func expectedBitPlaneNCHW(codes, logits *backend.Tensor, bits int) float64 {
	total := 0.0
	for i, raw := range codes.F32 {
		n, c, h, w := unpackOffset4ForTest(codes.Shape, i)
		symbol := clampIntForTest(int(math.Round(float64(raw))), 0, (1<<bits)-1)
		for bit := 0; bit < bits; bit++ {
			shift := bits - 1 - bit
			bitValue := (symbol >> shift) & 1
			base := offset4(logits.Shape, n, c*bits*2+bit*2, h, w)
			total += -math.Log2(softmaxProbabilityForTest(logits.F32, base, logits.Shape[2]*logits.Shape[3], 2, bitValue))
		}
	}
	return total
}

func expectedLogNormal(codes, params *backend.Tensor) float64 {
	total := 0.0
	for i, raw := range codes.F32 {
		n, h, w := unpackOffset3ForTest(codes.Shape, i)
		mu := float64(params.F32[offset4(params.Shape, n, 0, h, w)])
		sigmaRaw := float64(params.F32[offset4(params.Shape, n, 1, h, w)])
		sigma := math.Log1p(math.Exp(sigmaRaw)) + 1e-6
		sym := clampIntForTest(int(math.Round(float64(raw))), 0, 255)
		lo, hi := qNormLogBoundsForTest(sym)
		p := normalCDFForTest((hi-mu)/sigma) - normalCDFForTest((lo-mu)/sigma)
		if p < 1e-12 {
			p = 1e-12
		}
		total += -math.Log2(p)
	}
	return total
}

func expectedMSSSIMLoss(lhs, rhs *backend.Tensor) float64 {
	meanA, meanB := 0.0, 0.0
	for i := range lhs.F32 {
		meanA += float64(lhs.F32[i])
		meanB += float64(rhs.F32[i])
	}
	meanA /= float64(len(lhs.F32))
	meanB /= float64(len(rhs.F32))
	varA, varB, cov := 0.0, 0.0, 0.0
	for i := range lhs.F32 {
		da := float64(lhs.F32[i]) - meanA
		db := float64(rhs.F32[i]) - meanB
		varA += da * da
		varB += db * db
		cov += da * db
	}
	denom := float64(len(lhs.F32))
	varA /= denom
	varB /= denom
	cov /= denom
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

func softmaxProbabilityForTest(values []float32, base, stride, count, idx int) float64 {
	maxValue := float64(values[base])
	for i := 1; i < count; i++ {
		value := float64(values[base+i*stride])
		if value > maxValue {
			maxValue = value
		}
	}
	sum := 0.0
	target := 0.0
	for i := 0; i < count; i++ {
		value := math.Exp(float64(values[base+i*stride]) - maxValue)
		if i == idx {
			target = value
		}
		sum += value
	}
	return target / sum
}

func qNormLogBoundsForTest(sym int) (float64, float64) {
	span := 32.0
	switch sym {
	case 0:
		return math.Inf(-1), -16.0 + (0.5/255)*span
	case 255:
		return -16.0 + (254.5/255)*span, math.Inf(1)
	default:
		lower := (float64(sym) - 0.5) / 255
		upper := (float64(sym) + 0.5) / 255
		return -16.0 + lower*span, -16.0 + upper*span
	}
}

func normalCDFForTest(x float64) float64 {
	switch {
	case math.IsInf(x, -1):
		return 0
	case math.IsInf(x, 1):
		return 1
	default:
		return 0.5 * (1 + math.Erf(x/math.Sqrt2))
	}
}

func offset4(shape []int, n, c, h, w int) int {
	return ((n*shape[1]+c)*shape[2]+h)*shape[3] + w
}

func unpackOffset4ForTest(shape []int, offset int) (int, int, int, int) {
	w := offset % shape[3]
	offset /= shape[3]
	h := offset % shape[2]
	offset /= shape[2]
	c := offset % shape[1]
	n := offset / shape[1]
	return n, c, h, w
}

func unpackOffset3ForTest(shape []int, offset int) (int, int, int) {
	w := offset % shape[2]
	offset /= shape[2]
	h := offset % shape[1]
	n := offset / shape[1]
	return n, h, w
}

func clampIntForTest(value, lo, hi int) int {
	if value < lo {
		return lo
	}
	if value > hi {
		return hi
	}
	return value
}
