package cuda

import (
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestSupportsBuiltinGDNRequiresNCHWAndParameters(t *testing.T) {
	input := backend.NewTensorF32([]int{1, 2, 1, 2}, []float32{
		1, 2,
		3, 4,
	})
	beta := backend.NewTensorF32([]int{2}, []float32{0.5, 0.75})
	gamma := backend.NewTensorF32([]int{2, 2}, []float32{
		0.1, 0.2,
		0.3, 0.4,
	})

	if !supportsBuiltinGDN([]*backend.Tensor{input, beta, gamma}) {
		t.Fatal("valid NCHW input with beta/gamma should be supported")
	}
	if supportsBuiltinGDN([]*backend.Tensor{backend.NewTensorF32([]int{2, 2}, []float32{1, 2, 3, 4}), beta, gamma}) {
		t.Fatal("rank-2 input should not be supported")
	}
	if supportsBuiltinGDN([]*backend.Tensor{input, backend.NewTensorF32([]int{1}, []float32{0.5}), gamma}) {
		t.Fatal("short beta should not be supported")
	}
	if supportsBuiltinGDN([]*backend.Tensor{input, beta, backend.NewTensorF32([]int{4}, []float32{0.1, 0.2, 0.3, 0.4})}) {
		t.Fatal("flat gamma should not be supported")
	}
	if supportsBuiltinGDN([]*backend.Tensor{input, beta}) {
		t.Fatal("missing gamma should not be supported")
	}
}

func TestPlansBuiltinConvImageSteps(t *testing.T) {
	input := backend.NewTensorF32([]int{1, 2, 4, 4}, make([]float32, 32))
	weight := backend.NewTensorF32([]int{3, 2, 3, 3}, make([]float32, 54))
	bias := backend.NewTensorF32([]int{3}, []float32{0.1, 0.2, 0.3})
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: map[string]string{"stride": "2", "padding": "1"}}
	cfg, ok := planBuiltinConv2D(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("valid NCHW/OIHW conv2d should be supported")
	}
	if cfg.outHeight != 2 || cfg.outWidth != 2 || !cfg.hasBias {
		t.Fatalf("unexpected conv2d config: %+v", cfg)
	}
	if _, ok := planBuiltinConv2D(step, []*backend.Tensor{input, backend.NewTensorF32([]int{3, 1, 3, 3}, make([]float32, 27)), bias}); ok {
		t.Fatal("conv2d input/weight channel mismatch should not be supported")
	}

	transWeight := backend.NewTensorF32([]int{2, 3, 3, 3}, make([]float32, 54))
	transBias := backend.NewTensorF32([]int{3}, []float32{0.1, 0.2, 0.3})
	transStep := mantaartifact.Step{Kind: mantaartifact.StepConv2DTrans, Attributes: map[string]string{"stride": "2", "padding": "1", "output_padding": "1"}}
	transCfg, ok := planBuiltinConv2DTranspose(transStep, []*backend.Tensor{input, transWeight, transBias})
	if !ok {
		t.Fatal("valid NCHW/IOHW conv2d_transpose should be supported")
	}
	if transCfg.outHeight != 8 || transCfg.outWidth != 8 || transCfg.outChannels != 3 || !transCfg.hasBias {
		t.Fatalf("unexpected conv2d_transpose config: %+v", transCfg)
	}
	if _, ok := planBuiltinConv2DTranspose(transStep, []*backend.Tensor{input, backend.NewTensorF32([]int{3, 3, 3, 3}, make([]float32, 81)), transBias}); ok {
		t.Fatal("conv2d_transpose input/weight channel mismatch should not be supported")
	}
}

func TestSupportsBuiltinMirageScalarLossSteps(t *testing.T) {
	lhs := backend.NewTensorF32([]int{2, 2}, []float32{1, 2, 3, 4})
	rhs := backend.NewTensorF32([]int{2, 2}, []float32{1, 1, 1, 1})
	if !supportsBuiltinMSELoss([]*backend.Tensor{lhs, rhs}) {
		t.Fatal("matching dense MSE tensors should be supported")
	}
	if supportsBuiltinMSELoss([]*backend.Tensor{lhs, backend.NewTensorF32([]int{4}, []float32{1, 1, 1, 1})}) {
		t.Fatal("MSE shape mismatch should not be supported")
	}

	rateZ := backend.NewTensorF32([]int{1}, []float32{1})
	rateCoords := backend.NewTensorF32([]int{1}, []float32{2})
	rateNorms := backend.NewTensorF32([]int{1}, []float32{3})
	if !supportsBuiltinScalarAdd([]*backend.Tensor{rateZ, rateCoords, rateNorms}) {
		t.Fatal("scalar rate tensors should be supported")
	}
	if supportsBuiltinScalarAdd([]*backend.Tensor{rateZ, lhs}) {
		t.Fatal("non-scalar add input should not be supported")
	}
	if !supportsBuiltinRDLoss([]*backend.Tensor{rateZ, rateCoords}, 0.01) {
		t.Fatal("valid RD scalar tensors should be supported")
	}
	if supportsBuiltinRDLoss([]*backend.Tensor{rateZ, rateCoords}, -1) {
		t.Fatal("negative RD lambda should not be supported")
	}
}

func TestPlansBuiltinCrossEntropyMirageLayouts(t *testing.T) {
	codes := backend.NewTensorQ4([]int{1, 2, 1, 2}, []float32{0, 1, 2, 3})
	globalLogits := backend.NewTensorF32([]int{16}, make([]float32, 16))
	step := mantaartifact.Step{Kind: mantaartifact.StepCrossEntropy, Attributes: map[string]string{"bits": "4"}}
	plan, ok := planBuiltinCrossEntropy(step, []*backend.Tensor{codes, globalLogits})
	if !ok {
		t.Fatal("global categorical logits should be supported")
	}
	if plan.mode != cudaCrossEntropyCategorical || plan.layout != cudaCrossEntropyLayoutGlobal || plan.levels != 16 {
		t.Fatalf("unexpected global plan: %+v", plan)
	}

	alphabetLogits := backend.NewTensorF32([]int{1, 32, 1, 2}, make([]float32, 64))
	step.Attributes["logits_layout"] = "nchw_alphabet"
	plan, ok = planBuiltinCrossEntropy(step, []*backend.Tensor{codes, alphabetLogits})
	if !ok {
		t.Fatal("NCHW alphabet logits should be supported")
	}
	if plan.layout != cudaCrossEntropyLayoutNCHW {
		t.Fatalf("layout = %v, want NCHW", plan.layout)
	}

	bitLogits := backend.NewTensorF32([]int{1, 16, 1, 2}, make([]float32, 32))
	step.Attributes["factorization"] = "bit-plane"
	plan, ok = planBuiltinCrossEntropy(step, []*backend.Tensor{codes, bitLogits})
	if !ok {
		t.Fatal("NCHW bit-plane logits should be supported")
	}
	if plan.mode != cudaCrossEntropyBitPlane || plan.layout != cudaCrossEntropyLayoutNCHW {
		t.Fatalf("unexpected bit-plane plan: %+v", plan)
	}

	normCodes := backend.NewTensorQNorm([]int{1, 1, 2}, []float32{128, 129})
	normParams := backend.NewTensorF32([]int{1, 2, 1, 2}, []float32{0, 0.1, 0.2, 0.3})
	normStep := mantaartifact.Step{Kind: mantaartifact.StepCrossEntropy, Attributes: map[string]string{"distribution": "log_normal", "sigma_parameter": "softplus"}}
	plan, ok = planBuiltinCrossEntropy(normStep, []*backend.Tensor{normCodes, normParams})
	if !ok {
		t.Fatal("log-normal q_norm params should be supported")
	}
	if plan.mode != cudaCrossEntropyLogNormal || plan.sigmaMode != cudaSigmaSoftplus {
		t.Fatalf("unexpected norm plan: %+v", plan)
	}
	if _, ok := planBuiltinCrossEntropy(normStep, []*backend.Tensor{normCodes}); ok {
		t.Fatal("log-normal mode without params should not be supported")
	}
}
