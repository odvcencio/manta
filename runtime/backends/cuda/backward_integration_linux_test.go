//go:build linux && cgo

package cuda

import (
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDAConv2DBackwardMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{2, 4, 7, 8}, patternedFloats(2*4*7*8, 0.021))
	weight := backend.NewTensorF32([]int{6, 2, 3, 2}, patternedFloats(6*2*3*2, -0.033))
	bias := backend.NewTensorF32([]int{6}, patternedFloats(6, 0.017))
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: map[string]string{
		"groups":   "2",
		"stride_h": "2",
		"stride_w": "1",
		"pad_h":    "1",
		"pad_w":    "0",
	}}
	cfg, ok := planBuiltinConv2D(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d should be supported")
	}
	gradOut := backend.NewTensorF32([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, patternedFloats(cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth, 0.027))

	wantIn, wantW, wantB := hostConv2DBackward(input, weight, bias, gradOut, cfg)
	gotIn, gotW, gotB, err := rt.runConv2DBackward(input, weight, bias, gradOut, cfg)
	if err != nil {
		t.Fatalf("run conv2d backward: %v", err)
	}
	assertTensorClose(t, gotIn, input.Shape, wantIn)
	assertTensorClose(t, gotW, weight.Shape, wantW)
	assertTensorClose(t, gotB, bias.Shape, wantB)
}

func TestCUDAConv2DBackwardMatchesFiniteDifference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{1, 2, 4, 5}, patternedFloats(1*2*4*5, 0.019))
	weight := backend.NewTensorF32([]int{3, 2, 2, 3}, patternedFloats(3*2*2*3, -0.023))
	bias := backend.NewTensorF32([]int{3}, patternedFloats(3, 0.011))
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: map[string]string{
		"stride_h": "1",
		"stride_w": "2",
		"pad_h":    "1",
		"pad_w":    "1",
	}}
	cfg, ok := planBuiltinConv2D(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d should be supported")
	}
	gradOut := backend.NewTensorF32([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, patternedFloats(cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth, 0.031))
	gotIn, gotW, gotB, err := rt.runConv2DBackward(input, weight, bias, gradOut, cfg)
	if err != nil {
		t.Fatalf("run conv2d backward: %v", err)
	}

	const eps = 1e-3
	assertFiniteDiffTensor(t, "input", gotIn, input, eps, func() float64 {
		return conv2DFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
	assertFiniteDiffTensor(t, "weight", gotW, weight, eps, func() float64 {
		return conv2DFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
	assertFiniteDiffTensor(t, "bias", gotB, bias, eps, func() float64 {
		return conv2DFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
}

func TestCUDAConv2DTransposeBackwardMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{2, 4, 4, 5}, patternedFloats(2*4*4*5, 0.018))
	weight := backend.NewTensorF32([]int{4, 3, 3, 2}, patternedFloats(4*3*3*2, -0.026))
	bias := backend.NewTensorF32([]int{6}, patternedFloats(6, 0.013))
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2DTrans, Attributes: map[string]string{
		"groups":           "2",
		"stride_h":         "2",
		"stride_w":         "1",
		"pad_h":            "1",
		"pad_w":            "0",
		"output_padding_h": "1",
	}}
	cfg, ok := planBuiltinConv2DTranspose(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d_transpose should be supported")
	}
	gradOut := backend.NewTensorF32([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, patternedFloats(cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth, 0.024))

	wantIn, wantW, wantB := hostConv2DTransposeBackward(input, weight, bias, gradOut, cfg)
	gotIn, gotW, gotB, err := rt.runConv2DTransposeBackward(input, weight, bias, gradOut, cfg)
	if err != nil {
		t.Fatalf("run conv2d_transpose backward: %v", err)
	}
	assertTensorClose(t, gotIn, input.Shape, wantIn)
	assertTensorClose(t, gotW, weight.Shape, wantW)
	assertTensorClose(t, gotB, bias.Shape, wantB)
}

func TestCUDAConv2DTransposeBackwardMatchesFiniteDifference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{1, 2, 3, 4}, patternedFloats(1*2*3*4, 0.017))
	weight := backend.NewTensorF32([]int{2, 3, 2, 3}, patternedFloats(2*3*2*3, -0.021))
	bias := backend.NewTensorF32([]int{3}, patternedFloats(3, 0.009))
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2DTrans, Attributes: map[string]string{
		"stride_h":         "2",
		"stride_w":         "1",
		"pad_h":            "1",
		"pad_w":            "1",
		"output_padding_h": "1",
	}}
	cfg, ok := planBuiltinConv2DTranspose(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d_transpose should be supported")
	}
	gradOut := backend.NewTensorF32([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, patternedFloats(cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth, 0.029))
	gotIn, gotW, gotB, err := rt.runConv2DTransposeBackward(input, weight, bias, gradOut, cfg)
	if err != nil {
		t.Fatalf("run conv2d_transpose backward: %v", err)
	}

	const eps = 1e-3
	assertFiniteDiffTensor(t, "transpose input", gotIn, input, eps, func() float64 {
		return conv2DTransposeFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
	assertFiniteDiffTensor(t, "transpose weight", gotW, weight, eps, func() float64 {
		return conv2DTransposeFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
	assertFiniteDiffTensor(t, "transpose bias", gotB, bias, eps, func() float64 {
		return conv2DTransposeFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
}

func patternedFloats(count int, scale float32) []float32 {
	values := make([]float32, count)
	for i := range values {
		values[i] = float32((i%23)-11) * scale
	}
	return values
}

func hostConv2DBackward(input, weight, bias, gradOut *backend.Tensor, cfg cudaConv2DConfig) ([]float32, []float32, []float32) {
	gradIn := make([]float32, input.Elements())
	gradW := make([]float32, weight.Elements())
	var gradB []float32
	if bias != nil {
		gradB = make([]float32, bias.Elements())
	}
	for n := 0; n < cfg.batches; n++ {
		for group := 0; group < cfg.groups; group++ {
			for ocg := 0; ocg < cfg.outPerGroup; ocg++ {
				oc := group*cfg.outPerGroup + ocg
				for oy := 0; oy < cfg.outHeight; oy++ {
					for ox := 0; ox < cfg.outWidth; ox++ {
						goVal := gradOut.F32[offset4(gradOut.Shape, n, oc, oy, ox)]
						if gradB != nil {
							gradB[oc] += goVal
						}
						for icg := 0; icg < cfg.inPerGroup; icg++ {
							ic := group*cfg.inPerGroup + icg
							for ky := 0; ky < cfg.kernelH; ky++ {
								iy := oy*cfg.strideH + ky*cfg.dilationH - cfg.padH
								if iy < 0 || iy >= cfg.inHeight {
									continue
								}
								for kx := 0; kx < cfg.kernelW; kx++ {
									ix := ox*cfg.strideW + kx*cfg.dilationW - cfg.padW
									if ix < 0 || ix >= cfg.inWidth {
										continue
									}
									wIdx := ((oc*cfg.inPerGroup+icg)*cfg.kernelH+ky)*cfg.kernelW + kx
									inIdx := offset4(input.Shape, n, ic, iy, ix)
									gradIn[inIdx] += goVal * weight.F32[wIdx]
									gradW[wIdx] += goVal * input.F32[inIdx]
								}
							}
						}
					}
				}
			}
		}
	}
	return gradIn, gradW, gradB
}

func hostConv2DTransposeBackward(input, weight, bias, gradOut *backend.Tensor, cfg cudaConv2DTransposeConfig) ([]float32, []float32, []float32) {
	gradIn := make([]float32, input.Elements())
	gradW := make([]float32, weight.Elements())
	var gradB []float32
	if bias != nil {
		gradB = make([]float32, bias.Elements())
	}
	for n := 0; n < cfg.batches; n++ {
		for group := 0; group < cfg.groups; group++ {
			for icg := 0; icg < cfg.inPerGroup; icg++ {
				ic := group*cfg.inPerGroup + icg
				for iy := 0; iy < cfg.inHeight; iy++ {
					for ix := 0; ix < cfg.inWidth; ix++ {
						inVal := input.F32[offset4(input.Shape, n, ic, iy, ix)]
						for ocg := 0; ocg < cfg.outPerGroup; ocg++ {
							oc := group*cfg.outPerGroup + ocg
							for ky := 0; ky < cfg.kernelH; ky++ {
								oy := iy*cfg.strideH + ky*cfg.dilationH - cfg.padH
								if oy < 0 || oy >= cfg.outHeight {
									continue
								}
								for kx := 0; kx < cfg.kernelW; kx++ {
									ox := ix*cfg.strideW + kx*cfg.dilationW - cfg.padW
									if ox < 0 || ox >= cfg.outWidth {
										continue
									}
									goVal := gradOut.F32[offset4(gradOut.Shape, n, oc, oy, ox)]
									wIdx := ((ic*cfg.outPerGroup+ocg)*cfg.kernelH+ky)*cfg.kernelW + kx
									gradIn[offset4(input.Shape, n, ic, iy, ix)] += goVal * weight.F32[wIdx]
									gradW[wIdx] += goVal * inVal
								}
							}
						}
					}
				}
			}
			if gradB != nil {
				for ocg := 0; ocg < cfg.outPerGroup; ocg++ {
					oc := group*cfg.outPerGroup + ocg
					for y := 0; y < cfg.outHeight; y++ {
						for x := 0; x < cfg.outWidth; x++ {
							gradB[oc] += gradOut.F32[offset4(gradOut.Shape, n, oc, y, x)]
						}
					}
				}
			}
		}
	}
	return gradIn, gradW, gradB
}

func conv2DFiniteDiffLoss(input, weight, bias, gradOut *backend.Tensor, cfg cudaConv2DConfig) float64 {
	out := hostConv2D(input, weight, bias, cfg)
	loss := 0.0
	for i, v := range out {
		loss += float64(v * gradOut.F32[i])
	}
	return loss
}

func conv2DTransposeFiniteDiffLoss(input, weight, bias, gradOut *backend.Tensor, cfg cudaConv2DTransposeConfig) float64 {
	out := hostConv2DTranspose(input, weight, bias, cfg)
	loss := 0.0
	for i, v := range out {
		loss += float64(v * gradOut.F32[i])
	}
	return loss
}

func assertFiniteDiffTensor(t *testing.T, name string, analytic, variable *backend.Tensor, eps float32, loss func() float64) {
	t.Helper()
	for i := range variable.F32 {
		original := variable.F32[i]
		variable.F32[i] = original + eps
		plus := loss()
		variable.F32[i] = original - eps
		minus := loss()
		variable.F32[i] = original
		numeric := (plus - minus) / float64(2*eps)
		got := float64(analytic.F32[i])
		if math.Abs(got-numeric) > 2e-2 {
			t.Fatalf("%s grad[%d] = %.6f, finite diff %.6f", name, i, got, numeric)
		}
	}
}
