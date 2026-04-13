//go:build linux && cgo

package cuda

import (
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDAConvImageStepsMatchHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	outputType := mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f32"},
	}
	input := backend.NewTensorF32([]int{1, 2, 4, 4}, sequentialFloats(32, 0.1))
	weight := backend.NewTensorF32([]int{3, 2, 3, 3}, sequentialFloats(54, -0.03))
	bias := backend.NewTensorF32([]int{3}, []float32{0.25, -0.5, 0.75})
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: map[string]string{"stride": "2", "padding": "1"}}
	cfg, ok := planBuiltinConv2D(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d should be supported")
	}
	result, err := rt.runConv2DStep([]*backend.Tensor{input, weight, bias}, outputType, cfg)
	if err != nil {
		t.Fatalf("run conv2d: %v", err)
	}
	if result.VariantEntry != "__builtin_cuda_conv2d" {
		t.Fatalf("conv2d variant = %q", result.VariantEntry)
	}
	if result.Metadata["device_execution"] != true {
		t.Fatalf("conv2d device_execution = %v, want true", result.Metadata["device_execution"])
	}
	assertTensorClose(t, result.Outputs[0], []int{1, 3, 2, 2}, hostConv2D(input, weight, bias, cfg))

	transInput := backend.NewTensorF32([]int{1, 2, 2, 2}, []float32{
		1, -1,
		2, -2,
		0.5, -0.5,
		1.5, -1.5,
	})
	transWeight := backend.NewTensorF32([]int{2, 3, 3, 3}, sequentialFloats(54, 0.04))
	transBias := backend.NewTensorF32([]int{3}, []float32{0.1, -0.2, 0.3})
	transStep := mantaartifact.Step{Kind: mantaartifact.StepConv2DTrans, Attributes: map[string]string{"stride": "2", "padding": "1", "output_padding": "1"}}
	transCfg, ok := planBuiltinConv2DTranspose(transStep, []*backend.Tensor{transInput, transWeight, transBias})
	if !ok {
		t.Fatal("conv2d_transpose should be supported")
	}
	transResult, err := rt.runConv2DTransposeStep([]*backend.Tensor{transInput, transWeight, transBias}, outputType, transCfg)
	if err != nil {
		t.Fatalf("run conv2d_transpose: %v", err)
	}
	if transResult.VariantEntry != "__builtin_cuda_conv2d_transpose" {
		t.Fatalf("conv2d_transpose variant = %q", transResult.VariantEntry)
	}
	assertTensorClose(t, transResult.Outputs[0], []int{1, 3, 4, 4}, hostConv2DTranspose(transInput, transWeight, transBias, transCfg))
}

func sequentialFloats(count int, scale float32) []float32 {
	values := make([]float32, count)
	for i := range values {
		values[i] = float32(i+1) * scale
	}
	return values
}

func hostConv2D(input, weight, bias *backend.Tensor, cfg cudaConv2DConfig) []float32 {
	out := make([]float32, cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth)
	for n := 0; n < cfg.batches; n++ {
		for oc := 0; oc < cfg.outChannels; oc++ {
			group := oc / cfg.outPerGroup
			for oy := 0; oy < cfg.outHeight; oy++ {
				for ox := 0; ox < cfg.outWidth; ox++ {
					sum := float32(0)
					if cfg.hasBias {
						sum = bias.F32[oc]
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
								sum += input.F32[offset4(input.Shape, n, ic, iy, ix)] * weight.F32[((oc*cfg.inPerGroup+icg)*cfg.kernelH+ky)*cfg.kernelW+kx]
							}
						}
					}
					out[offset4([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, n, oc, oy, ox)] = sum
				}
			}
		}
	}
	return out
}

func hostConv2DTranspose(input, weight, bias *backend.Tensor, cfg cudaConv2DTransposeConfig) []float32 {
	out := make([]float32, cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth)
	outShape := []int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}
	for n := 0; n < cfg.batches; n++ {
		for oc := 0; oc < cfg.outChannels; oc++ {
			for oy := 0; oy < cfg.outHeight; oy++ {
				for ox := 0; ox < cfg.outWidth; ox++ {
					if cfg.hasBias {
						out[offset4(outShape, n, oc, oy, ox)] = bias.F32[oc]
					}
				}
			}
		}
	}
	for n := 0; n < cfg.batches; n++ {
		for group := 0; group < cfg.groups; group++ {
			for icg := 0; icg < cfg.inPerGroup; icg++ {
				ic := group*cfg.inPerGroup + icg
				for iy := 0; iy < cfg.inHeight; iy++ {
					for ix := 0; ix < cfg.inWidth; ix++ {
						value := input.F32[offset4(input.Shape, n, ic, iy, ix)]
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
									out[offset4(outShape, n, oc, oy, ox)] += value * weight.F32[((ic*cfg.outPerGroup+ocg)*cfg.kernelH+ky)*cfg.kernelW+kx]
								}
							}
						}
					}
				}
			}
		}
	}
	return out
}
