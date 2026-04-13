//go:build linux && cgo

package cuda

import (
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDAGDNStepMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{1, 2, 1, 2}, []float32{
		1, 2,
		3, 4,
	})
	beta := backend.NewTensorF32([]int{2}, []float32{1, 1})
	gamma := backend.NewTensorF32([]int{2, 2}, []float32{
		0.1, 0.2,
		0.3, 0.4,
	})
	outputType := mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f32"},
	}

	result, err := rt.runGDNStep([]*backend.Tensor{input, beta, gamma}, outputType, false)
	if err != nil {
		t.Fatalf("run gdn: %v", err)
	}
	if result.VariantEntry != "__builtin_cuda_gdn" {
		t.Fatalf("variant = %q, want __builtin_cuda_gdn", result.VariantEntry)
	}
	if result.Metadata["device_execution"] != true {
		t.Fatalf("device_execution = %v, want true", result.Metadata["device_execution"])
	}
	assertTensorClose(t, result.Outputs[0], input.Shape, hostGDN(input, beta, gamma, false))

	inverse, err := rt.runGDNStep([]*backend.Tensor{input, beta, gamma}, outputType, true)
	if err != nil {
		t.Fatalf("run igdn: %v", err)
	}
	if inverse.VariantEntry != "__builtin_cuda_igdn" {
		t.Fatalf("variant = %q, want __builtin_cuda_igdn", inverse.VariantEntry)
	}
	assertTensorClose(t, inverse.Outputs[0], input.Shape, hostGDN(input, beta, gamma, true))
}

func hostGDN(input, beta, gamma *backend.Tensor, inverse bool) []float32 {
	out := make([]float32, len(input.F32))
	channels, height, width := input.Shape[1], input.Shape[2], input.Shape[3]
	spatial := height * width
	for idx, value := range input.F32 {
		c := (idx / spatial) % channels
		n := idx / (channels * spatial)
		hw := idx % spatial
		sum := beta.F32[c]
		for j := 0; j < channels; j++ {
			inputIdx := ((n*channels + j) * spatial) + hw
			channelValue := input.F32[inputIdx]
			sum += gamma.F32[c*channels+j] * channelValue * channelValue
		}
		if sum < 1.0e-12 {
			sum = 1.0e-12
		}
		scale := float32(math.Sqrt(float64(sum)))
		if inverse {
			out[idx] = value * scale
		} else {
			out[idx] = value / scale
		}
	}
	return out
}
