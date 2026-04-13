//go:build linux && cgo

package cuda

import (
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
	turboquant "github.com/odvcencio/turboquant"
)

func TestCUDATurboQuantStepsMatchTurboQuantSpec(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{1, 5, 2, 2}, []float32{
		0.10, -0.20, 0.30, -0.40,
		0.50, -0.60, 0.70, -0.80,
		0.90, -1.00, 1.10, -1.20,
		1.30, -1.40, 1.50, -1.60,
		1.70, -1.80, 1.90, -2.00,
	})
	step := mantaartifact.Step{Kind: mantaartifact.StepTurboQEncode, Attributes: map[string]string{"bits": "4", "seed": "77"}}
	cfg, ok := planBuiltinTurboQEncode(step, []*backend.Tensor{input})
	if !ok {
		t.Fatal("turboquant encode should be supported")
	}
	encoded, err := rt.runTurboQEncodeStep([]*backend.Tensor{input}, scalarF32ValueType(), cfg)
	if err != nil {
		t.Fatalf("run turboquant encode: %v", err)
	}
	if encoded.VariantEntry != "__builtin_cuda_turboquant_encode" {
		t.Fatalf("encode variant = %q", encoded.VariantEntry)
	}
	if encoded.Metadata["device_execution"] != true {
		t.Fatalf("encode device_execution = %v, want true", encoded.Metadata["device_execution"])
	}
	wantCoords, wantNorms := hostTurboQEncode(input, 4, 77)
	assertTensorClose(t, encoded.Outputs[0], input.Shape, wantCoords)
	assertTensorClose(t, encoded.Outputs[1], []int{1, 2, 2}, wantNorms)

	decodeType := mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f32"},
	}
	decodeStep := mantaartifact.Step{Kind: mantaartifact.StepTurboQDecode, Attributes: map[string]string{"bits": "4", "seed": "77"}}
	decodeCfg, ok := planBuiltinTurboQDecode(decodeStep, encoded.Outputs)
	if !ok {
		t.Fatal("turboquant decode should be supported")
	}
	decoded, err := rt.runTurboQDecodeStep(encoded.Outputs, decodeType, decodeCfg)
	if err != nil {
		t.Fatalf("run turboquant decode: %v", err)
	}
	if decoded.VariantEntry != "__builtin_cuda_turboquant_decode" {
		t.Fatalf("decode variant = %q", decoded.VariantEntry)
	}
	wantDecoded := hostTurboQDecode(encoded.Outputs[0], encoded.Outputs[1], 4, 77)
	assertTensorClose(t, decoded.Outputs[0], input.Shape, wantDecoded)
}

func hostTurboQEncode(input *backend.Tensor, bits int, seed int64) ([]float32, []float32) {
	q := turboquant.NewHadamardWithSeed(input.Shape[1], bits, seed)
	n, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	coords := make([]float32, len(input.F32))
	norms := make([]float32, n*height*width)
	vec := make([]float32, channels)
	indices := make([]int, channels)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					vec[c] = input.F32[offset4(input.Shape, b, c, y, x)]
				}
				norm := q.QuantizeIndicesTo(indices, vec)
				for c := 0; c < channels; c++ {
					coords[offset4(input.Shape, b, c, y, x)] = float32(indices[c])
				}
				norms[(b*height+y)*width+x] = float32(quantizeQNormForTest(norm))
			}
		}
	}
	return coords, norms
}

func hostTurboQDecode(coords, norms *backend.Tensor, bits int, seed int64) []float32 {
	q := turboquant.NewHadamardWithSeed(coords.Shape[1], bits, seed)
	n, channels, height, width := coords.Shape[0], coords.Shape[1], coords.Shape[2], coords.Shape[3]
	out := make([]float32, len(coords.F32))
	vec := make([]float32, channels)
	indices := make([]int, channels)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					indices[c] = clampIntForTest(int(math.Round(float64(coords.F32[offset4(coords.Shape, b, c, y, x)]))), 0, (1<<bits)-1)
				}
				q.DequantizeIndicesTo(vec, indices)
				norm := dequantizeQNormForTest(byte(clampIntForTest(int(math.Round(float64(norms.F32[(b*height+y)*width+x]))), 0, 255)))
				for c := 0; c < channels; c++ {
					out[offset4(coords.Shape, b, c, y, x)] = vec[c] * norm
				}
			}
		}
	}
	return out
}

func quantizeQNormForTest(norm float32) byte {
	if norm <= 0 || math.IsNaN(float64(norm)) {
		return 0
	}
	if math.IsInf(float64(norm), 1) {
		return 255
	}
	t := (math.Log(float64(norm)) + 16.0) / 32.0
	if t <= 0 {
		return 0
	}
	if t >= 1 {
		return 255
	}
	return byte(math.Round(t * 255))
}

func dequantizeQNormForTest(encoded byte) float32 {
	t := float64(encoded) / 255
	return float32(math.Exp(-16.0 + t*32.0))
}
