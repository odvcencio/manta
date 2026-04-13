package backend

import (
	"math"
	"testing"
)

func TestConv2DTensorHostReference(t *testing.T) {
	input := NewTensorF16([]int{1, 1, 3, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	weight := NewTensorF16([]int{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})
	out, err := conv2DTensor(input, weight, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := out.Shape, []int{1, 1, 2, 2}; !sameInts(got, want) {
		t.Fatalf("shape: got %v want %v", got, want)
	}
	want := []float32{6, 8, 12, 14}
	for i := range want {
		if out.F32[i] != want[i] {
			t.Fatalf("out[%d]=%v want %v", i, out.F32[i], want[i])
		}
	}
}

func TestConv2DTransposeTensorHostReference(t *testing.T) {
	input := NewTensorF16([]int{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	weight := NewTensorF16([]int{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})
	out, err := conv2DTransposeTensor(input, weight, nil, map[string]string{"stride": "2"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := out.Shape, []int{1, 1, 4, 4}; !sameInts(got, want) {
		t.Fatalf("shape: got %v want %v", got, want)
	}
	if out.F32[0] != 1 || out.F32[5] != 1 || out.F32[10] != 4 || out.F32[15] != 4 {
		t.Fatalf("unexpected transpose conv output: %v", out.F32)
	}
}

func TestGDNAndIGDNShape(t *testing.T) {
	input := NewTensorF16([]int{1, 2, 1, 2}, []float32{1, 2, 3, 4})
	beta := NewTensorF16([]int{2}, []float32{1, 1})
	gamma := NewTensorF16([]int{2, 2}, []float32{
		0.1, 0,
		0, 0.1,
	})
	gdn, err := gdnTensor(input, beta, gamma, false)
	if err != nil {
		t.Fatal(err)
	}
	igdn, err := gdnTensor(gdn, beta, gamma, true)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(igdn.Shape, input.Shape) {
		t.Fatalf("shape: got %v want %v", igdn.Shape, input.Shape)
	}
	for i, v := range igdn.F32 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("invalid igdn value at %d: %v", i, v)
		}
	}
}

func TestTurboQuantEncodeDecodeTensor(t *testing.T) {
	input := NewTensorF16([]int{1, 4, 2, 2}, []float32{
		1, 0, 0, 1,
		0, 1, 1, 0,
		0.5, -0.5, 0.25, -0.25,
		-1, 1, -1, 1,
	})
	coords, norms, err := turboQuantEncodeTensor(input, map[string]string{"bits": "4", "seed": "7"})
	if err != nil {
		t.Fatal(err)
	}
	if coords.DType != "q4" || norms.DType != "q_norm" {
		t.Fatalf("dtypes: %s %s", coords.DType, norms.DType)
	}
	decoded, err := turboQuantDecodeTensor(coords, norms, map[string]string{"bits": "4", "seed": "7"})
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(decoded.Shape, input.Shape) {
		t.Fatalf("shape: got %v want %v", decoded.Shape, input.Shape)
	}
	for i, v := range decoded.F32 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("invalid decoded value at %d: %v", i, v)
		}
	}
}

func TestLosses(t *testing.T) {
	a := NewTensorF16([]int{1, 1, 1, 4}, []float32{1, 2, 3, 4})
	b := NewTensorF16([]int{1, 1, 1, 4}, []float32{1, 2, 3, 5})
	mse, err := mseLossTensor(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if mse.F32[0] <= 0 {
		t.Fatalf("mse should be positive: %v", mse.F32[0])
	}
	ssim, err := msSSIMLossTensor(a, a)
	if err != nil {
		t.Fatal(err)
	}
	if ssim.F32[0] != 0 {
		t.Fatalf("identical tensors should have zero ms-ssim loss, got %v", ssim.F32[0])
	}
	ce, err := crossEntropyFactorizedTensor(NewTensorQ4([]int{2}, []float32{0, 1}), nil, map[string]string{"bits": "4"})
	if err != nil {
		t.Fatal(err)
	}
	if ce.F32[0] != 8 {
		t.Fatalf("uniform q4 two-symbol CE: got %v want 8", ce.F32[0])
	}
	codes := NewTensorQ2([]int{1, 2, 1, 1}, []float32{1, 2})
	logits := NewTensorF16([]int{1, 2 * 4, 1, 1}, []float32{
		0, 4, 0, 0,
		0, 0, 4, 0,
	})
	catCE, err := crossEntropyFactorizedTensor(codes, logits, map[string]string{
		"bits":          "2",
		"logits_layout": "nchw_alphabet",
	})
	if err != nil {
		t.Fatal(err)
	}
	if catCE.F32[0] >= 1 {
		t.Fatalf("peaked categorical CE should be low, got %v", catCE.F32[0])
	}
	bitLogits := NewTensorF16([]int{1, 2 * 2 * 2, 1, 1}, []float32{
		4, 0, 0, 4,
		0, 4, 4, 0,
	})
	bitCE, err := crossEntropyFactorizedTensor(codes, bitLogits, map[string]string{
		"bits":          "2",
		"factorization": "bit-plane",
	})
	if err != nil {
		t.Fatal(err)
	}
	if bitCE.F32[0] >= 1 {
		t.Fatalf("peaked bit-plane CE should be low, got %v", bitCE.F32[0])
	}
	norms := NewTensorQNorm([]int{1, 1, 1}, []float32{float32(quantizeQNorm(1))})
	normParams := NewTensorF16([]int{1, 2, 1, 1}, []float32{0, 0.5})
	normCE, err := crossEntropyFactorizedTensor(norms, normParams, map[string]string{"distribution": "log_normal"})
	if err != nil {
		t.Fatal(err)
	}
	if normCE.F32[0] <= 0 {
		t.Fatalf("norm CE should be positive, got %v", normCE.F32[0])
	}
	sum, err := scalarAddTensor(ce, normCE)
	if err != nil {
		t.Fatal(err)
	}
	rd, err := rateDistortionLossTensor(mse, sum, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	if rd.F32[0] <= mse.F32[0] {
		t.Fatalf("RD loss should include rate term: mse=%v rd=%v", mse.F32[0], rd.F32[0])
	}
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
