package cuda

import (
	"testing"

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
