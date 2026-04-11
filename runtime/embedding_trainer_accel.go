package barruntime

import (
	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

func newTrainerMatMulAccelerator() (backend.MatMulAccelerator, barr.BackendKind, error) {
	return backend.NewPreferredMatMulAccelerator(barr.BackendCUDA, barr.BackendMetal)
}
