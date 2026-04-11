package barruntime

import (
	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

func newTrainerMatMulAccelerator() (backend.MatMulAccelerator, barr.BackendKind, error) {
	return backend.NewPreferredMatMulAccelerator(barr.BackendCUDA, barr.BackendMetal)
}
