package barruntime

import (
	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

func newTrainerOptimizerAccelerator() (backend.OptimizerAccelerator, barr.BackendKind, error) {
	return backend.NewPreferredOptimizerAccelerator(barr.BackendCUDA, barr.BackendMetal)
}
