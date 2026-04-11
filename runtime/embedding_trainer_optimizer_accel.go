package barruntime

import (
	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

func newTrainerOptimizerAccelerator() (backend.OptimizerAccelerator, barr.BackendKind, error) {
	return backend.NewPreferredOptimizerAccelerator(barr.BackendCUDA, barr.BackendMetal)
}
