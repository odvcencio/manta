package mantaruntime

import (
	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func newTrainerOptimizerAccelerator() (backend.OptimizerAccelerator, mantaartifact.BackendKind, error) {
	return backend.NewPreferredOptimizerAccelerator(mantaartifact.BackendCUDA, mantaartifact.BackendMetal)
}
