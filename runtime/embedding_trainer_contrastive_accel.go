package mantaruntime

import (
	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func newTrainerContrastiveAccelerator() (backend.ContrastiveAccelerator, mantaartifact.BackendKind, error) {
	return backend.NewPreferredContrastiveAccelerator(mantaartifact.BackendCUDA, mantaartifact.BackendMetal)
}
