package barruntime

import (
	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

func newTrainerContrastiveAccelerator() (backend.ContrastiveAccelerator, barr.BackendKind, error) {
	return backend.NewPreferredContrastiveAccelerator(barr.BackendCUDA, barr.BackendMetal)
}
