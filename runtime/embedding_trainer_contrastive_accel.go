package barruntime

import (
	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

func newTrainerContrastiveAccelerator() (backend.ContrastiveAccelerator, barr.BackendKind, error) {
	return backend.NewPreferredContrastiveAccelerator(barr.BackendCUDA, barr.BackendMetal)
}
