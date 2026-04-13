package mantaruntime

import (
	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func newTrainerMatMulAccelerator() (backend.MatMulAccelerator, mantaartifact.BackendKind, error) {
	return backend.NewPreferredMatMulAccelerator(mantaartifact.BackendCUDA, mantaartifact.BackendMetal)
}
