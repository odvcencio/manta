package barruntime

import (
	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

func newTrainerActivationAccelerator() (backend.ActivationAccelerator, barr.BackendKind, error) {
	if trainEnvFlagEnabled("BARR_TRAIN_DISABLE_ACTIVATION_ACCEL") || !trainEnvFlagEnabled("BARR_TRAIN_ENABLE_ACTIVATION_ACCEL") {
		return nil, "", nil
	}
	return backend.NewPreferredActivationAccelerator(barr.BackendCUDA, barr.BackendMetal)
}
