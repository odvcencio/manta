package barruntime

import (
	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

type trainerActivationAccelMode struct {
	fullBackward    bool
	softmaxBackward bool
}

func trainerActivationAccelModeFromEnv() trainerActivationAccelMode {
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_ACTIVATION_ACCEL") {
		return trainerActivationAccelMode{}
	}
	full := trainEnvFlagEnabled("MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL")
	softmax := full || trainEnvFlagEnabled("MANTA_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL")
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL") {
		softmax = false
	}
	return trainerActivationAccelMode{
		fullBackward:    full,
		softmaxBackward: softmax,
	}
}

func newTrainerActivationAccelerator() (backend.ActivationAccelerator, barr.BackendKind, trainerActivationAccelMode, error) {
	mode := trainerActivationAccelModeFromEnv()
	if !mode.fullBackward && !mode.softmaxBackward {
		return nil, "", mode, nil
	}
	accel, backendKind, err := backend.NewPreferredActivationAccelerator(barr.BackendCUDA, barr.BackendMetal)
	return accel, backendKind, mode, err
}

func (t *EmbeddingTrainer) fullActivationBackwardAccelEnabled() bool {
	return t != nil && t.activationAccel != nil && t.activationAccelFull
}

func (t *EmbeddingTrainer) softmaxBackwardAccelEnabled() bool {
	return t != nil && t.activationAccel != nil && t.softmaxBackwardAccel
}
