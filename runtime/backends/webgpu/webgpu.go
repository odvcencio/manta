package webgpu

import (
	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backends/internal/fallback"
)

// New returns the WebGPU backend surface. Device execution is added behind the
// same Backend contract; until then this backend executes through host fallback.
func New() *fallback.Backend {
	return fallback.New(mantaartifact.BackendWebGPU, "WebGPU")
}
