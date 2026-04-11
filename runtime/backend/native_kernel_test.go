package backend

import (
	"testing"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/compiler"
)

func TestCompileNativeKernelProgramCUDAConfig(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	compiled, err := CompileVariants(bundle.Artifact, barr.BackendCUDA)
	if err != nil {
		t.Fatalf("compile variants: %v", err)
	}
	kernel := bundle.Artifact.Kernels[0]
	prog, err := CompileNativeKernelProgram(barr.BackendCUDA, kernel, compiled[kernel.Name])
	if err != nil {
		t.Fatalf("compile native kernel: %v", err)
	}
	if got := prog.LaunchConfig["launch_api"]; got != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", got)
	}
	if got := prog.LaunchConfig["launch_block_size"]; got != 128 {
		t.Fatalf("launch_block_size = %v, want 128", got)
	}
	if got := prog.LaunchConfig["dispatch_mode"]; got != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", got)
	}
}

func TestCompileNativeKernelProgramMetalConfig(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	compiled, err := CompileVariants(bundle.Artifact, barr.BackendMetal)
	if err != nil {
		t.Fatalf("compile variants: %v", err)
	}
	kernel := bundle.Artifact.Kernels[0]
	prog, err := CompileNativeKernelProgram(barr.BackendMetal, kernel, compiled[kernel.Name])
	if err != nil {
		t.Fatalf("compile native kernel: %v", err)
	}
	if got := prog.LaunchConfig["launch_api"]; got != "dispatchThreadgroups" {
		t.Fatalf("launch_api = %v, want dispatchThreadgroups", got)
	}
	if got := prog.LaunchConfig["launch_threadgroup_size"]; got != 128 {
		t.Fatalf("launch_threadgroup_size = %v, want 128", got)
	}
}

func TestCompileNativeKernelProgramRejectsBackendSourceMismatch(t *testing.T) {
	kernel := barr.Kernel{Name: "bad"}
	compiled := CompiledKernel{
		Name:    "bad",
		Backend: barr.BackendCUDA,
		Entry:   "bad_cuda",
		Source:  "kernel void bad_cuda() {}",
	}
	_, err := CompileNativeKernelProgram(barr.BackendCUDA, kernel, compiled)
	if err == nil {
		t.Fatal("expected backend source mismatch")
	}
}
