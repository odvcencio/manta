package backend

import (
	"fmt"
	"strconv"
	"strings"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

// CompileNativeKernelProgram builds a backend-owned kernel program from a
// compiled variant plus the backend-neutral kernel body. The program still
// executes the supported v0 ops numerically for now, but the launch path is
// owned by the selected backend rather than the generic plan executor.
func CompileNativeKernelProgram(kind mantaartifact.BackendKind, kernel mantaartifact.Kernel, compiled CompiledKernel) (NativeKernelProgram, error) {
	if kernel.Name == "" {
		return NativeKernelProgram{}, fmt.Errorf("kernel name is required")
	}
	if compiled.Entry == "" {
		return NativeKernelProgram{}, fmt.Errorf("kernel %q compiled entry is required", kernel.Name)
	}
	if err := validateCompiledKernelSource(kind, compiled); err != nil {
		return NativeKernelProgram{}, err
	}
	config := nativeLaunchConfig(kind, kernel, compiled)
	fallback := func(inputs []*Tensor) ([]*Tensor, error) {
		return executeKernel(kernel, inputs)
	}
	return NativeKernelProgram{
		Compiled:     compiled,
		LaunchConfig: config,
		Fallback:     fallback,
		Run:          fallback,
	}, nil
}

func validateCompiledKernelSource(kind mantaartifact.BackendKind, compiled CompiledKernel) error {
	switch kind {
	case mantaartifact.BackendCUDA:
		if !strings.Contains(compiled.Source, "__global__ void "+compiled.Entry+"(") {
			return fmt.Errorf("kernel %q CUDA source does not define entry %q", compiled.Name, compiled.Entry)
		}
	case mantaartifact.BackendMetal:
		if !strings.Contains(compiled.Source, "kernel void "+compiled.Entry+"(") {
			return fmt.Errorf("kernel %q Metal source does not define entry %q", compiled.Name, compiled.Entry)
		}
	case mantaartifact.BackendVulkan:
		if !strings.Contains(compiled.Source, "void "+compiled.Entry+"(") {
			return fmt.Errorf("kernel %q Vulkan source does not define entry %q", compiled.Name, compiled.Entry)
		}
	case mantaartifact.BackendDirectML:
		if !strings.Contains(compiled.Source, "manta_directml_graph "+compiled.Entry+"(") {
			return fmt.Errorf("kernel %q DirectML source does not define entry %q", compiled.Name, compiled.Entry)
		}
	case mantaartifact.BackendWebGPU:
		if !strings.Contains(compiled.Source, "fn "+compiled.Entry+"(") {
			return fmt.Errorf("kernel %q WebGPU source does not define entry %q", compiled.Name, compiled.Entry)
		}
	default:
		return fmt.Errorf("unsupported backend %q", kind)
	}
	return nil
}

func nativeLaunchConfig(kind mantaartifact.BackendKind, kernel mantaartifact.Kernel, compiled CompiledKernel) map[string]any {
	config := map[string]any{
		"dispatch_mode":       "backend_native",
		"dispatch_backend":    string(kind),
		"device_execution":    false,
		"launch_entry":        compiled.Entry,
		"launch_tile":         compiled.Meta["tile"],
		"launch_tile_2d":      compiled.Meta["tile_2d"],
		"launch_vector_width": compiled.Meta["vector_width"],
		"launch_memory":       compiled.Meta["memory"],
		"launch_subgroup":     compiled.Meta["subgroup"],
		"launch_subgroup_2d":  compiled.Meta["subgroup_2d"],
		"launch_halo":         compiled.Meta["halo"],
	}
	tile := firstTileSize(kernel, compiled)
	grid := "1d"
	if len(kernel.Hints.Tile2D) > 0 || compiled.Meta["tile_2d"] != "" {
		grid = "2d"
	}
	switch kind {
	case mantaartifact.BackendCUDA:
		config["launch_api"] = "cuLaunchKernel"
		config["launch_grid"] = grid
		config["launch_block_size"] = tile
		config["launch_shared_bytes"] = estimatedSharedBytes(kernel, tile)
	case mantaartifact.BackendMetal:
		config["launch_api"] = "dispatchThreadgroups"
		config["launch_grid"] = grid
		config["launch_threadgroup_size"] = tile
		config["launch_threadgroup_memory_bytes"] = estimatedSharedBytes(kernel, tile)
	case mantaartifact.BackendVulkan:
		config["launch_api"] = "vkCmdDispatch"
		config["launch_grid"] = grid
		config["launch_workgroup_size"] = tile
		config["launch_shared_bytes"] = estimatedSharedBytes(kernel, tile)
	case mantaartifact.BackendDirectML:
		config["launch_api"] = "IDMLCommandRecorder::RecordDispatch"
		config["launch_grid"] = grid
		config["launch_threadgroup_size"] = tile
		config["launch_temporary_resource_bytes"] = estimatedSharedBytes(kernel, tile)
	case mantaartifact.BackendWebGPU:
		config["launch_api"] = "GPUComputePassEncoder.dispatchWorkgroups"
		config["launch_grid"] = grid
		config["launch_workgroup_size"] = tile
		config["launch_workgroup_memory_bytes"] = estimatedSharedBytes(kernel, tile)
	}
	return config
}

func firstTileSize(kernel mantaartifact.Kernel, compiled CompiledKernel) int {
	if len(kernel.Hints.Tile) > 0 && kernel.Hints.Tile[0] > 0 {
		return kernel.Hints.Tile[0]
	}
	if raw := compiled.Meta["tile"]; raw != "" {
		raw = strings.TrimPrefix(raw, "[")
		raw = strings.TrimSuffix(raw, "]")
		if raw != "" {
			parts := strings.Split(raw, ",")
			if n, err := strconv.Atoi(strings.TrimSpace(parts[0])); err == nil && n > 0 {
				return n
			}
		}
	}
	return 1
}

func estimatedSharedBytes(kernel mantaartifact.Kernel, tile int) int {
	if tile <= 0 {
		tile = 1
	}
	if kernel.Hints.Memory != "workgroup_local" {
		return 0
	}
	width := kernel.Hints.VectorWidth
	if width <= 0 {
		width = 1
	}
	valueCount := len(kernel.Inputs) + len(kernel.Outputs)
	if valueCount == 0 {
		valueCount = 1
	}
	return tile * width * valueCount * 4
}
