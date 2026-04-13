package backend

import (
	"context"
	"fmt"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

// ExecuteSymbolic runs the current Manta runtime path. The selected
// backend resolves compiled variants at load time, and plan steps execute
// through backend-owned dispatch where promoted kernels exist and through the
// host/reference path where they do not yet.
func ExecuteSymbolic(ctx context.Context, mod *mantaartifact.Module, weights map[string]WeightBinding, compiled map[string]CompiledKernel, dispatch KernelDispatcher, dispatchStep StepDispatcher, kind mantaartifact.BackendKind, req Request) (Result, error) {
	if mod == nil {
		return Result{}, fmt.Errorf("nil module")
	}
	entry, ok := entryPointByName(mod, req.Entry)
	if !ok {
		return Result{}, fmt.Errorf("unknown entrypoint %q", req.Entry)
	}
	if err := validateRequestInputs(entry, req.Inputs); err != nil {
		return Result{}, err
	}

	bindings := map[string]int{}
	env := map[string]Value{}
	paramUses := countEntryParamUses(stepsForEntry(mod, entry.Name), mod.Params)
	unusedEntryParams := countUnusedEntryParams(paramUses)
	eagerMaterialized := 0
	lazyMaterialized := 0
	releasedParams := 0
	for _, param := range mod.Params {
		weight, ok := weights[param.Name]
		if !ok {
			return Result{}, fmt.Errorf("missing weight binding for param %q", param.Name)
		}
		if !shouldEagerMaterializeParam(weight) {
			continue
		}
		data, concreteType, err := PreviewValueWithBindings(param.Type, weight.Data, bindings)
		if err != nil {
			return Result{}, fmt.Errorf("param %q: %w", param.Name, err)
		}
		env[param.Name] = Value{
			Type:     concreteType,
			Data:     data,
			Producer: "param:" + param.Name,
			Metadata: map[string]any{
				"materialized":        "eager",
				"weight_residency":    weight.Residency,
				"weight_access_count": weight.AccessCount,
			},
		}
		eagerMaterialized++
	}
	for _, input := range entry.Inputs {
		data, concreteType, err := MaterializeValueWithBindings(input.Type, req.Inputs[input.Name], bindings)
		if err != nil {
			return Result{}, fmt.Errorf("input %q: %w", input.Name, err)
		}
		env[input.Name] = Value{
			Type:     concreteType,
			Data:     data,
			Producer: "input:" + input.Name,
		}
	}

	steps := stepsForEntry(mod, entry.Name)
	if len(steps) == 0 {
		return Result{}, fmt.Errorf("entrypoint %q has no plan steps", entry.Name)
	}

	trace := make([]TraceStep, 0, len(steps))
	result := Result{
		Outputs: map[string]Value{},
		Metadata: map[string]string{
			"backend":                   string(kind),
			"status":                    "hybrid",
			"kernel_dispatch":           "backend_native",
			"weight_bindings":           fmt.Sprintf("%d", len(weights)),
			"compiled_kernels":          fmt.Sprintf("%d", len(compiled)),
			"entrypoint":                entry.Name,
			"step_count":                fmt.Sprintf("%d", len(steps)),
			"params_total":              fmt.Sprintf("%d", len(mod.Params)),
			"params_eager_materialized": fmt.Sprintf("%d", eagerMaterialized),
		},
	}

	for _, step := range steps {
		for _, input := range step.Inputs {
			if _, ok := env[input]; !ok {
				if isParamName(mod.Params, input) {
					value, concreteType, materialized, err := materializeParamValue(input, mod.Params, weights, bindings, env)
					if err != nil {
						return Result{}, fmt.Errorf("param %q: %w", input, err)
					}
					if materialized {
						_ = concreteType
						env[input] = value
						lazyMaterialized++
					}
				}
			}
			if _, ok := env[input]; !ok {
				return Result{}, fmt.Errorf("entrypoint %q step %q missing runtime input %q", entry.Name, step.Name, input)
			}
		}
		values, variantEntry, err := executeStep(ctx, mod, entry, step, env, compiled, dispatch, dispatchStep, bindings, kind)
		if err != nil {
			return Result{}, fmt.Errorf("entrypoint %q step %q: %w", entry.Name, step.Name, err)
		}
		trace = append(trace, TraceStep{
			Entry:   step.Entry,
			Kind:    step.Kind,
			Name:    step.Name,
			Kernel:  step.Kernel,
			Variant: variantEntry,
			Inputs:  cloneStrings(step.Inputs),
			Outputs: cloneStrings(step.Outputs),
		})
		switch step.Kind {
		case mantaartifact.StepReturn:
			for _, name := range step.Outputs {
				value, ok := env[name]
				if !ok {
					return Result{}, fmt.Errorf("entrypoint %q return references unknown value %q", entry.Name, name)
				}
				result.Outputs[name] = value
			}
		default:
			for i, name := range step.Outputs {
				if i >= len(values) {
					return Result{}, fmt.Errorf("step produced %d values for %d outputs", len(values), len(step.Outputs))
				}
				env[name] = values[i]
			}
		}
		for _, input := range step.Inputs {
			if !isParamName(mod.Params, input) {
				continue
			}
			if paramUses[input] > 0 {
				paramUses[input]--
			}
			if paramUses[input] == 0 && shouldReleaseMaterializedParam(weights[input]) {
				if _, ok := env[input]; ok {
					delete(env, input)
					releasedParams++
				}
			}
		}
	}
	if len(result.Outputs) == 0 {
		return Result{}, fmt.Errorf("entrypoint %q produced no outputs", entry.Name)
	}
	result.Metadata["params_lazy_materialized"] = fmt.Sprintf("%d", lazyMaterialized)
	result.Metadata["params_released"] = fmt.Sprintf("%d", releasedParams)
	result.Metadata["params_unused_for_entry"] = fmt.Sprintf("%d", unusedEntryParams)
	result.Metadata["param_materialization"] = paramMaterializationMode(eagerMaterialized, lazyMaterialized)
	result.Trace = trace
	return result, nil
}

func shouldEagerMaterializeParam(weight WeightBinding) bool {
	return weight.Residency == "" || weight.Residency == "device_resident"
}

func shouldReleaseMaterializedParam(weight WeightBinding) bool {
	return weight.Residency == "lazy_staged"
}

func materializeParamValue(name string, params []mantaartifact.Param, weights map[string]WeightBinding, bindings map[string]int, env map[string]Value) (Value, mantaartifact.ValueType, bool, error) {
	param, ok := paramByName(params, name)
	if !ok {
		return Value{}, mantaartifact.ValueType{}, false, fmt.Errorf("unknown param %q", name)
	}
	if value, ok := env[name]; ok {
		return value, value.Type, false, nil
	}
	weight, ok := weights[name]
	if !ok {
		return Value{}, mantaartifact.ValueType{}, false, fmt.Errorf("missing weight binding")
	}
	data, concreteType, err := PreviewValueWithBindings(param.Type, weight.Data, bindings)
	if err != nil {
		return Value{}, mantaartifact.ValueType{}, false, err
	}
	return Value{
		Type:     concreteType,
		Data:     data,
		Producer: "param:" + name,
		Metadata: map[string]any{
			"materialized":        "lazy",
			"weight_residency":    weight.Residency,
			"weight_access_count": weight.AccessCount,
		},
	}, concreteType, true, nil
}

func countEntryParamUses(steps []mantaartifact.Step, params []mantaartifact.Param) map[string]int {
	counts := map[string]int{}
	if len(steps) == 0 || len(params) == 0 {
		return counts
	}
	paramNames := map[string]bool{}
	for _, param := range params {
		paramNames[param.Name] = true
		counts[param.Name] = 0
	}
	for _, step := range steps {
		for _, input := range step.Inputs {
			if paramNames[input] {
				counts[input]++
			}
		}
	}
	return counts
}

func countUnusedEntryParams(uses map[string]int) int {
	unused := 0
	for _, count := range uses {
		if count == 0 {
			unused++
		}
	}
	return unused
}

func isParamName(params []mantaartifact.Param, name string) bool {
	_, ok := paramByName(params, name)
	return ok
}

func paramByName(params []mantaartifact.Param, name string) (mantaartifact.Param, bool) {
	for _, param := range params {
		if param.Name == name {
			return param, true
		}
	}
	return mantaartifact.Param{}, false
}

func paramMaterializationMode(eager, lazy int) string {
	switch {
	case lazy == 0:
		return "eager_all"
	case eager == 0:
		return "lazy_on_demand"
	default:
		return "mixed"
	}
}

func entryPointByName(mod *mantaartifact.Module, name string) (mantaartifact.EntryPoint, bool) {
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, true
		}
	}
	return mantaartifact.EntryPoint{}, false
}

func validateRequestInputs(entry mantaartifact.EntryPoint, inputs map[string]any) error {
	for _, input := range entry.Inputs {
		if _, ok := inputs[input.Name]; !ok {
			return fmt.Errorf("entrypoint %q missing input %q", entry.Name, input.Name)
		}
	}
	for name := range inputs {
		if !entryHasInput(entry, name) {
			return fmt.Errorf("entrypoint %q does not declare input %q", entry.Name, name)
		}
	}
	return nil
}

func entryHasInput(entry mantaartifact.EntryPoint, name string) bool {
	for _, input := range entry.Inputs {
		if input.Name == name {
			return true
		}
	}
	return false
}

func stepsForEntry(mod *mantaartifact.Module, entry string) []mantaartifact.Step {
	out := make([]mantaartifact.Step, 0, len(mod.Steps))
	for _, step := range mod.Steps {
		if step.Entry == entry {
			out = append(out, step)
		}
	}
	return out
}

func resolveStepOutputType(mod *mantaartifact.Module, entry mantaartifact.EntryPoint, step mantaartifact.Step, outputIndex int, env map[string]Value) mantaartifact.ValueType {
	if outputIndex < len(step.Outputs) {
		name := step.Outputs[outputIndex]
		if binding, ok := entryOutputByName(entry, name); ok {
			return binding.Type
		}
		if buf, ok := bufferByName(mod, name); ok {
			return valueTypeForBuffer(buf)
		}
	}
	if step.Kind == mantaartifact.StepLaunchKernel {
		if kernel, ok := kernelByName(mod, step.Kernel); ok && outputIndex < len(kernel.Outputs) {
			return kernel.Outputs[outputIndex].Type
		}
	}
	if len(step.Inputs) > 0 {
		if value, ok := env[step.Inputs[0]]; ok {
			return value.Type
		}
	}
	return mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f32"}}
}

func entryOutputByName(entry mantaartifact.EntryPoint, name string) (mantaartifact.ValueBinding, bool) {
	for _, output := range entry.Outputs {
		if output.Name == name {
			return output, true
		}
	}
	return mantaartifact.ValueBinding{}, false
}

func bufferByName(mod *mantaartifact.Module, name string) (mantaartifact.Buffer, bool) {
	for _, buf := range mod.Buffers {
		if buf.Name == name {
			return buf, true
		}
	}
	return mantaartifact.Buffer{}, false
}

func kernelVariantForBackend(kernel mantaartifact.Kernel, kind mantaartifact.BackendKind) (mantaartifact.KernelVariant, bool) {
	for _, variant := range kernel.Variants {
		if variant.Backend == kind {
			return variant, true
		}
	}
	return mantaartifact.KernelVariant{}, false
}

func valueTypeForBuffer(buf mantaartifact.Buffer) mantaartifact.ValueType {
	if buf.DType == "kv_cache" {
		return mantaartifact.ValueType{Kind: mantaartifact.ValueKVCache}
	}
	if buf.DType == "candidate_pack" {
		return mantaartifact.ValueType{
			Kind:          mantaartifact.ValueCandidatePack,
			CandidatePack: &mantaartifact.CandidatePackType{Shape: cloneStrings(buf.Shape)},
		}
	}
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: buf.DType, Shape: cloneStrings(buf.Shape)},
	}
}

func producerName(step mantaartifact.Step) string {
	if step.Kernel != "" {
		return "kernel:" + step.Kernel
	}
	if step.Name != "" {
		return string(step.Kind) + ":" + step.Name
	}
	return string(step.Kind)
}

func cloneStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, len(in))
	copy(out, in)
	return out
}
