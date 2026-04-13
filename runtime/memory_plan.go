package mantaruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
	mll "github.com/odvcencio/mll"
)

const MemoryPlanVersion = "manta/memory/v0alpha1"

var tagXMEM = [4]byte{'X', 'M', 'E', 'M'}

type MemoryResidency string

const (
	ResidencyDeviceResident MemoryResidency = "device_resident"
	ResidencyHostPinned     MemoryResidency = "host_pinned"
	ResidencyHostShared     MemoryResidency = "host_shared"
	ResidencyLazyStaged     MemoryResidency = "lazy_staged"
)

type MemoryPlanOptions struct {
	Backend           mantaartifact.BackendKind `json:"backend,omitempty"`
	DeviceBudgetBytes int64                     `json:"device_budget_bytes,omitempty"`
	SharedHostWeights bool                      `json:"shared_host_weights,omitempty"`
}

type WeightMemoryPlan struct {
	Name        string          `json:"name"`
	DType       string          `json:"dtype"`
	Shape       []int           `json:"shape,omitempty"`
	Trainable   bool            `json:"trainable,omitempty"`
	Bytes       int64           `json:"bytes"`
	AccessCount int             `json:"access_count,omitempty"`
	Residency   MemoryResidency `json:"residency"`
}

type MemoryPlan struct {
	Version             string                    `json:"version"`
	ModuleName          string                    `json:"module_name"`
	Backend             mantaartifact.BackendKind `json:"backend,omitempty"`
	DeviceBudgetBytes   int64                     `json:"device_budget_bytes,omitempty"`
	TotalWeightBytes    int64                     `json:"total_weight_bytes"`
	DeviceResidentBytes int64                     `json:"device_resident_bytes,omitempty"`
	HostResidentBytes   int64                     `json:"host_resident_bytes,omitempty"`
	LazyStagedBytes     int64                     `json:"lazy_staged_bytes,omitempty"`
	Weights             []WeightMemoryPlan        `json:"weights"`
}

func DefaultMemoryPlanPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".memory.mll")
}

func NewMemoryPlan(mod *mantaartifact.Module, weights map[string]*backend.Tensor, opts MemoryPlanOptions) MemoryPlan {
	plan := MemoryPlan{
		Version:           MemoryPlanVersion,
		Backend:           opts.Backend,
		DeviceBudgetBytes: opts.DeviceBudgetBytes,
	}
	if mod != nil {
		plan.ModuleName = mod.Name
	}
	if len(weights) == 0 || mod == nil {
		return plan
	}

	accessCounts := countParamAccesses(mod)
	params := map[string]mantaartifact.Param{}
	for _, param := range mod.Params {
		params[param.Name] = param
	}

	weightsOrdered := make([]WeightMemoryPlan, 0, len(weights))
	for name, tensor := range weights {
		if tensor == nil {
			continue
		}
		param := params[name]
		item := WeightMemoryPlan{
			Name:        name,
			DType:       tensor.DType,
			Shape:       append([]int(nil), tensor.Shape...),
			Trainable:   param.Trainable,
			Bytes:       tensorLogicalBytes(tensor),
			AccessCount: accessCounts[name],
			Residency:   ResidencyDeviceResident,
		}
		weightsOrdered = append(weightsOrdered, item)
		plan.TotalWeightBytes += item.Bytes
	}

	sort.Slice(weightsOrdered, func(i, j int) bool {
		if weightsOrdered[i].Trainable != weightsOrdered[j].Trainable {
			return weightsOrdered[i].Trainable
		}
		if weightsOrdered[i].AccessCount != weightsOrdered[j].AccessCount {
			return weightsOrdered[i].AccessCount > weightsOrdered[j].AccessCount
		}
		if weightsOrdered[i].Bytes != weightsOrdered[j].Bytes {
			return weightsOrdered[i].Bytes < weightsOrdered[j].Bytes
		}
		return weightsOrdered[i].Name < weightsOrdered[j].Name
	})

	residentUsed := int64(0)
	budgeted := opts.DeviceBudgetBytes > 0
	for i := range weightsOrdered {
		item := &weightsOrdered[i]
		fits := !budgeted || residentUsed+item.Bytes <= opts.DeviceBudgetBytes
		if item.Trainable || fits {
			item.Residency = ResidencyDeviceResident
			residentUsed += item.Bytes
			plan.DeviceResidentBytes += item.Bytes
			continue
		}
		if opts.SharedHostWeights {
			item.Residency = ResidencyHostShared
			plan.HostResidentBytes += item.Bytes
			continue
		}
		if item.AccessCount <= 1 {
			item.Residency = ResidencyLazyStaged
			plan.LazyStagedBytes += item.Bytes
			continue
		}
		item.Residency = ResidencyHostPinned
		plan.HostResidentBytes += item.Bytes
	}

	sort.Slice(weightsOrdered, func(i, j int) bool { return weightsOrdered[i].Name < weightsOrdered[j].Name })
	plan.Weights = weightsOrdered
	return plan
}

func (p MemoryPlan) Validate() error {
	if p.Version == "" {
		return fmt.Errorf("memory plan version is required")
	}
	if p.Version != MemoryPlanVersion {
		return fmt.Errorf("memory plan version %q is not supported, want %q", p.Version, MemoryPlanVersion)
	}
	if p.ModuleName == "" {
		return fmt.Errorf("memory plan module_name is required")
	}
	if len(p.Weights) == 0 {
		return fmt.Errorf("memory plan has no weights")
	}
	seen := map[string]bool{}
	var total, device, host, lazy int64
	for _, item := range p.Weights {
		if item.Name == "" {
			return fmt.Errorf("memory plan weight name is required")
		}
		if seen[item.Name] {
			return fmt.Errorf("duplicate memory plan weight %q", item.Name)
		}
		seen[item.Name] = true
		if item.DType == "" {
			return fmt.Errorf("memory plan weight %q dtype is required", item.Name)
		}
		if item.Bytes <= 0 {
			return fmt.Errorf("memory plan weight %q bytes must be positive", item.Name)
		}
		switch item.Residency {
		case ResidencyDeviceResident:
			device += item.Bytes
		case ResidencyHostPinned, ResidencyHostShared:
			host += item.Bytes
		case ResidencyLazyStaged:
			lazy += item.Bytes
		default:
			return fmt.Errorf("memory plan weight %q has unsupported residency %q", item.Name, item.Residency)
		}
		total += item.Bytes
	}
	if p.TotalWeightBytes != total {
		return fmt.Errorf("memory plan total bytes = %d, want %d", p.TotalWeightBytes, total)
	}
	if p.DeviceResidentBytes != device {
		return fmt.Errorf("memory plan device resident bytes = %d, want %d", p.DeviceResidentBytes, device)
	}
	if p.HostResidentBytes != host {
		return fmt.Errorf("memory plan host resident bytes = %d, want %d", p.HostResidentBytes, host)
	}
	if p.LazyStagedBytes != lazy {
		return fmt.Errorf("memory plan lazy staged bytes = %d, want %d", p.LazyStagedBytes, lazy)
	}
	return nil
}

func (p MemoryPlan) WriteFile(path string) error {
	if err := p.Validate(); err != nil {
		return err
	}
	data, err := encodeMemoryPlanMLL(p)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func ReadMemoryPlanFile(path string) (MemoryPlan, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return MemoryPlan{}, err
	}
	if !mantaartifact.IsMLLBytes(data) {
		return MemoryPlan{}, fmt.Errorf("memory plan %q is not an MLL file", path)
	}
	return decodeMemoryPlanMLL(data)
}

func encodeMemoryPlanMLL(plan MemoryPlan) ([]byte, error) {
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	head := mll.HeadSection{
		Name:        strg.Intern(plan.ModuleName),
		Description: strg.Intern("Manta memory plan"),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, "memory_plan_version", plan.Version),
			headStringMeta(strg, "backend", string(plan.Backend)),
			headIntMeta(strg, "weight_count", int64(len(plan.Weights))),
			headIntMeta(strg, "total_weight_bytes", plan.TotalWeightBytes),
			headIntMeta(strg, "device_resident_bytes", plan.DeviceResidentBytes),
			headIntMeta(strg, "host_resident_bytes", plan.HostResidentBytes),
			headIntMeta(strg, "lazy_staged_bytes", plan.LazyStagedBytes),
		},
	}
	body, err := json.Marshal(plan)
	if err != nil {
		return nil, err
	}

	var (
		dimsBuilder mll.DimsBuilder
		typeBuilder mll.TypeBuilder
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		tnsrBuilder mll.TnsrBuilder
	)

	sections := make([]mll.SectionInput, 0, 9)
	if body, digestBody, err := encodeHeadSection(head, mll.ProfileSealed); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagHEAD,
			Body:          body,
			DigestBody:    digestBody,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(strg.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagSTRG,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(dimsBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagDIMS,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(typeBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagPARM,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(entrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagENTR,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagTNSR,
			Body:          body,
			Flags:         mll.SectionFlagRequired | mll.SectionFlagAligned,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection((mll.SchmSection{}).Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSCHM, Body: body, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           tagXMEM,
		Body:          body,
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})
	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func decodeMemoryPlanMLL(data []byte) (MemoryPlan, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return MemoryPlan{}, err
	}
	if reader.Profile() != mll.ProfileSealed {
		return MemoryPlan{}, fmt.Errorf("memory plan profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}
	body, ok := reader.Section(tagXMEM)
	if !ok {
		return MemoryPlan{}, fmt.Errorf("memory plan missing XMEM section")
	}
	var plan MemoryPlan
	if err := json.Unmarshal(body, &plan); err != nil {
		return MemoryPlan{}, err
	}
	if err := plan.Validate(); err != nil {
		return MemoryPlan{}, err
	}
	return plan, nil
}

func cloneMemoryPlan(plan *MemoryPlan) *MemoryPlan {
	if plan == nil {
		return nil
	}
	cp := *plan
	cp.Weights = append([]WeightMemoryPlan(nil), plan.Weights...)
	for i := range cp.Weights {
		cp.Weights[i].Shape = append([]int(nil), plan.Weights[i].Shape...)
	}
	return &cp
}

func countParamAccesses(mod *mantaartifact.Module) map[string]int {
	counts := map[string]int{}
	if mod == nil {
		return counts
	}
	params := map[string]bool{}
	for _, param := range mod.Params {
		params[param.Name] = true
	}
	for _, step := range mod.Steps {
		for _, input := range step.Inputs {
			if params[input] {
				counts[input]++
			}
		}
	}
	return counts
}

func tensorLogicalBytes(t *backend.Tensor) int64 {
	if t == nil {
		return 0
	}
	elements := int64(1)
	for _, dim := range t.Shape {
		if dim <= 0 {
			return 0
		}
		elements *= int64(dim)
	}
	switch t.DType {
	case "q4":
		return (elements + 1) / 2
	case "q8":
		return elements
	case "f16":
		return elements * 2
	case "f32", "i32":
		return elements * 4
	case "i64":
		return elements * 8
	default:
		return elements * 4
	}
}
