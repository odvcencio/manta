package models

import (
	"fmt"
	"os"
	"path/filepath"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

const DefaultMirageV1ModelName = "mirage_image_v1"

// MirageV1Config controls the built-in Mirage Image v1 Manta artifact shape.
type MirageV1Config struct {
	Name           string
	ImageChannels  int
	ImageHeight    int
	ImageWidth     int
	LatentChannels int
	BitWidth       int
	Seed           int64
}

// DefaultMirageV1Module returns a runnable host-reference Mirage Image v1
// module that exercises the Manta image-codec operator surface.
func DefaultMirageV1Module(cfg MirageV1Config) (*mantaartifact.Module, error) {
	cfg = cfg.normalized()
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	latentH := ceilDiv(cfg.ImageHeight, 2)
	latentW := ceilDiv(cfg.ImageWidth, 2)
	bitsDType := fmt.Sprintf("q%d", cfg.BitWidth)
	imageShape := []string{"1", itoa(cfg.ImageChannels), itoa(cfg.ImageHeight), itoa(cfg.ImageWidth)}
	latentShape := []string{"1", itoa(cfg.LatentChannels), itoa(latentH), itoa(latentW)}
	normShape := []string{"1", itoa(latentH), itoa(latentW)}
	scalarShape := []string{"1"}

	mod := mantaartifact.NewModule(cfg.Name)
	mod.Requirements.Capabilities = []string{
		mantaartifact.CapabilityImageOps,
		mantaartifact.CapabilityTurboQuant,
		mantaartifact.CapabilityTrainingLosses,
		mantaartifact.CapabilityHostFallback,
	}
	mod.Params = []mantaartifact.Param{
		paramTensor("ga_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.ImageChannels), "3", "3"}),
		paramTensor("ga_bias", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gdn_beta", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gdn_gamma", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels)}),
		paramTensor("gs_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.ImageChannels), "3", "3"}),
		paramTensor("gs_bias", "f16", []string{itoa(cfg.ImageChannels)}),
		paramTensor("prior_logits", "f16", []string{itoa(1 << cfg.BitWidth)}),
	}
	mod.EntryPoints = []mantaartifact.EntryPoint{
		{
			Name: "train_step",
			Kind: mantaartifact.EntryPointPipeline,
			Inputs: []mantaartifact.ValueBinding{
				valueBinding("x", "f16", imageShape),
			},
			Outputs: []mantaartifact.ValueBinding{
				valueBinding("x_hat", "f16", imageShape),
				valueBinding("rate", "f32", scalarShape),
				valueBinding("mse", "f32", scalarShape),
				valueBinding("ms_ssim", "f32", scalarShape),
			},
		},
		{
			Name: "synthesize_image",
			Kind: mantaartifact.EntryPointPipeline,
			Inputs: []mantaartifact.ValueBinding{
				valueBinding("c_coords", bitsDType, latentShape),
				valueBinding("c_norms", "q_norm", normShape),
			},
			Outputs: []mantaartifact.ValueBinding{
				valueBinding("x_hat", "f16", imageShape),
			},
		},
	}
	mod.Buffers = []mantaartifact.Buffer{
		buffer("y", "f16", latentShape),
		buffer("y_gdn", "f16", latentShape),
		buffer("c_coords", bitsDType, latentShape),
		buffer("c_norms", "q_norm", normShape),
		buffer("rate", "f32", scalarShape),
		buffer("y_hat", "f16", latentShape),
		buffer("y_igdn", "f16", latentShape),
		buffer("x_hat", "f16", imageShape),
		buffer("mse", "f32", scalarShape),
		buffer("ms_ssim", "f32", scalarShape),
	}
	commonAttrs := map[string]string{
		"bits": itoa(cfg.BitWidth),
		"seed": itoa64(cfg.Seed),
	}
	mod.Steps = []mantaartifact.Step{
		step("train_step", mantaartifact.StepConv2D, "analysis", []string{"x", "ga_weight", "ga_bias"}, []string{"y"}, convAttrs("2", "1", "1")),
		step("train_step", mantaartifact.StepGDN, "analysis_gdn", []string{"y", "gdn_beta", "gdn_gamma"}, []string{"y_gdn"}, nil),
		step("train_step", mantaartifact.StepTurboQEncode, "latent_quantize", []string{"y_gdn"}, []string{"c_coords", "c_norms"}, commonAttrs),
		step("train_step", mantaartifact.StepCrossEntropy, "rate_model", []string{"c_coords", "prior_logits"}, []string{"rate"}, map[string]string{"bits": itoa(cfg.BitWidth)}),
		step("train_step", mantaartifact.StepTurboQDecode, "latent_dequantize", []string{"c_coords", "c_norms"}, []string{"y_hat"}, commonAttrs),
		step("train_step", mantaartifact.StepIGDN, "synthesis_igdn", []string{"y_hat", "gdn_beta", "gdn_gamma"}, []string{"y_igdn"}, nil),
		step("train_step", mantaartifact.StepConv2DTrans, "synthesis", []string{"y_igdn", "gs_weight", "gs_bias"}, []string{"x_hat"}, convTransposeAttrs("2", "1", "1", "1")),
		step("train_step", mantaartifact.StepMSELoss, "distortion_mse", []string{"x", "x_hat"}, []string{"mse"}, nil),
		step("train_step", mantaartifact.StepMSSSIMLoss, "distortion_ms_ssim", []string{"x", "x_hat"}, []string{"ms_ssim"}, nil),
		step("train_step", mantaartifact.StepReturn, "return", nil, []string{"x_hat", "rate", "mse", "ms_ssim"}, nil),
		step("synthesize_image", mantaartifact.StepTurboQDecode, "latent_dequantize", []string{"c_coords", "c_norms"}, []string{"y_hat"}, commonAttrs),
		step("synthesize_image", mantaartifact.StepIGDN, "synthesis_igdn", []string{"y_hat", "gdn_beta", "gdn_gamma"}, []string{"y_igdn"}, nil),
		step("synthesize_image", mantaartifact.StepConv2DTrans, "synthesis", []string{"y_igdn", "gs_weight", "gs_bias"}, []string{"x_hat"}, convTransposeAttrs("2", "1", "1", "1")),
		step("synthesize_image", mantaartifact.StepReturn, "return", nil, []string{"x_hat"}, nil),
	}
	mod.Metadata = map[string]any{
		"family":          "mirage",
		"phase":           "v1",
		"image_shape":     imageShape,
		"latent_shape":    latentShape,
		"bit_width":       cfg.BitWidth,
		"execution_scope": "host_reference",
	}
	return mod, mod.Validate()
}

// InitMirageV1Artifact writes the built-in Mirage Image v1 module as an .mll.
func InitMirageV1Artifact(path string, cfg MirageV1Config) error {
	if path == "" {
		return fmt.Errorf("output artifact path is required")
	}
	mod, err := DefaultMirageV1Module(cfg)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return mantaartifact.WriteFile(path, mod)
}

func (cfg MirageV1Config) normalized() MirageV1Config {
	if cfg.Name == "" {
		cfg.Name = DefaultMirageV1ModelName
	}
	if cfg.ImageChannels == 0 {
		cfg.ImageChannels = 3
	}
	if cfg.ImageHeight == 0 {
		cfg.ImageHeight = 8
	}
	if cfg.ImageWidth == 0 {
		cfg.ImageWidth = 8
	}
	if cfg.LatentChannels == 0 {
		cfg.LatentChannels = 4
	}
	if cfg.BitWidth == 0 {
		cfg.BitWidth = 4
	}
	if cfg.Seed == 0 {
		cfg.Seed = 0x4d697261676531
	}
	return cfg
}

func (cfg MirageV1Config) validate() error {
	if cfg.ImageChannels <= 0 || cfg.ImageHeight <= 0 || cfg.ImageWidth <= 0 {
		return fmt.Errorf("image shape must be positive")
	}
	if cfg.LatentChannels < 2 {
		return fmt.Errorf("latent channels must be at least 2")
	}
	if cfg.BitWidth != 2 && cfg.BitWidth != 4 && cfg.BitWidth != 8 {
		return fmt.Errorf("bit width must be 2, 4, or 8")
	}
	return nil
}

func paramTensor(name, dtype string, shape []string) mantaartifact.Param {
	return mantaartifact.Param{
		Name:      name,
		Type:      tensorType(dtype, shape),
		Binding:   "weights/" + name,
		Trainable: true,
	}
}

func valueBinding(name, dtype string, shape []string) mantaartifact.ValueBinding {
	return mantaartifact.ValueBinding{Name: name, Type: tensorType(dtype, shape)}
}

func tensorType(dtype string, shape []string) mantaartifact.ValueType {
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: dtype, Shape: append([]string(nil), shape...)},
	}
}

func buffer(name, dtype string, shape []string) mantaartifact.Buffer {
	return mantaartifact.Buffer{Name: name, DType: dtype, Shape: append([]string(nil), shape...)}
}

func step(entry string, kind mantaartifact.StepKind, name string, inputs, outputs []string, attrs map[string]string) mantaartifact.Step {
	return mantaartifact.Step{
		Entry:      entry,
		Kind:       kind,
		Name:       name,
		Inputs:     append([]string(nil), inputs...),
		Outputs:    append([]string(nil), outputs...),
		Attributes: cloneStringMap(attrs),
	}
}

func convAttrs(stride, padding, dilation string) map[string]string {
	return map[string]string{
		"stride":   stride,
		"padding":  padding,
		"dilation": dilation,
	}
}

func convTransposeAttrs(stride, padding, dilation, outputPadding string) map[string]string {
	attrs := convAttrs(stride, padding, dilation)
	attrs["output_padding"] = outputPadding
	return attrs
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func ceilDiv(n, d int) int {
	return (n + d - 1) / d
}

func itoa(n int) string {
	return fmt.Sprintf("%d", n)
}

func itoa64(n int64) string {
	return fmt.Sprintf("%d", n)
}
