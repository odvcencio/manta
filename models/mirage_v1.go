package models

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

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
	HyperChannels  int
	BitWidth       int
	Seed           int64
	Factorization  string
	Lambda         float64
	LambdaSet      bool
}

// DefaultMirageV1Module returns a runnable host-reference Mirage Image v1 module
// with the three deployment entrypoints from the codec spec plus a training
// smoke entrypoint over the same graph.
func DefaultMirageV1Module(cfg MirageV1Config) (*mantaartifact.Module, error) {
	cfg = cfg.normalized()
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	factorization := canonicalMirageFactorization(cfg.Factorization)
	levels := 1 << cfg.BitWidth
	latentH := cfg.ImageHeight / 16
	latentW := cfg.ImageWidth / 16
	hyperH := latentH
	hyperW := latentW
	rateScale := 1 / float64(cfg.ImageWidth*cfg.ImageHeight)
	bitsDType := fmt.Sprintf("q%d", cfg.BitWidth)
	imageShape := []string{"1", itoa(cfg.ImageChannels), itoa(cfg.ImageHeight), itoa(cfg.ImageWidth)}
	latentShape := []string{"1", itoa(cfg.LatentChannels), itoa(latentH), itoa(latentW)}
	hyperShape := []string{"1", itoa(cfg.HyperChannels), itoa(hyperH), itoa(hyperW)}
	normShape := []string{"1", itoa(latentH), itoa(latentW)}
	hyperNormShape := []string{"1", itoa(hyperH), itoa(hyperW)}
	coordLogitChannels := cfg.LatentChannels * levels
	if factorization == "bit-plane" {
		coordLogitChannels = cfg.LatentChannels * cfg.BitWidth * 2
	}
	coordLogitShape := []string{"1", itoa(coordLogitChannels), itoa(latentH), itoa(latentW)}
	normParamShape := []string{"1", "2", itoa(latentH), itoa(latentW)}
	scalarShape := []string{"1"}

	mod := mantaartifact.NewModule(cfg.Name)
	mod.Requirements.Capabilities = []string{
		mantaartifact.CapabilityImageOps,
		mantaartifact.CapabilityTurboQuant,
		mantaartifact.CapabilityTrainingLosses,
		mantaartifact.CapabilityHostFallback,
	}
	mod.Params = append(mod.Params,
		paramTensor("ga0_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.ImageChannels), "5", "5"}),
		paramTensor("ga0_bias", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("ga1_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels), "5", "5"}),
		paramTensor("ga1_bias", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("ga2_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels), "5", "5"}),
		paramTensor("ga2_bias", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("ga3_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels), "5", "5"}),
		paramTensor("ga3_bias", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gdn0_beta", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gdn0_gamma", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels)}),
		paramTensor("gdn1_beta", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gdn1_gamma", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels)}),
		paramTensor("gdn2_beta", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gdn2_gamma", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels)}),
		paramTensor("ha_weight", "f16", []string{itoa(cfg.HyperChannels), itoa(cfg.LatentChannels), "3", "3"}),
		paramTensor("ha_bias", "f16", []string{itoa(cfg.HyperChannels)}),
		paramTensor("hs_logits_weight", "f16", []string{itoa(coordLogitChannels), itoa(cfg.HyperChannels), "3", "3"}),
		paramTensor("hs_logits_bias", "f16", []string{itoa(coordLogitChannels)}),
		paramTensor("hs_norm_weight", "f16", []string{"2", itoa(cfg.HyperChannels), "3", "3"}),
		paramTensor("hs_norm_bias", "f16", []string{"2"}),
		paramTensor("prior_z_logits", "f16", []string{itoa(levels)}),
		paramTensor("gs0_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels), "5", "5"}),
		paramTensor("gs0_bias", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gs1_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels), "5", "5"}),
		paramTensor("gs1_bias", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gs2_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels), "5", "5"}),
		paramTensor("gs2_bias", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("gs3_weight", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.ImageChannels), "5", "5"}),
		paramTensor("gs3_bias", "f16", []string{itoa(cfg.ImageChannels)}),
		paramTensor("igdn0_beta", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("igdn0_gamma", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels)}),
		paramTensor("igdn1_beta", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("igdn1_gamma", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels)}),
		paramTensor("igdn2_beta", "f16", []string{itoa(cfg.LatentChannels)}),
		paramTensor("igdn2_gamma", "f16", []string{itoa(cfg.LatentChannels), itoa(cfg.LatentChannels)}),
	)
	mod.EntryPoints = []mantaartifact.EntryPoint{
		{
			Name: "train_step",
			Kind: mantaartifact.EntryPointPipeline,
			Inputs: []mantaartifact.ValueBinding{
				valueBinding("x", "f16", imageShape),
			},
			Outputs: []mantaartifact.ValueBinding{
				valueBinding("loss", "f32", scalarShape),
				valueBinding("x_hat", "f16", imageShape),
				valueBinding("c_z", bitsDType, hyperShape),
				valueBinding("c_z_norms", "q_norm", hyperNormShape),
				valueBinding("c_coords", bitsDType, latentShape),
				valueBinding("c_norms", "q_norm", normShape),
				valueBinding("pi_logits", "f16", coordLogitShape),
				valueBinding("norm_params", "f16", normParamShape),
				valueBinding("rate", "f32", scalarShape),
				valueBinding("rate_z", "f32", scalarShape),
				valueBinding("rate_coords", "f32", scalarShape),
				valueBinding("rate_norms", "f32", scalarShape),
				valueBinding("mse", "f32", scalarShape),
				valueBinding("ms_ssim", "f32", scalarShape),
			},
		},
		{
			Name: "analyze",
			Kind: mantaartifact.EntryPointPipeline,
			Inputs: []mantaartifact.ValueBinding{
				valueBinding("x", "f16", imageShape),
			},
			Outputs: []mantaartifact.ValueBinding{
				valueBinding("c_z", bitsDType, hyperShape),
				valueBinding("c_z_norms", "q_norm", hyperNormShape),
				valueBinding("c_coords", bitsDType, latentShape),
				valueBinding("c_norms", "q_norm", normShape),
				valueBinding("pi_logits", "f16", coordLogitShape),
				valueBinding("norm_params", "f16", normParamShape),
			},
		},
		{
			Name: "synthesize_hyperprior",
			Kind: mantaartifact.EntryPointPipeline,
			Inputs: []mantaartifact.ValueBinding{
				valueBinding("c_z", bitsDType, hyperShape),
				valueBinding("c_z_norms", "q_norm", hyperNormShape),
			},
			Outputs: []mantaartifact.ValueBinding{
				valueBinding("pi_logits", "f16", coordLogitShape),
				valueBinding("norm_params", "f16", normParamShape),
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
		buffer("ga0", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 2), itoa(cfg.ImageWidth / 2)}),
		buffer("ga0_gdn", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 2), itoa(cfg.ImageWidth / 2)}),
		buffer("ga1", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 4), itoa(cfg.ImageWidth / 4)}),
		buffer("ga1_gdn", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 4), itoa(cfg.ImageWidth / 4)}),
		buffer("ga2", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 8), itoa(cfg.ImageWidth / 8)}),
		buffer("ga2_gdn", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 8), itoa(cfg.ImageWidth / 8)}),
		buffer("y", "f16", latentShape),
		buffer("c_coords", bitsDType, latentShape),
		buffer("c_norms", "q_norm", normShape),
		buffer("y_hat", "f16", latentShape),
		buffer("z", "f16", hyperShape),
		buffer("c_z", bitsDType, hyperShape),
		buffer("c_z_norms", "q_norm", hyperNormShape),
		buffer("z_hat", "f16", hyperShape),
		buffer("pi_logits", "f16", coordLogitShape),
		buffer("norm_params", "f16", normParamShape),
		buffer("gs0", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 8), itoa(cfg.ImageWidth / 8)}),
		buffer("gs0_igdn", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 8), itoa(cfg.ImageWidth / 8)}),
		buffer("gs1", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 4), itoa(cfg.ImageWidth / 4)}),
		buffer("gs1_igdn", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 4), itoa(cfg.ImageWidth / 4)}),
		buffer("gs2", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 2), itoa(cfg.ImageWidth / 2)}),
		buffer("gs2_igdn", "f16", []string{"1", itoa(cfg.LatentChannels), itoa(cfg.ImageHeight / 2), itoa(cfg.ImageWidth / 2)}),
		buffer("x_hat", "f16", imageShape),
		buffer("rate_z", "f32", scalarShape),
		buffer("rate_coords", "f32", scalarShape),
		buffer("rate_norms", "f32", scalarShape),
		buffer("rate", "f32", scalarShape),
		buffer("mse", "f32", scalarShape),
		buffer("ms_ssim", "f32", scalarShape),
		buffer("loss", "f32", scalarShape),
	}
	commonAttrs := map[string]string{
		"bits": itoa(cfg.BitWidth),
		"seed": itoa64(cfg.Seed),
	}
	coordRateAttrs := map[string]string{
		"bits":          itoa(cfg.BitWidth),
		"factorization": factorization,
		"logits_layout": "nchw_alphabet",
	}
	zRateAttrs := map[string]string{"bits": itoa(cfg.BitWidth)}
	normRateAttrs := map[string]string{"distribution": "log_normal", "sigma_parameter": "softplus"}

	addAnalysis := func(entry string) {
		mod.Steps = append(mod.Steps,
			step(entry, mantaartifact.StepConv2D, "analysis_0", []string{"x", "ga0_weight", "ga0_bias"}, []string{"ga0"}, convAttrs("2", "2", "1")),
			step(entry, mantaartifact.StepGDN, "analysis_gdn_0", []string{"ga0", "gdn0_beta", "gdn0_gamma"}, []string{"ga0_gdn"}, nil),
			step(entry, mantaartifact.StepConv2D, "analysis_1", []string{"ga0_gdn", "ga1_weight", "ga1_bias"}, []string{"ga1"}, convAttrs("2", "2", "1")),
			step(entry, mantaartifact.StepGDN, "analysis_gdn_1", []string{"ga1", "gdn1_beta", "gdn1_gamma"}, []string{"ga1_gdn"}, nil),
			step(entry, mantaartifact.StepConv2D, "analysis_2", []string{"ga1_gdn", "ga2_weight", "ga2_bias"}, []string{"ga2"}, convAttrs("2", "2", "1")),
			step(entry, mantaartifact.StepGDN, "analysis_gdn_2", []string{"ga2", "gdn2_beta", "gdn2_gamma"}, []string{"ga2_gdn"}, nil),
			step(entry, mantaartifact.StepConv2D, "analysis_3", []string{"ga2_gdn", "ga3_weight", "ga3_bias"}, []string{"y"}, convAttrs("2", "2", "1")),
			step(entry, mantaartifact.StepTurboQEncode, "latent_quantize", []string{"y"}, []string{"c_coords", "c_norms"}, commonAttrs),
		)
	}
	addHyperAnalysis := func(entry string) {
		mod.Steps = append(mod.Steps,
			step(entry, mantaartifact.StepTurboQDecode, "latent_dequantize", []string{"c_coords", "c_norms"}, []string{"y_hat"}, commonAttrs),
			step(entry, mantaartifact.StepConv2D, "hyper_analysis", []string{"y_hat", "ha_weight", "ha_bias"}, []string{"z"}, convAttrs("1", "1", "1")),
			step(entry, mantaartifact.StepTurboQEncode, "hyper_quantize", []string{"z"}, []string{"c_z", "c_z_norms"}, commonAttrs),
			step(entry, mantaartifact.StepTurboQDecode, "hyper_dequantize", []string{"c_z", "c_z_norms"}, []string{"z_hat"}, commonAttrs),
			step(entry, mantaartifact.StepConv2D, "hyper_synthesis_logits", []string{"z_hat", "hs_logits_weight", "hs_logits_bias"}, []string{"pi_logits"}, convAttrs("1", "1", "1")),
			step(entry, mantaartifact.StepConv2D, "hyper_synthesis_norms", []string{"z_hat", "hs_norm_weight", "hs_norm_bias"}, []string{"norm_params"}, convAttrs("1", "1", "1")),
		)
	}
	addHyperSynthesis := func(entry string) {
		mod.Steps = append(mod.Steps,
			step(entry, mantaartifact.StepTurboQDecode, "hyper_dequantize", []string{"c_z", "c_z_norms"}, []string{"z_hat"}, commonAttrs),
			step(entry, mantaartifact.StepConv2D, "hyper_synthesis_logits", []string{"z_hat", "hs_logits_weight", "hs_logits_bias"}, []string{"pi_logits"}, convAttrs("1", "1", "1")),
			step(entry, mantaartifact.StepConv2D, "hyper_synthesis_norms", []string{"z_hat", "hs_norm_weight", "hs_norm_bias"}, []string{"norm_params"}, convAttrs("1", "1", "1")),
		)
	}
	addSynthesisLayers := func(entry, latentInput string) {
		mod.Steps = append(mod.Steps,
			step(entry, mantaartifact.StepConv2DTrans, "synthesis_0", []string{latentInput, "gs0_weight", "gs0_bias"}, []string{"gs0"}, convTransposeAttrs("2", "2", "1", "1")),
			step(entry, mantaartifact.StepIGDN, "synthesis_igdn_0", []string{"gs0", "igdn0_beta", "igdn0_gamma"}, []string{"gs0_igdn"}, nil),
			step(entry, mantaartifact.StepConv2DTrans, "synthesis_1", []string{"gs0_igdn", "gs1_weight", "gs1_bias"}, []string{"gs1"}, convTransposeAttrs("2", "2", "1", "1")),
			step(entry, mantaartifact.StepIGDN, "synthesis_igdn_1", []string{"gs1", "igdn1_beta", "igdn1_gamma"}, []string{"gs1_igdn"}, nil),
			step(entry, mantaartifact.StepConv2DTrans, "synthesis_2", []string{"gs1_igdn", "gs2_weight", "gs2_bias"}, []string{"gs2"}, convTransposeAttrs("2", "2", "1", "1")),
			step(entry, mantaartifact.StepIGDN, "synthesis_igdn_2", []string{"gs2", "igdn2_beta", "igdn2_gamma"}, []string{"gs2_igdn"}, nil),
			step(entry, mantaartifact.StepConv2DTrans, "synthesis_3", []string{"gs2_igdn", "gs3_weight", "gs3_bias"}, []string{"x_hat"}, convTransposeAttrs("2", "2", "1", "1")),
		)
	}
	addSynthesis := func(entry string) {
		mod.Steps = append(mod.Steps,
			step(entry, mantaartifact.StepTurboQDecode, "latent_dequantize", []string{"c_coords", "c_norms"}, []string{"y_hat"}, commonAttrs),
		)
		addSynthesisLayers(entry, "y_hat")
	}
	addAnalysis("train_step")
	addHyperAnalysis("train_step")
	mod.Steps = append(mod.Steps,
		step("train_step", mantaartifact.StepCrossEntropy, "rate_hyperprior", []string{"c_z", "prior_z_logits"}, []string{"rate_z"}, zRateAttrs),
		step("train_step", mantaartifact.StepCrossEntropy, "rate_coords", []string{"c_coords", "pi_logits"}, []string{"rate_coords"}, coordRateAttrs),
		step("train_step", mantaartifact.StepCrossEntropy, "rate_norms", []string{"c_norms", "norm_params"}, []string{"rate_norms"}, normRateAttrs),
		step("train_step", mantaartifact.StepScalarAdd, "rate_sum", []string{"rate_z", "rate_coords", "rate_norms"}, []string{"rate"}, nil),
	)
	addSynthesisLayers("train_step", "y")
	mod.Steps = append(mod.Steps,
		step("train_step", mantaartifact.StepMSELoss, "distortion_mse", []string{"x", "x_hat"}, []string{"mse"}, nil),
		step("train_step", mantaartifact.StepMSSSIMLoss, "distortion_ms_ssim", []string{"x", "x_hat"}, []string{"ms_ssim"}, nil),
		step("train_step", mantaartifact.StepRDLoss, "loss", []string{"mse", "rate"}, []string{"loss"}, map[string]string{
			"lambda":     fmt.Sprintf("%.8g", cfg.Lambda),
			"rate_scale": fmt.Sprintf("%.8g", rateScale),
		}),
		step("train_step", mantaartifact.StepReturn, "return", nil, []string{"loss", "x_hat", "c_z", "c_z_norms", "c_coords", "c_norms", "pi_logits", "norm_params", "rate", "rate_z", "rate_coords", "rate_norms", "mse", "ms_ssim"}, nil),
	)
	addAnalysis("analyze")
	addHyperAnalysis("analyze")
	mod.Steps = append(mod.Steps, step("analyze", mantaartifact.StepReturn, "return", nil, []string{"c_z", "c_z_norms", "c_coords", "c_norms", "pi_logits", "norm_params"}, nil))
	addHyperSynthesis("synthesize_hyperprior")
	mod.Steps = append(mod.Steps, step("synthesize_hyperprior", mantaartifact.StepReturn, "return", nil, []string{"pi_logits", "norm_params"}, nil))
	addSynthesis("synthesize_image")
	mod.Steps = append(mod.Steps, step("synthesize_image", mantaartifact.StepReturn, "return", nil, []string{"x_hat"}, nil))
	mod.Metadata = map[string]any{
		"family":              "mirage",
		"phase":               "v1",
		"image_shape":         imageShape,
		"latent_shape":        latentShape,
		"hyper_shape":         hyperShape,
		"coord_logits_shape":  coordLogitShape,
		"norm_params_shape":   normParamShape,
		"bit_width":           cfg.BitWidth,
		"factorization":       factorization,
		"lambda":              cfg.Lambda,
		"rate_scale":          rateScale,
		"rate_unit":           "bits_per_pixel",
		"execution_scope":     "host_reference",
		"entrypoint_surface":  "analyze+synthesize_hyperprior+synthesize_image+train_step",
		"hyperprior_payloads": "c_z and c_z_norms are packed into the .mrg c_z stream by the host codec",
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
		cfg.ImageHeight = 16
	}
	if cfg.ImageWidth == 0 {
		cfg.ImageWidth = 16
	}
	if cfg.LatentChannels == 0 {
		cfg.LatentChannels = 4
	}
	if cfg.HyperChannels == 0 {
		cfg.HyperChannels = cfg.LatentChannels
	}
	if cfg.BitWidth == 0 {
		cfg.BitWidth = 4
	}
	if cfg.Seed == 0 {
		cfg.Seed = 0x4d697261676531
	}
	if cfg.Factorization == "" {
		cfg.Factorization = "categorical"
	}
	if cfg.Lambda == 0 && !cfg.LambdaSet {
		cfg.Lambda = 0.01
	}
	return cfg
}

func (cfg MirageV1Config) validate() error {
	if cfg.ImageChannels <= 0 || cfg.ImageHeight <= 0 || cfg.ImageWidth <= 0 {
		return fmt.Errorf("image shape must be positive")
	}
	if cfg.ImageHeight%16 != 0 || cfg.ImageWidth%16 != 0 {
		return fmt.Errorf("image height and width must be divisible by 16")
	}
	if cfg.LatentChannels < 2 {
		return fmt.Errorf("latent channels must be at least 2")
	}
	if cfg.HyperChannels < 2 {
		return fmt.Errorf("hyper channels must be at least 2")
	}
	if cfg.BitWidth != 2 && cfg.BitWidth != 4 && cfg.BitWidth != 8 {
		return fmt.Errorf("bit width must be 2, 4, or 8")
	}
	if canonicalMirageFactorization(cfg.Factorization) == "" {
		return fmt.Errorf("factorization must be categorical or bit-plane")
	}
	if cfg.Lambda < 0 {
		return fmt.Errorf("lambda must be non-negative")
	}
	return nil
}

func canonicalMirageFactorization(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "categorical", "cat":
		return "categorical"
	case "bit-plane", "bitplane", "bit_plane", "bits":
		return "bit-plane"
	default:
		return ""
	}
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
