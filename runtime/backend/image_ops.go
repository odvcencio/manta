package backend

import (
	"fmt"
	"math"
	"strconv"

	turboquant "github.com/odvcencio/turboquant"
)

const (
	qNormLogMin = -16.0
	qNormLogMax = 16.0
)

func conv2DTensor(input, weight, bias *Tensor, attrs map[string]string) (*Tensor, error) {
	if input == nil || weight == nil {
		return nil, fmt.Errorf("conv2d expects input and weight")
	}
	if len(input.Shape) != 4 || len(weight.Shape) != 4 {
		return nil, fmt.Errorf("conv2d expects input NCHW and weight OIHW")
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	outC, inPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	groups := attrInt(attrs, "groups", 1)
	if groups <= 0 {
		return nil, fmt.Errorf("conv2d groups must be positive")
	}
	if inPerGroup*groups != inC {
		return nil, fmt.Errorf("conv2d input channels %d do not match weight channels %d and groups %d", inC, inPerGroup, groups)
	}
	if outC%groups != 0 {
		return nil, fmt.Errorf("conv2d output channels %d not divisible by groups %d", outC, groups)
	}
	strideH, strideW := attrInt(attrs, "stride_h", attrInt(attrs, "stride", 1)), attrInt(attrs, "stride_w", attrInt(attrs, "stride", 1))
	padH, padW := attrInt(attrs, "pad_h", attrInt(attrs, "padding", 0)), attrInt(attrs, "pad_w", attrInt(attrs, "padding", 0))
	dilH, dilW := attrInt(attrs, "dilation_h", attrInt(attrs, "dilation", 1)), attrInt(attrs, "dilation_w", attrInt(attrs, "dilation", 1))
	if strideH <= 0 || strideW <= 0 || dilH <= 0 || dilW <= 0 {
		return nil, fmt.Errorf("conv2d stride and dilation must be positive")
	}
	outH := (inH+2*padH-dilH*(kH-1)-1)/strideH + 1
	outW := (inW+2*padW-dilW*(kW-1)-1)/strideW + 1
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("conv2d produced non-positive output shape [%d,%d]", outH, outW)
	}
	if bias != nil && len(bias.F32) < outC {
		return nil, fmt.Errorf("conv2d bias has %d values, want at least %d", len(bias.F32), outC)
	}
	out := tensorForDType(input.DType, []int{n, outC, outH, outW}, n*outC*outH*outW)
	outPerGroup := outC / groups
	for b := 0; b < n; b++ {
		for g := 0; g < groups; g++ {
			for ocg := 0; ocg < outPerGroup; ocg++ {
				oc := g*outPerGroup + ocg
				for oy := 0; oy < outH; oy++ {
					for ox := 0; ox < outW; ox++ {
						sum := float32(0)
						if bias != nil {
							sum = bias.F32[oc]
						}
						for icg := 0; icg < inPerGroup; icg++ {
							ic := g*inPerGroup + icg
							for ky := 0; ky < kH; ky++ {
								iy := oy*strideH + ky*dilH - padH
								if iy < 0 || iy >= inH {
									continue
								}
								for kx := 0; kx < kW; kx++ {
									ix := ox*strideW + kx*dilW - padW
									if ix < 0 || ix >= inW {
										continue
									}
									sum += input.F32[offset4(input.Shape, b, ic, iy, ix)] * weight.F32[((oc*inPerGroup+icg)*kH+ky)*kW+kx]
								}
							}
						}
						out.F32[offset4(out.Shape, b, oc, oy, ox)] = sum
					}
				}
			}
		}
	}
	return out, nil
}

// Conv2DReference runs the backend-neutral reference conv2d implementation.
func Conv2DReference(input, weight, bias *Tensor, attrs map[string]string) (*Tensor, error) {
	return conv2DTensor(input, weight, bias, attrs)
}

func conv2DTransposeTensor(input, weight, bias *Tensor, attrs map[string]string) (*Tensor, error) {
	if input == nil || weight == nil {
		return nil, fmt.Errorf("conv2d_transpose expects input and weight")
	}
	if len(input.Shape) != 4 || len(weight.Shape) != 4 {
		return nil, fmt.Errorf("conv2d_transpose expects input NCHW and weight IOHW")
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	weightInC, outPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	if weightInC != inC {
		return nil, fmt.Errorf("conv2d_transpose input channels %d do not match weight channels %d", inC, weightInC)
	}
	groups := attrInt(attrs, "groups", 1)
	if groups <= 0 || inC%groups != 0 {
		return nil, fmt.Errorf("conv2d_transpose groups must divide input channels")
	}
	outC := outPerGroup * groups
	strideH, strideW := attrInt(attrs, "stride_h", attrInt(attrs, "stride", 1)), attrInt(attrs, "stride_w", attrInt(attrs, "stride", 1))
	padH, padW := attrInt(attrs, "pad_h", attrInt(attrs, "padding", 0)), attrInt(attrs, "pad_w", attrInt(attrs, "padding", 0))
	dilH, dilW := attrInt(attrs, "dilation_h", attrInt(attrs, "dilation", 1)), attrInt(attrs, "dilation_w", attrInt(attrs, "dilation", 1))
	outPadH, outPadW := attrInt(attrs, "output_padding_h", attrInt(attrs, "output_padding", 0)), attrInt(attrs, "output_padding_w", attrInt(attrs, "output_padding", 0))
	if strideH <= 0 || strideW <= 0 || dilH <= 0 || dilW <= 0 {
		return nil, fmt.Errorf("conv2d_transpose stride and dilation must be positive")
	}
	outH := (inH-1)*strideH - 2*padH + dilH*(kH-1) + outPadH + 1
	outW := (inW-1)*strideW - 2*padW + dilW*(kW-1) + outPadW + 1
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("conv2d_transpose produced non-positive output shape [%d,%d]", outH, outW)
	}
	if bias != nil && len(bias.F32) < outC {
		return nil, fmt.Errorf("conv2d_transpose bias has %d values, want at least %d", len(bias.F32), outC)
	}
	out := tensorForDType(input.DType, []int{n, outC, outH, outW}, n*outC*outH*outW)
	inPerGroup := inC / groups
	for b := 0; b < n; b++ {
		for g := 0; g < groups; g++ {
			for icg := 0; icg < inPerGroup; icg++ {
				ic := g*inPerGroup + icg
				for iy := 0; iy < inH; iy++ {
					for ix := 0; ix < inW; ix++ {
						v := input.F32[offset4(input.Shape, b, ic, iy, ix)]
						for ocg := 0; ocg < outPerGroup; ocg++ {
							oc := g*outPerGroup + ocg
							for ky := 0; ky < kH; ky++ {
								oy := iy*strideH + ky*dilH - padH
								if oy < 0 || oy >= outH {
									continue
								}
								for kx := 0; kx < kW; kx++ {
									ox := ix*strideW + kx*dilW - padW
									if ox < 0 || ox >= outW {
										continue
									}
									out.F32[offset4(out.Shape, b, oc, oy, ox)] += v * weight.F32[((ic*outPerGroup+ocg)*kH+ky)*kW+kx]
								}
							}
						}
					}
				}
			}
		}
	}
	if bias != nil {
		for b := 0; b < n; b++ {
			for oc := 0; oc < outC; oc++ {
				for y := 0; y < outH; y++ {
					for x := 0; x < outW; x++ {
						out.F32[offset4(out.Shape, b, oc, y, x)] += bias.F32[oc]
					}
				}
			}
		}
	}
	return out, nil
}

// Conv2DTransposeReference runs the backend-neutral reference conv2d_transpose implementation.
func Conv2DTransposeReference(input, weight, bias *Tensor, attrs map[string]string) (*Tensor, error) {
	return conv2DTransposeTensor(input, weight, bias, attrs)
}

func gdnTensor(input, beta, gamma *Tensor, inverse bool) (*Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("gdn expects input")
	}
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("gdn expects NCHW input")
	}
	n, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	out := input.Clone()
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					sum := betaValue(beta, c)
					for j := 0; j < channels; j++ {
						v := input.F32[offset4(input.Shape, b, j, y, x)]
						sum += gammaValue(gamma, c, j, channels) * v * v
					}
					if sum < 1e-12 {
						sum = 1e-12
					}
					scale := float32(math.Sqrt(float64(sum)))
					idx := offset4(input.Shape, b, c, y, x)
					if inverse {
						out.F32[idx] = input.F32[idx] * scale
					} else {
						out.F32[idx] = input.F32[idx] / scale
					}
				}
			}
		}
	}
	return out, nil
}

// GDNReference runs the backend-neutral reference GDN/IGDN implementation.
func GDNReference(input, beta, gamma *Tensor, inverse bool) (*Tensor, error) {
	return gdnTensor(input, beta, gamma, inverse)
}

func turboQuantEncodeTensor(input *Tensor, attrs map[string]string) (*Tensor, *Tensor, error) {
	if input == nil {
		return nil, nil, fmt.Errorf("turboquant_encode expects input")
	}
	if len(input.Shape) != 4 {
		return nil, nil, fmt.Errorf("turboquant_encode expects NCHW input")
	}
	bits := attrInt(attrs, "bits", 4)
	if bits != 2 && bits != 4 && bits != 8 {
		return nil, nil, fmt.Errorf("turboquant bits must be 2, 4, or 8")
	}
	seed := int64(attrInt(attrs, "seed", 0x4d697261))
	n, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	coords := tensorForDType(fmt.Sprintf("q%d", bits), append([]int(nil), input.Shape...), input.Elements())
	norms := NewTensorQNorm([]int{n, height, width}, make([]float32, n*height*width))
	q := turboquant.NewHadamardWithSeed(channels, bits, seed)
	indices := make([]int, channels)
	vec := make([]float32, channels)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					vec[c] = input.F32[offset4(input.Shape, b, c, y, x)]
				}
				norm := q.QuantizeIndicesTo(indices, vec)
				for c := 0; c < channels; c++ {
					coords.F32[offset4(coords.Shape, b, c, y, x)] = float32(indices[c])
				}
				norms.F32[(b*height+y)*width+x] = float32(quantizeQNorm(norm))
			}
		}
	}
	return coords, norms, nil
}

// TurboQuantEncodeReference runs the backend-neutral reference TurboQuant encode implementation.
func TurboQuantEncodeReference(input *Tensor, attrs map[string]string) (*Tensor, *Tensor, error) {
	return turboQuantEncodeTensor(input, attrs)
}

func turboQuantDecodeTensor(coords, norms *Tensor, attrs map[string]string) (*Tensor, error) {
	if coords == nil || norms == nil {
		return nil, fmt.Errorf("turboquant_decode expects coords and norms")
	}
	if len(coords.Shape) != 4 || len(norms.Shape) != 3 {
		return nil, fmt.Errorf("turboquant_decode expects coords NCHW and norms NHW")
	}
	n, channels, height, width := coords.Shape[0], coords.Shape[1], coords.Shape[2], coords.Shape[3]
	if norms.Shape[0] != n || norms.Shape[1] != height || norms.Shape[2] != width {
		return nil, fmt.Errorf("turboquant_decode norms shape %v does not match coords %v", norms.Shape, coords.Shape)
	}
	bits := attrInt(attrs, "bits", bitsForQTensor(coords))
	if bits != 2 && bits != 4 && bits != 8 {
		return nil, fmt.Errorf("turboquant bits must be 2, 4, or 8")
	}
	seed := int64(attrInt(attrs, "seed", 0x4d697261))
	out := NewTensorF16(append([]int(nil), coords.Shape...), make([]float32, coords.Elements()))
	q := turboquant.NewHadamardWithSeed(channels, bits, seed)
	indices := make([]int, channels)
	vec := make([]float32, channels)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					indices[c] = clampInt(int(math.Round(float64(coords.F32[offset4(coords.Shape, b, c, y, x)]))), 0, (1<<bits)-1)
				}
				q.DequantizeIndicesTo(vec, indices)
				norm := dequantizeQNorm(byte(clampInt(int(math.Round(float64(norms.F32[(b*height+y)*width+x]))), 0, 255)))
				for c := 0; c < channels; c++ {
					out.F32[offset4(out.Shape, b, c, y, x)] = vec[c] * norm
				}
			}
		}
	}
	return out, nil
}

// TurboQuantDecodeReference runs the backend-neutral reference TurboQuant decode implementation.
func TurboQuantDecodeReference(coords, norms *Tensor, attrs map[string]string) (*Tensor, error) {
	return turboQuantDecodeTensor(coords, norms, attrs)
}

func crossEntropyFactorizedTensor(codes, logits *Tensor, attrs map[string]string) (*Tensor, error) {
	if codes == nil {
		return nil, fmt.Errorf("cross_entropy_factorized expects codes")
	}
	if attrs != nil && attrs["distribution"] == "log_normal" {
		return logNormalNormCrossEntropyTensor(codes, logits, attrs)
	}
	levels := attrInt(attrs, "levels", 0)
	if levels <= 0 {
		if bits := attrInt(attrs, "bits", bitsForQTensor(codes)); bits > 0 {
			levels = 1 << bits
		} else {
			levels = 256
		}
	}
	bits := attrInt(attrs, "bits", bitsForQTensor(codes))
	total := 0.0
	switch factorizationAttr(attrs) {
	case "bit-plane":
		if bits <= 0 {
			return nil, fmt.Errorf("cross_entropy_factorized bit-plane mode requires bits")
		}
		for i, raw := range codes.F32 {
			idx := clampInt(int(math.Round(float64(raw))), 0, (1<<bits)-1)
			for bit := 0; bit < bits; bit++ {
				shift := bits - 1 - bit
				bitValue := (idx >> shift) & 1
				p := probabilityForBit(logits, codes.Shape, i, bit, bitValue, bits)
				if p < 1e-12 {
					p = 1e-12
				}
				total += -math.Log2(p)
			}
		}
	default:
		for i, raw := range codes.F32 {
			idx := clampInt(int(math.Round(float64(raw))), 0, levels-1)
			p := probabilityForCode(logits, codes.Shape, i, idx, levels, attrs)
			if p < 1e-12 {
				p = 1e-12
			}
			total += -math.Log2(p)
		}
	}
	return NewTensorF32([]int{1}, []float32{float32(total)}), nil
}

func mseLossTensor(lhs, rhs *Tensor) (*Tensor, error) {
	if lhs == nil || rhs == nil {
		return nil, fmt.Errorf("mse_loss expects two tensors")
	}
	if !lhs.EqualShape(rhs) {
		return nil, fmt.Errorf("mse_loss shape mismatch %v vs %v", lhs.Shape, rhs.Shape)
	}
	if len(lhs.F32) != len(rhs.F32) {
		return nil, fmt.Errorf("mse_loss expects floating tensors")
	}
	if len(lhs.F32) == 0 {
		return NewTensorF32([]int{1}, []float32{0}), nil
	}
	sum := 0.0
	for i := range lhs.F32 {
		diff := float64(lhs.F32[i] - rhs.F32[i])
		sum += diff * diff
	}
	return NewTensorF32([]int{1}, []float32{float32(sum / float64(len(lhs.F32)))}), nil
}

func msSSIMLossTensor(lhs, rhs *Tensor) (*Tensor, error) {
	if lhs == nil || rhs == nil {
		return nil, fmt.Errorf("ms_ssim_loss expects two tensors")
	}
	if !lhs.EqualShape(rhs) {
		return nil, fmt.Errorf("ms_ssim_loss shape mismatch %v vs %v", lhs.Shape, rhs.Shape)
	}
	if len(lhs.F32) != len(rhs.F32) {
		return nil, fmt.Errorf("ms_ssim_loss expects floating tensors")
	}
	if len(lhs.F32) == 0 {
		return NewTensorF32([]int{1}, []float32{0}), nil
	}
	meanA, meanB := 0.0, 0.0
	for i := range lhs.F32 {
		meanA += float64(lhs.F32[i])
		meanB += float64(rhs.F32[i])
	}
	meanA /= float64(len(lhs.F32))
	meanB /= float64(len(rhs.F32))
	varA, varB, cov := 0.0, 0.0, 0.0
	for i := range lhs.F32 {
		da := float64(lhs.F32[i]) - meanA
		db := float64(rhs.F32[i]) - meanB
		varA += da * da
		varB += db * db
		cov += da * db
	}
	denom := float64(len(lhs.F32))
	varA /= denom
	varB /= denom
	cov /= denom
	const c1 = 0.01 * 0.01
	const c2 = 0.03 * 0.03
	ssim := ((2*meanA*meanB + c1) * (2*cov + c2)) / ((meanA*meanA + meanB*meanB + c1) * (varA + varB + c2))
	if math.IsNaN(ssim) || math.IsInf(ssim, 0) {
		ssim = 0
	}
	loss := 1 - ssim
	if loss < 0 {
		loss = 0
	}
	if loss > 1 {
		loss = 1
	}
	return NewTensorF32([]int{1}, []float32{float32(loss)}), nil
}

func scalarAddTensor(inputs ...*Tensor) (*Tensor, error) {
	sum := float32(0)
	for i, input := range inputs {
		if input == nil || len(input.F32) != 1 {
			return nil, fmt.Errorf("scalar_add input %d must be a scalar tensor", i)
		}
		sum += input.F32[0]
	}
	return NewTensorF32([]int{1}, []float32{sum}), nil
}

func rateDistortionLossTensor(distortion, rate *Tensor, lambda float64) (*Tensor, error) {
	if distortion == nil || rate == nil || len(distortion.F32) != 1 || len(rate.F32) != 1 {
		return nil, fmt.Errorf("rate_distortion_loss expects scalar distortion and rate")
	}
	if math.IsNaN(lambda) || math.IsInf(lambda, 0) || lambda < 0 {
		return nil, fmt.Errorf("rate_distortion_loss lambda must be finite and non-negative")
	}
	loss := float64(distortion.F32[0]) + lambda*float64(rate.F32[0])
	return NewTensorF32([]int{1}, []float32{float32(loss)}), nil
}

func probabilityForCode(logits *Tensor, codeShape []int, offset, idx, levels int, attrs map[string]string) float64 {
	if logits == nil || len(logits.F32) == 0 {
		return 1 / float64(levels)
	}
	if attrs != nil && attrs["logits_layout"] == "nchw_alphabet" {
		if p, ok := probabilityForNCHWAlphabet(logits, codeShape, offset, idx, levels); ok {
			return p
		}
	}
	if len(logits.F32) == levels {
		return softmaxProbability(logits.F32, 0, levels, idx)
	}
	if len(logits.F32) >= (offset+1)*levels {
		return softmaxProbability(logits.F32, offset*levels, levels, idx)
	}
	p := 1 / (1 + math.Exp(-float64(logits.F32[offset%len(logits.F32)])))
	if idx == 0 {
		return 1 - p
	}
	return p / float64(maxInt(levels-1, 1))
}

func probabilityForBit(logits *Tensor, codeShape []int, offset, bit, bitValue, bitWidth int) float64 {
	if logits == nil || len(logits.F32) == 0 {
		return 0.5
	}
	if p, ok := probabilityForNCHWBitPair(logits, codeShape, offset, bit, bitValue, bitWidth); ok {
		return p
	}
	if len(logits.F32) == bitWidth*2 {
		return softmaxProbability(logits.F32, bit*2, 2, bitValue)
	}
	base := (offset*bitWidth + bit) * 2
	if len(logits.F32) >= base+2 {
		return softmaxProbability(logits.F32, base, 2, bitValue)
	}
	p := 1 / (1 + math.Exp(-float64(logits.F32[(offset*bitWidth+bit)%len(logits.F32)])))
	if bitValue == 1 {
		return p
	}
	return 1 - p
}

func probabilityForNCHWBitPair(logits *Tensor, codeShape []int, offset, bit, bitValue, bitWidth int) (float64, bool) {
	if len(codeShape) != 4 || len(logits.Shape) != 4 {
		return 0, false
	}
	n, c, h, w := unpackOffset4(codeShape, offset)
	if logits.Shape[0] != codeShape[0] || logits.Shape[2] != codeShape[2] || logits.Shape[3] != codeShape[3] {
		return 0, false
	}
	ch := c*bitWidth*2 + bit*2
	if logits.Shape[1] < ch+2 {
		return 0, false
	}
	base0 := offset4(logits.Shape, n, ch, h, w)
	base1 := offset4(logits.Shape, n, ch+1, h, w)
	a := float64(logits.F32[base0])
	b := float64(logits.F32[base1])
	maxV := math.Max(a, b)
	pa := math.Exp(a - maxV)
	pb := math.Exp(b - maxV)
	sum := pa + pb
	if sum == 0 {
		return 0.5, true
	}
	if bitValue == 0 {
		return pa / sum, true
	}
	return pb / sum, true
}

func probabilityForNCHWAlphabet(logits *Tensor, codeShape []int, offset, idx, levels int) (float64, bool) {
	if len(codeShape) != 4 || len(logits.Shape) != 4 {
		return 0, false
	}
	n, c, h, w := unpackOffset4(codeShape, offset)
	if logits.Shape[0] != codeShape[0] || logits.Shape[2] != codeShape[2] || logits.Shape[3] != codeShape[3] {
		return 0, false
	}
	if logits.Shape[1] < (c+1)*levels {
		return 0, false
	}
	baseChannel := c * levels
	maxV := float64(logits.F32[offset4(logits.Shape, n, baseChannel, h, w)])
	for i := 1; i < levels; i++ {
		v := float64(logits.F32[offset4(logits.Shape, n, baseChannel+i, h, w)])
		if v > maxV {
			maxV = v
		}
	}
	sum := 0.0
	target := 0.0
	for i := 0; i < levels; i++ {
		v := math.Exp(float64(logits.F32[offset4(logits.Shape, n, baseChannel+i, h, w)]) - maxV)
		if i == idx {
			target = v
		}
		sum += v
	}
	if sum == 0 {
		return 1 / float64(levels), true
	}
	return target / sum, true
}

func logNormalNormCrossEntropyTensor(codes, params *Tensor, attrs map[string]string) (*Tensor, error) {
	if params == nil {
		return nil, fmt.Errorf("cross_entropy_factorized log_normal mode expects norm params")
	}
	if len(codes.Shape) != 3 || len(params.Shape) != 4 {
		return nil, fmt.Errorf("cross_entropy_factorized log_normal expects codes NHW and params N2HW")
	}
	if params.Shape[0] != codes.Shape[0] || params.Shape[1] < 2 || params.Shape[2] != codes.Shape[1] || params.Shape[3] != codes.Shape[2] {
		return nil, fmt.Errorf("cross_entropy_factorized norm param shape %v does not match codes %v", params.Shape, codes.Shape)
	}
	total := 0.0
	for i, raw := range codes.F32 {
		n, y, x := unpackOffset3(codes.Shape, i)
		mu := float64(params.F32[offset4(params.Shape, n, 0, y, x)])
		sigma := normSigma(float64(params.F32[offset4(params.Shape, n, 1, y, x)]), attrs)
		if math.IsNaN(mu) || math.IsInf(mu, 0) || math.IsNaN(sigma) || math.IsInf(sigma, 0) || sigma <= 0 {
			return nil, fmt.Errorf("cross_entropy_factorized invalid norm params at offset %d", i)
		}
		sym := clampInt(int(math.Round(float64(raw))), 0, 255)
		lo, hi := qNormLogBounds(sym)
		p := normalCDF((hi-mu)/sigma) - normalCDF((lo-mu)/sigma)
		if p < 1e-12 {
			p = 1e-12
		}
		total += -math.Log2(p)
	}
	return NewTensorF32([]int{1}, []float32{float32(total)}), nil
}

func normSigma(raw float64, attrs map[string]string) float64 {
	if attrs == nil {
		return raw
	}
	switch attrs["sigma_parameter"] {
	case "softplus":
		if raw > 32 {
			return raw + 1e-6
		}
		return math.Log1p(math.Exp(raw)) + 1e-6
	case "exp":
		return math.Exp(raw)
	default:
		return raw
	}
}

func factorizationAttr(attrs map[string]string) string {
	if attrs == nil {
		return "categorical"
	}
	switch attrs["factorization"] {
	case "bit-plane", "bitplane", "bit_plane":
		return "bit-plane"
	default:
		return "categorical"
	}
}

func qNormLogBounds(sym int) (float64, float64) {
	span := qNormLogMax - qNormLogMin
	switch sym {
	case 0:
		return math.Inf(-1), qNormLogMin + (0.5/255)*span
	case 255:
		return qNormLogMin + (254.5/255)*span, math.Inf(1)
	default:
		lower := (float64(sym) - 0.5) / 255
		upper := (float64(sym) + 0.5) / 255
		return qNormLogMin + lower*span, qNormLogMin + upper*span
	}
}

func normalCDF(x float64) float64 {
	switch {
	case math.IsInf(x, -1):
		return 0
	case math.IsInf(x, 1):
		return 1
	default:
		return 0.5 * (1 + math.Erf(x/math.Sqrt2))
	}
}

func unpackOffset4(shape []int, offset int) (int, int, int, int) {
	w := offset % shape[3]
	offset /= shape[3]
	h := offset % shape[2]
	offset /= shape[2]
	c := offset % shape[1]
	n := offset / shape[1]
	return n, c, h, w
}

func unpackOffset3(shape []int, offset int) (int, int, int) {
	w := offset % shape[2]
	offset /= shape[2]
	h := offset % shape[1]
	n := offset / shape[1]
	return n, h, w
}

func softmaxProbability(values []float32, base, count, idx int) float64 {
	maxV := float64(values[base])
	for i := 1; i < count; i++ {
		if v := float64(values[base+i]); v > maxV {
			maxV = v
		}
	}
	sum := 0.0
	target := 0.0
	for i := 0; i < count; i++ {
		v := math.Exp(float64(values[base+i]) - maxV)
		if i == idx {
			target = v
		}
		sum += v
	}
	if sum == 0 {
		return 1 / float64(count)
	}
	return target / sum
}

func bitsForQTensor(t *Tensor) int {
	if t == nil {
		return 0
	}
	switch t.DType {
	case "q2":
		return 2
	case "q4":
		return 4
	case "q8", "q_norm":
		return 8
	default:
		return 0
	}
}

func attrInt(attrs map[string]string, key string, fallback int) int {
	if attrs == nil {
		return fallback
	}
	raw := attrs[key]
	if raw == "" {
		return fallback
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return n
}

func attrFloat(attrs map[string]string, key string, fallback float64) float64 {
	if attrs == nil {
		return fallback
	}
	raw := attrs[key]
	if raw == "" {
		return fallback
	}
	n, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return fallback
	}
	return n
}

func offset4(shape []int, n, c, h, w int) int {
	return ((n*shape[1]+c)*shape[2]+h)*shape[3] + w
}

func betaValue(beta *Tensor, c int) float32 {
	if beta == nil || len(beta.F32) == 0 {
		return 1
	}
	if len(beta.F32) == 1 {
		return maxFloat32(beta.F32[0], 1e-6)
	}
	return maxFloat32(beta.F32[c%len(beta.F32)], 1e-6)
}

func gammaValue(gamma *Tensor, c, j, channels int) float32 {
	if gamma == nil || len(gamma.F32) == 0 {
		return 0
	}
	if len(gamma.Shape) == 2 && gamma.Shape[0] == channels && gamma.Shape[1] == channels {
		return gamma.F32[c*channels+j]
	}
	if len(gamma.F32) == channels {
		if c == j {
			return gamma.F32[c]
		}
		return 0
	}
	return gamma.F32[(c*channels+j)%len(gamma.F32)]
}

func quantizeQNorm(norm float32) byte {
	if norm <= 0 || math.IsNaN(float64(norm)) {
		return 0
	}
	if math.IsInf(float64(norm), 1) {
		return 255
	}
	t := (math.Log(float64(norm)) - qNormLogMin) / (qNormLogMax - qNormLogMin)
	if t <= 0 {
		return 0
	}
	if t >= 1 {
		return 255
	}
	return byte(math.Round(t * 255))
}

func dequantizeQNorm(encoded byte) float32 {
	t := float64(encoded) / 255
	return float32(math.Exp(qNormLogMin + t*(qNormLogMax-qNormLogMin)))
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
