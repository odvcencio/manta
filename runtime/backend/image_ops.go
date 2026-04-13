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
	packed := make([]byte, turboquant.PackedSize(channels, bits))
	indices := make([]int, channels)
	vec := make([]float32, channels)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					vec[c] = input.F32[offset4(input.Shape, b, c, y, x)]
				}
				norm := q.QuantizeTo(packed, vec)
				unpackIndices(indices, packed, channels, bits)
				for c := 0; c < channels; c++ {
					coords.F32[offset4(coords.Shape, b, c, y, x)] = float32(indices[c])
				}
				norms.F32[(b*height+y)*width+x] = float32(quantizeQNorm(norm))
			}
		}
	}
	return coords, norms, nil
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
	packed := make([]byte, turboquant.PackedSize(channels, bits))
	indices := make([]int, channels)
	vec := make([]float32, channels)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					indices[c] = clampInt(int(math.Round(float64(coords.F32[offset4(coords.Shape, b, c, y, x)]))), 0, (1<<bits)-1)
				}
				packIndices(packed, indices, bits)
				q.DequantizeTo(vec, packed)
				norm := dequantizeQNorm(byte(clampInt(int(math.Round(float64(norms.F32[(b*height+y)*width+x]))), 0, 255)))
				for c := 0; c < channels; c++ {
					out.F32[offset4(out.Shape, b, c, y, x)] = vec[c] * norm
				}
			}
		}
	}
	return out, nil
}

func crossEntropyFactorizedTensor(codes, logits *Tensor, attrs map[string]string) (*Tensor, error) {
	if codes == nil {
		return nil, fmt.Errorf("cross_entropy_factorized expects codes")
	}
	levels := attrInt(attrs, "levels", 0)
	if levels <= 0 {
		if bits := attrInt(attrs, "bits", bitsForQTensor(codes)); bits > 0 {
			levels = 1 << bits
		} else {
			levels = 256
		}
	}
	total := 0.0
	for i, raw := range codes.F32 {
		idx := clampInt(int(math.Round(float64(raw))), 0, levels-1)
		p := probabilityForCode(logits, i, idx, levels)
		if p < 1e-12 {
			p = 1e-12
		}
		total += -math.Log2(p)
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

func probabilityForCode(logits *Tensor, offset, idx, levels int) float64 {
	if logits == nil || len(logits.F32) == 0 {
		return 1 / float64(levels)
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

func packIndices(dst []byte, indices []int, bitWidth int) {
	for i := range dst {
		dst[i] = 0
	}
	switch bitWidth {
	case 2:
		for i, idx := range indices {
			dst[i/4] |= byte(idx&3) << uint((i%4)*2)
		}
	case 4:
		for i, idx := range indices {
			dst[i/2] |= byte(idx&15) << uint((i%2)*4)
		}
	case 8:
		for i, idx := range indices {
			dst[i] = byte(idx)
		}
	}
}

func unpackIndices(indices []int, src []byte, count, bitWidth int) {
	switch bitWidth {
	case 2:
		for i := 0; i < count; i++ {
			indices[i] = int((src[i/4] >> uint((i%4)*2)) & 3)
		}
	case 4:
		for i := 0; i < count; i++ {
			indices[i] = int((src[i/2] >> uint((i%2)*4)) & 15)
		}
	case 8:
		for i := 0; i < count; i++ {
			indices[i] = int(src[i])
		}
	}
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
