package onnx

// MeanPool computes masked mean over tokens.
// hidden: [T*H] flattened row-major; mask: [T] (0/1)
func MeanPool(hidden []float32, mask []int32, H int) []float32 {
	T := len(mask)
	out := make([]float32, H)
	var denom float32 = 0
	for t := 0; t < T; t++ {
		if mask[t] == 0 {
			continue
		}
		denom += 1
		base := t * H
		for h := 0; h < H; h++ {
			out[h] += hidden[base+h]
		}
	}
	if denom == 0 {
		return out
	} // all zeros fallback
	scale := 1.0 / denom
	for h := 0; h < H; h++ {
		out[h] *= float32(scale)
	}
	return out
}
