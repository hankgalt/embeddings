package domain

type ONNXEncoderConfig struct {
	// directory containing model.onnx and tokenizer.json
	ModelPath string
	// e.g. "input_ids"
	InputNameIDs string
	// e.g. "attention_mask"
	InputNameMask string
	// e.g. "last_hidden_state" or "sentence_embedding"
	OutputName string
	// e.g. 256
	MaxSeqLen int
	// if true, skip L2 normalization
	SkipNormalize bool
	// if true, this instance will skip shutting down the ONNX runtime on close
	// This is useful if multiple encoders are used in the same process.
	// In that case, the application should manage the ONNX runtime lifecycle.
	// Default: false (each encoder manages its own runtime lifecycle & shuts runtime down on Close)
	GlobalRuntime bool
}
