package domain

type ONNXEncoderConfig struct {
	ModelPath     string
	InputNameIDs  string // e.g. "input_ids"
	InputNameMask string // e.g. "attention_mask"
	OutputName    string // e.g. "last_hidden_state" or "sentence_embedding"
	MaxSeqLen     int
	SkipNormalize bool // if true, skip L2 normalization
}
