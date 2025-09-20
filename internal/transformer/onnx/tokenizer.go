package onnx

import (
	"fmt"

	tk "github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// HFTokenizer wraps a HuggingFace tokenizer for ONNX models.
type HFTokenizer struct {
	tok                 *tk.Tokenizer
	clsID, sepID, padID int
}

// NewHFTokenizerFromLocal loads a tokenizer from a local tokenizer.json file.
func NewHFTokenizerFromLocal(path string) (*HFTokenizer, error) {
	tok, err := pretrained.FromFile(path) // loads tokenizer.json
	if err != nil {
		return nil, err
	}

	// Common special IDs; adjust if your model differs.
	clsID := idOrDefault(tok, "[CLS]", 101)
	sepID := idOrDefault(tok, "[SEP]", 102)
	padID := idOrDefault(tok, "[PAD]", 0)
	return &HFTokenizer{tok: tok, clsID: clsID, sepID: sepID, padID: padID}, nil
}

// idOrDefault returns the token ID for a given token or a default if not found.
func idOrDefault(t *tk.Tokenizer, token string, def int) int {
	id, ok := t.TokenToId(token)
	if !ok {
		return def
	}
	return int(id)
}

// VocabSize returns the size of the tokenizer's vocabulary.
func (h *HFTokenizer) VocabSize() (int, error) {
	if h.tok == nil {
		return 0, fmt.Errorf("tokenizer nil")
	}
	return int(h.tok.GetVocabSize(false)), nil
}

// EncodeBatch returns input_ids and attention_mask shaped [B][T].
func (h *HFTokenizer) EncodeBatch(texts []string, maxLen int) ([][]int32, [][]int32, int, error) {
	if h.tok == nil {
		return nil, nil, 0, fmt.Errorf("tokenizer nil")
	}
	if maxLen <= 0 {
		maxLen = 512
	}
	if len(texts) == 0 {
		return [][]int32{}, [][]int32{}, 0, nil
	}

	// 1) Encode each text individually & find longest
	longest := 0
	encs := make([]*tk.Encoding, 0, len(texts))
	for _, s := range texts {
		enc, err := h.tok.EncodeSingle(s, true)
		if err != nil {
			return nil, nil, 0, err
		}
		encs = append(encs, enc)
		if l := len(enc.Ids); l > longest {
			longest = l
		}
	}

	// 2) Determine final T (sequence length) with truncation cap
	T := longest
	if T > maxLen {
		T = maxLen
	}
	B := len(encs)

	// 3) Materialize input_ids and attention_mask (right-pad with [PAD])
	ids := make([][]int32, B)
	mask := make([][]int32, B)
	for i, e := range encs {
		ids[i] = make([]int32, T)
		mask[i] = make([]int32, T)

		// Copy (and truncate if needed)
		L := len(e.Ids)
		if L > T {
			L = T
		}
		for t := 0; t < L; t++ {
			ids[i][t] = int32(e.Ids[t])
			if e.Ids[t] != h.padID {
				mask[i][t] = 1
			}
		}
		// Remaining positions are already zero-initialized:
		// ids -> PAD id, mask -> 0
		for t := L; t < T; t++ {
			ids[i][t] = int32(h.padID)
			mask[i][t] = 0
		}
	}
	return ids, mask, T, nil
}
