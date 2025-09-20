package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/comfforts/logger"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/hankgalt/embeddings/pkg/domain"
)

// Encoder wraps an ONNX model session and tokenizer to produce sentence embeddings.
type Encoder struct {
	cfg        domain.ONNXEncoderConfig
	tok        *HFTokenizer
	sess       *ort.DynamicAdvancedSession
	pooledOut  bool // true if output rank == 2
	hiddenSize int  // H dimension
}

// NewEncoder loads an ONNX model and prepares it for inference.
func NewEncoder(cfg domain.ONNXEncoderConfig, tok *HFTokenizer) (*Encoder, error) {
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("missing ModelPath")
	}
	if tok == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	if cfg.MaxSeqLen <= 0 || cfg.MaxSeqLen > 512 {
		cfg.MaxSeqLen = 256
	}

	// Initialize global ORT env once per process (safe to call multiple times).
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("error initializing ORT env: %w", err)
	}

	// Discover model's input/output shape/dtype validation & to pre-size output tensors.
	infosIn, infosOut, err := ort.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("error getting model input/output info: %w", err)
	}
	modelHasInputConfig := 0
	for _, i := range infosIn {
		if i.Name == cfg.InputNameIDs || i.Name == cfg.InputNameMask {
			modelHasInputConfig++
		}
	}

	// verify configured inputs exist in model
	if modelHasInputConfig < 2 {
		return nil, fmt.Errorf("configured inputs %q and %q not found in model",
			cfg.InputNameIDs, cfg.InputNameMask)
	}

	var outInfo *ort.InputOutputInfo
	for i := range infosOut {
		if infosOut[i].Name == cfg.OutputName {
			outInfo = &infosOut[i]
			break
		}
	}
	// verify configured output exists in model
	if outInfo == nil {
		return nil, fmt.Errorf("configured output %q not found in model", cfg.OutputName)
	}

	outDims := outInfo.Dimensions // Shape == []int64
	rank := len(outDims)
	var H int

	switch rank {
	case 2: // [B, H]
		if len(outDims) < 2 || outDims[1] <= 0 {
			return nil, fmt.Errorf("can't resolve H from dims %v", outDims)
		}
		H = int(outDims[1])
	case 3: // [B, T, H]
		if len(outDims) < 3 || outDims[2] <= 0 {
			return nil, fmt.Errorf("can't resolve H from dims %v", outDims)
		}
		H = int(outDims[2])
	default:
		return nil, fmt.Errorf("unexpected output rank %d (dims=%v), want 2 or 3", rank, outDims)
	}

	sess, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{cfg.InputNameIDs, cfg.InputNameMask},
		[]string{cfg.OutputName},
		nil, // no special SessionOptions
	)
	if err != nil {
		return nil, fmt.Errorf("NewDynamicAdvancedSession: %w", err)
	}

	return &Encoder{
		cfg:        cfg,
		tok:        tok,
		sess:       sess,
		pooledOut:  rank == 2,
		hiddenSize: H,
	}, nil
}

// Encode processes a batch of texts and returns their embeddings.
func (e *Encoder) Encode(ctx context.Context, texts []string) ([][]float32, error) {
	l, err := logger.LoggerFromContext(ctx)
	if err != nil {
		l = logger.GetSlogLogger()
	}
	if e.sess == nil {
		l.Error("onnx.Encoder:Encode - nil session")
		return nil, errors.New("nil session")
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	// 1) Tokenize
	inputIDs, attnMask, seqLen, err := e.tok.EncodeBatch(texts, e.cfg.MaxSeqLen)
	if err != nil {
		return nil, err
	}
	B, T, H := len(texts), seqLen, e.hiddenSize

	// 2) Inputs [B,T] int64
	shape2 := ort.NewShape(int64(B), int64(T))
	idTensor, err := ort.NewTensor(shape2, flattenInt32ToInt64(inputIDs))
	if err != nil {
		l.Error("onnx.Encoder:Encode - error creating input_ids tensor: %w", err)
		return nil, err
	}
	defer idTensor.Destroy()

	maskTensor, err := ort.NewTensor(shape2, flattenInt32ToInt64(attnMask))
	if err != nil {
		l.Error("onnx.Encoder:Encode - error creating attention_mask tensor: %w", err)
		return nil, err
	}
	defer maskTensor.Destroy()

	// 3) Output tensor(s) float32
	var outTensor *ort.Tensor[float32]
	if e.pooledOut {
		outTensor, err = ort.NewEmptyTensor[float32](ort.NewShape(int64(B), int64(H))) // [B,H]
	} else {
		outTensor, err = ort.NewEmptyTensor[float32](ort.NewShape(int64(B), int64(T), int64(H))) // [B,T,H]
	}
	if err != nil {
		l.Error("onnx.Encoder:Encode - error creating output tensor: %w", err)
		return nil, err
	}
	defer outTensor.Destroy()

	// 4) Run
	if err := e.sess.Run([]ort.Value{idTensor, maskTensor}, []ort.Value{outTensor}); err != nil {
		l.Error("onnx.Encoder:Encode - error during ORT Run: %w", err)
		return nil, err
	}

	// 5) Collect (+ optional pooling)
	data := outTensor.GetData() // []float32

	if e.pooledOut {
		embs := make([][]float32, B)
		for b := 0; b < B; b++ {
			row := make([]float32, H)
			copy(row, data[b*H:(b+1)*H])
			if !e.cfg.SkipNormalize {
				L2Normalize(row)
			}
			embs[b] = row
		}
		return embs, nil
	}

	embs := make([][]float32, B)
	strideBT := T * H
	for b := 0; b < B; b++ {
		start := b * strideBT
		pooled := MeanPool(data[start:start+strideBT], attnMask[b], H)
		if !e.cfg.SkipNormalize {
			L2Normalize(pooled)
		}
		embs[b] = pooled
	}
	return embs, nil
}

// Close releases resources associated with the Encoder.
func (e *Encoder) Close(ctx context.Context) error {
	l, err := logger.LoggerFromContext(ctx)
	if err != nil {
		l = logger.GetSlogLogger()
	}

	if e.sess != nil {
		// Destroy the ONNX session
		err = e.sess.Destroy()
		if err != nil {
			l.Error("onnx.Encoder:Close - error destroying session", "error", err.Error())
		}
		e.sess = nil
	}
	// Destroy the process ORT environment if encoder/process owns it
	if !e.cfg.GlobalRuntime {
		l.Info("onnx.Encoder:Close - ORT environment is not global, destroying process ORT environment")
		if eErr := ort.DestroyEnvironment(); eErr != nil {
			if err != nil {
				l.Error("onnx.Encoder:Close - error destroying process ORT environment", "error", eErr.Error())
				return errors.Join(err, eErr)
			}
			return eErr
		}
	}
	return nil
}

// flattenInt32ToInt64 flattens a 2D slice of int32 to a 1D slice of int64.
func flattenInt32ToInt64(xs [][]int32) []int64 {
	if len(xs) == 0 {
		return nil
	}
	out := make([]int64, 0, len(xs)*len(xs[0]))
	for _, row := range xs {
		for _, v := range row {
			out = append(out, int64(v))
		}
	}
	return out
}

// L2Normalize normalizes a vector in place to unit length.
func L2Normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	n := float32(1.0 / math.Sqrt(s))
	for i := range v {
		v[i] *= n
	}
}
