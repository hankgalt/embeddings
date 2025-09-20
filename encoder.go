package embeddings

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/comfforts/logger"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/hankgalt/embeddings/internal/transformer/onnx"
	"github.com/hankgalt/embeddings/pkg/domain"
)

type SentenceTransformer interface {
	Encode(ctx context.Context, texts []string) ([][]float32, error)
	Close(ctx context.Context) error
}

type onnxSentenceTransformer struct {
	encoder *onnx.Encoder
}

func NewONNXSentenceTransformer(ctx context.Context, cfg domain.ONNXEncoderConfig) (*onnxSentenceTransformer, error) {
	l, err := logger.LoggerFromContext(ctx)
	if err != nil {
		l = logger.GetSlogLogger()
	}

	if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	} else {
		l.Error("NewONNXSentenceTransformer - missing path to onnxruntime")
		return nil, errors.New("missing path to onnxruntime")
	}

	tokenizerpath := fmt.Sprintf("%s/tokenizer.json", cfg.ModelPath)
	modelPath := fmt.Sprintf("%s/model.onnx", cfg.ModelPath)

	tok, err := onnx.NewHFTokenizerFromLocal(tokenizerpath)
	if err != nil {
		l.Error("NewONNXSentenceTransformer - error loading tokenizer", "error", err.Error())
		return nil, err
	}

	enc, err := onnx.NewEncoder(domain.ONNXEncoderConfig{
		ModelPath:     modelPath,
		InputNameIDs:  cfg.InputNameIDs,
		InputNameMask: cfg.InputNameMask,
		OutputName:    cfg.OutputName,
		MaxSeqLen:     cfg.MaxSeqLen,
	}, tok)
	if err != nil {
		l.Error("NewONNXSentenceTransformer - error loading encoder", "error", err.Error())
		return nil, err
	}

	return &onnxSentenceTransformer{
		encoder: enc,
	}, nil
}

func (o *onnxSentenceTransformer) Encode(ctx context.Context, texts []string) ([][]float32, error) {
	l, err := logger.LoggerFromContext(ctx)
	if err != nil {
		l = logger.GetSlogLogger()
	}

	if o.encoder == nil {
		l.Error("onnxSentenceTransformer:Encode - nil encoder")
		return nil, errors.New("nil encoder")
	}

	return o.encoder.Encode(ctx, texts)
}

func (o *onnxSentenceTransformer) Close(ctx context.Context) error {
	if o.encoder != nil {
		return o.encoder.Close()
	}
	return nil
}
