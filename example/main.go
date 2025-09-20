package main

import (
	"context"
	"log"

	"github.com/comfforts/logger"
	emb "github.com/hankgalt/embeddings"
	"github.com/hankgalt/embeddings/pkg/domain"
)

func main() {
	l := logger.GetSlogLogger()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx = logger.WithLogger(ctx, l)

	encoder, err := emb.NewONNXSentenceTransformer(ctx, domain.ONNXEncoderConfig{
		ModelPath:     "../models/sentence-t5-base-onnx",
		InputNameIDs:  "input_ids",
		InputNameMask: "attention_mask",
		OutputName:    "sentence_embedding",
		MaxSeqLen:     256,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer encoder.Close(ctx)

	// Encode a batch of texts.
	vecs, err := encoder.Encode(ctx,
		[]string{"Query: golang embeddings", "Passage: computing sentence vectors in Go"})
	if err != nil {
		log.Fatal(err)
	}

	l.Info("got embeddings", "embeddings-dimensions", len(vecs[0]), "num-embeddings", len(vecs))
}
