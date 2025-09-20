# ONNX Runtime version and architecture
# archs: osx-arm64/osx-universal2/osx-x86_64/linux-x64/linux-aarch-64/linux-x64-gpu
# See available versions & supported architectures at: https://github.com/microsoft/onnxruntime/releases
ORT_VER := 1.22.0
ORT_ARCH := osx-arm64

# Set up ONNX Runtime for macOS ARM64
setup-ort:
	@echo "setting up onnx runtime..."
	curl -fsSL -o ./runtime/onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VER)/onnxruntime-$(ORT_ARCH)-$(ORT_VER).tgz \
	&& mkdir -p ./runtime/onnxruntime \
	&& tar -xzf ./runtime/onnxruntime.tgz -C ./runtime/onnxruntime --strip-components=1 \
	&& rm -f ./runtime/onnxruntime.tgz
	@echo "done setting up onnxruntime."