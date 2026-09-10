# 🌍 TranslateGemma - Multilingual AI Translation

Google's open-source translation model supporting 55+ languages with image text extraction capabilities.

![TranslateGemma](https://img.shields.io/badge/Model-TranslateGemma-blue) ![Gradio](https://img.shields.io/badge/Gradio-5.50-orange) ![Python](https://img.shields.io/badge/Python-3.10%2B-green)

## ✨ Features

- 🔄 **Text Translation** - Translate text across 55+ languages
- 🖼️ **Image Translation** - Extract and translate text from images
- ⚡ **Multiple Model Sizes** - Choose between 4B, 12B, and 27B parameter models
- 🎨 **Modern UI** - Clean Gradio interface with easy-to-use controls
- 🚀 **GPU Accelerated** - CUDA support for faster inference

## 📋 Requirements

### Hardware
- **GPU**: CUDA-compatible GPU recommended (optional, CPU supported)
- **VRAM** (bfloat16 weights):
  - 4B model: ~8GB
  - 12B model: ~24GB
  - 27B model: ~54GB
- **RAM**: 16GB+ recommended

### Software
- Python 3.10 or higher (required by Gradio 5)
- PyTorch with CUDA support (installed automatically)
- Hugging Face account with accepted TranslateGemma license

## 🚀 Installation

### Via Pinokio (Recommended)

1. Click **Install** in Pinokio
2. Wait for dependencies to install
3. Click **Start** to launch the app

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/PierrunoYT/TranslateGemma-Pinokio.git
cd TranslateGemma-Pinokio/app
```

2. Create virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

For a CPU-only install, swap `cu128` for `cpu`.

4. Run the app:
```bash
python app.py
```

5. Open browser at `http://localhost:7860`

## 🔑 Setup Hugging Face Authentication

Before using TranslateGemma, you need to authenticate with Hugging Face:

1. **Create Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Accept License**: Visit [google/translategemma-12b-it](https://huggingface.co/google/translategemma-12b-it) and accept the license
3. **Create Token**: Go to Settings → Access Tokens → Create new token (Read access)
4. **Set Token in UI**: Enter your token in the "🔑 Hugging Face Authentication" section

## 📖 Usage

### Text Translation

1. **Load Model**: Select model size (4B/12B/27B) and click "🚀 Load Model"
2. **Enter Text**: Type or paste text in the input box
3. **Select Languages**: Choose source and target languages
4. **Translate**: Click "🔄 Translate"

### Image Translation

1. **Upload Image**: Click to upload an image containing text
2. **Select Languages**: Choose source and target languages
3. **Extract & Translate**: Click "🔄 Extract & Translate"

## 🌐 Supported Languages

TranslateGemma supports 55+ languages including:

| Language | Code | Language | Code | Language | Code |
|----------|------|----------|------|----------|------|
| English | en | Spanish | es | French | fr |
| German | de | Chinese | zh | Japanese | ja |
| Korean | ko | Arabic | ar | Hindi | hi |
| Portuguese | pt | Russian | ru | Italian | it |
| Dutch | nl | Polish | pl | Turkish | tr |
| Vietnamese | vi | Thai | th | Indonesian | id |
| Hebrew | he | Greek | el | Czech | cs |
| Swedish | sv | Danish | da | Finnish | fi |

And many more with regional variants (en-US, es-MX, pt-BR, etc.)!

## 🎯 Model Sizes

| Model | Parameters | Use Case | Download Size |
|-------|-----------|----------|---------------|
| 4B | 4 Billion | Mobile, Edge devices | ~8GB |
| 12B | 12 Billion | Laptops, Desktops (Recommended) | ~24GB |
| 27B | 27 Billion | Cloud, High-end GPUs | ~54GB |

**Recommendation**: Start with the 12B model for the best balance of quality and performance.

## 🔧 Configuration

### Model Settings

- **Model Size**: Choose between 4B, 12B, or 27B
- **Max Output Tokens**: Control translation length (50-500 tokens)
- **Device**: Automatically uses CUDA if available, falls back to CPU

### Server Settings

The app listens on `127.0.0.1:7860` by default. Override either with an
environment variable before starting it:

- `GRADIO_SERVER_NAME` - host to bind (e.g. `0.0.0.0` to expose on the LAN)
- `GRADIO_SERVER_PORT` - port to bind, if 7860 is already taken

### Performance Tips

- Use GPU for best performance
- Start with 12B model for balanced performance
- Reduce max tokens for faster inference
- Close other GPU applications to free VRAM

## 📊 Performance

According to Google's benchmarks:

- **26% better accuracy** than base Gemma models
- **30% improvement** on rare language pairs (e.g., English-Icelandic)
- **12B model** achieves lower error rates than 27B baseline
- **Multimodal support** without additional fine-tuning

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses Google's TranslateGemma models which are licensed under the [Gemma License](https://ai.google.dev/gemma/terms).

## 🔗 Resources

- **Official Models**: [Hugging Face Collection](https://huggingface.co/collections/google/translategemma)
- **Technical Paper**: [TranslateGemma Technical Report](https://arxiv.org/pdf/2601.09012)
- **Google Blog**: [Introducing TranslateGemma](https://blog.google/technology/ai/translategemma/)
- **Kaggle**: [TranslateGemma Models](https://www.kaggle.com/models/google/translategemma/)
- **Vertex AI**: [Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/translategemma)

## ⚠️ Troubleshooting

### Common Issues

**Error: "401 Unauthorized"**
- Make sure you've accepted the license at huggingface.co/google/translategemma-12b-it
- Enter a valid Hugging Face token in the UI

**Error: "CUDA out of memory"**
- Try a smaller model size (4B instead of 12B)
- Close other GPU applications
- Reduce max output tokens

**Slow performance**
- Use GPU instead of CPU
- Ensure CUDA is properly installed
- Try a smaller model size

**Model not downloading**
- Check internet connection
- Verify Hugging Face token is valid
- Ensure sufficient disk space (~8-54GB depending on model)

## 💬 Support

For issues and questions:
- Open an issue on GitHub
- Check the [Hugging Face discussions](https://huggingface.co/google/translategemma-12b-it/discussions)

## 🙏 Acknowledgments

- Google Translate Team for TranslateGemma models
- Hugging Face for model hosting and transformers library
- Gradio team for the UI framework

---

Made with ❤️ for multilingual communication
