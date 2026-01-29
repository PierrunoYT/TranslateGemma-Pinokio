module.exports = {
  run: [
    // Install required packages for TranslateGemma
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "pip install -r requirements.txt"
      }
    },
    // Install PyTorch with CUDA support after other requirements
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: false,
          flashattn: false,
          triton: false
        }
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Before starting, you need to:<br>1. Create a Hugging Face account at <a href='https://huggingface.co' target='_blank'>huggingface.co</a><br>2. Accept the TranslateGemma model license at <a href='https://huggingface.co/google/translategemma-12b-it' target='_blank'>huggingface.co/google/translategemma-12b-it</a><br>3. Create a Read access token in Settings → Access Tokens<br><br>The model will download on first use (4B: ~8GB, 12B: ~24GB, 27B: ~54GB)."
      }
    }
  ]
}
