module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Install PyTorch first so requirements (e.g. accelerate) don't pull in a
    // default PyPI torch build that would then have to be replaced
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
          xformers: false,
          flashattn: false,
          triton: false
        }
      }
    },
    // Install required packages for TranslateGemma
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install -r requirements.txt"
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
