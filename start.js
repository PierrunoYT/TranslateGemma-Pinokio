module.exports = {
  daemon: true,
  run: [
    // Launch TranslateGemma Gradio server
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python app.py"
        ],
        on: [{
          // Capture server URL (Gepeto / mochi pattern); app uses server_name 127.0.0.1 in app.py
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    // Set the local URL variable for the "Open Web UI" button
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    },
    {
      method: "notify",
      params: {
        html: "TranslateGemma is running! Click 'Open Web UI' to start translating. Enter your Hugging Face token in the UI to download models. CUDA GPU recommended for best performance."
      }
    }
  ]
}
