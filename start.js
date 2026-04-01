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
          // Monitor for Gradio server URL output
          "event": "/(http:\\/\\/\\S+)/",
          "done": true
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
