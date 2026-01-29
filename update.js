module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "notify",
      params: {
        html: "Update complete! The launcher and dependencies have been updated to the latest versions."
      }
    }
  ]
}
