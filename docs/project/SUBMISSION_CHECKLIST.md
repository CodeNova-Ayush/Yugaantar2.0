# Final publication checklist

- Final weights are selected using validation, with hash and metrics in FINAL_RESULTS.json.
- `submission/agent.py` loads the selected checkpoint once, resets state per episode,
  and returns nine official buttons every four ticks. It uses no network or file writes.
- `.gitattributes` tracks weights with Git LFS. Git LFS must be installed on the publishing machine.
- Keep source, final weights, manifests, compact CSV/JSON reports and the two-page brief.
  Full intermediate training-state files are local recovery artifacts and are excluded.
- Run `python validate.py --device cpu` after final weight selection. A local macOS
  pass does not verify Linux RAM limits or CUDA execution. Prefer the organizer's
  official Docker image when available; do not create a custom submission image.
- Inspect `git status`, stage only project files, verify `git lfs ls-files`, commit,
  and push normally to the default branch before the deadline. Never force-push.
- Verify GitHub displays the source and LFS weight pointer, then submit the PUBLIC
  repository URL and the maximum-two-page PDF to the organizers.
- The inherited Dockerfile belongs to the official starter runtime, not a custom
  agent runtime. The pasted event rules say not to submit a Dockerfile; clarify
  whether they require removal of the inherited official file before publication.

Publishing and sending the repository URL are external actions; they are not
performed by local training or by a successful validator run.
