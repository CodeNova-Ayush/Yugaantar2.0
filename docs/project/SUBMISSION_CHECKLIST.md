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
- The supplied event rules say not to submit a Dockerfile. The inherited official
  recipe is preserved locally but removed from Git tracking and ignored. Use the
  organizer-provided image for external Linux validation; no custom Dockerfile is
  included in the prepared submission.

Publishing and sending the repository URL are external actions; they are not
performed by local training or by a successful validator run.

## Prepared local state

The final implementation is committed locally. Git LFS integrity (`git lfs fsck`)
passed; the prepared tree including LFS weights is approximately 40.8 MB. The
committed local validator passed, reached the supplied-map exit, and reported only
the two macOS CPU/RAM measurement warnings. Read `verification.json` for exact
source commit and checks. No GitHub push or organizer submission has been performed.

The optional official Linux image build was stopped during large dependency downloads.
Linux RAM/CPU enforcement and CUDA behavior remain unverified. The inherited recipe
is still available locally if you choose to resume the build.
