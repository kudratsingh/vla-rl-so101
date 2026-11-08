# Pod launch runbook

Every paid pod is launched in this order. Each launch still needs the owner's explicit go for that
run (CLAUDE.md rule 1). Spend control has three independent layers, each meant to remove the pod
before `DEADLINE = T_req + SPEND_CAP_SECONDS`, where `T_req` is the epoch taken just before the
create call:

| Layer | Where it runs | Fires at | Starts | Fails when |
|---|---|---|---|---|
| Job EXIT trap + soft deadline (`scripts/pod/job_lib.sh`) | in the job | hard - 1200 s | job launch | job's bash is SIGKILLed / OOM, the API is down |
| Hard-cap timer (`scripts/pod/pod_start_cmd.sh`) | in the container | min(first boot + cap - 180 s, `DEADLINE` - 480 s) | container boot | container never boots, restarts without `/workspace`, no key, the API is down |
| Outside watchdog (`scripts/pod/watchdog.py`), two copies | GitHub-hosted runner, and the owner's Mac | `DEADLINE` - 400 s | right after the create call | GitHub run never starts; the Mac sleeps or is off; the API is down for everyone |

The watchdog polls the pod every 60 s and exits 0 as soon as it has verified the pod is gone. A
pod counts as gone after two consecutive reads at least 10 s apart that return 404, status
`TERMINATED`, or a null GraphQL `pod`. An accepted delete is not enough. At `DEADLINE` - 400 s the
watchdog tries, in this order: REST v2 `DELETE https://api.runpod.io/v2/pods/{id}`, then REST v1
`DELETE https://rest.runpod.io/v1/pods/{id}`, then GraphQL `podTerminate`. A GraphQL reply counts
only when it has no `errors`. Each delete has a 15 s total timeout and each read an 8 s one, so a
worst-case cycle takes about 128 s and three cycles fit inside the 400 s lead. If the pod is not
verified gone by `DEADLINE`, the watchdog alerts (macOS notification and bell, or a GitHub
`::error::` annotation, which emails the owner when the run fails). It keeps trying for another
900 s and then exits 2.

## One-time setup (owner)

1. **RunPod API key for the watchdogs.** RunPod's API-key docs list three levels: All, Restricted
   and Read Only. Restricted scopes are documented per Serverless endpoint only, and the docs give
   no pod-only scope (docs.runpod.io/get-started/api-keys, read 2026-11-08). The watchdog has to
   delete pods, so the key needs pod write access. Create a dedicated key, named for example
   `vla-rl-so101-watchdog`, so that it can be revoked on its own without touching any other key.
   If the console shows a pods-only option, use that.
2. **Mac Keychain** (stores the key without putting it in a file or in shell history; the command
   prompts for the key):
   ```zsh
   security add-generic-password -a "$USER" -s vla-rl-so101-runpod -w
   security find-generic-password -s vla-rl-so101-runpod > /dev/null && echo ok   # prints no key
   ```
3. **GitHub repo secret** (the command prompts for the value):
   ```zsh
   gh secret set RUNPOD_API_KEY --repo kudratsingh/vla-rl-so101
   ```
   The repo is public, so workflow logs are public too. The watchdog prints no key, and GitHub masks
   secrets in logs. The log shows the pod id, timestamps, HTTP codes and truncated API error bodies.
4. `.github/workflows/pod-watchdog.yml` has to be on `main` before `gh workflow run` can dispatch
   it, so it cannot be used until PR #28 merges. A GitHub-hosted job runs for at most 360 min, so the
   workflow refuses any deadline more than 5.5 h away. A longer pod needs a chain of runs, which is
   not built yet.
5. The RunPod connector must be connected in the agent's session. It returned ENOTFOUND on
   2026-11-08.

## Launch order (agent, after the owner's go)

Pod name: `vla-rl-so101-<purpose>`, for example `vla-rl-so101-diag`. Both watchdogs pass
`--expect-name-prefix vla-rl-so101` and refuse to delete a pod with any other name, which guards
against a mistyped id on the shared account.

1. **Take `T_req`** just before the create call:
   ```zsh
   CAP=12420; T_REQ=$(date +%s); DEADLINE=$((T_REQ + CAP)); echo "$T_REQ $DEADLINE"
   ```
2. **Create the pod through the RunPod connector**:
   - image `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, Secure Cloud, GPU type from the estimate;
   - entrypoint `["/bin/bash","-c"]`, with the whole of `scripts/pod/pod_start_cmd.sh` as the
     one start-command string;
   - env `SPEND_CAP_SECONDS=$CAP`, `POD_DEADLINE_EPOCH=$DEADLINE` and `PUBLIC_KEY=<owner ssh
     public key>`. Never pass a RunPod or HF key as env, because RunPod injects a pod-scoped
     `RUNPOD_API_KEY` itself;
   - container disk 80 GB, a **pod volume** (`volumeInGb` 20, `volumeMountPath` `/workspace`, not a
     network volume) and port `22/tcp`.

   If the create call returns no id, list pods by name and treat any match as created.
3. **Start both watchdogs at once**, before anything else:
   ```zsh
   gh workflow run pod-watchdog.yml --repo kudratsingh/vla-rl-so101 \
     -f pod_id=<id> -f deadline_epoch=$DEADLINE -f cap_s=$CAP
   gh run list --repo kudratsingh/vla-rl-so101 --workflow pod-watchdog.yml -L 1   # must be in_progress
   bash scripts/pod/watchdog_mac.sh <id> $DEADLINE $CAP
   ```
   Each watchdog warns if it starts more than 180 s after `T_req` or after the pod's `createdAt`.
   With `cap_s` set, each one also tightens its deadline to `createdAt + cap` when that is earlier.
   If the GitHub run is not `in_progress` within ~3 min, the Mac copy is the only outside layer.
   Say so to the owner.
4. **Read the pod back** through the connector. The start command must equal the file, the env must
   hold `SPEND_CAP_SECONDS`, `POD_DEADLINE_EPOCH` and `PUBLIC_KEY`, and the volume must be at
   `/workspace`. Note `createdAt`.
5. **Over SSH**, `cat /root/backstop_status` must read `KEY_OK http=200 timer=armed
   cap_from=creation`. Anything else means: delete the pod through the connector now. The watchdogs
   then see it gone and exit 0.
6. **HF token** to `/root/.hf_token` over SSH (`umask 077`), never as pod env or on a command line.
7. **Launch the job**: `nohup bash scripts/pod/<job>.sh > /workspace/job.log 2>&1 &`.
8. **Approval gates**: post the probe numbers. On the owner's explicit go, `touch
   /workspace/APPROVED`.
9. **End**: both watchdogs log `verified_gone` (exit 0). Check with `gh run view` and
   `runs/watchdog/<id>.jsonl`. Write the run record and spend row.

Deleting a pod early through the connector needs no change to the watchdogs. They have seen the
pod, so they verify it is gone and exit 0. To stop one by hand, do it only after the pod is
verified gone: `gh run cancel <run>` and `pkill -f 'watchdog.py --pod-id <id>'`.

## Limits

- The Mac copy runs under `caffeinate -i -s`. That prevents idle and system sleep only on AC power.
  Closing the lid on battery still sleeps the Mac, and a dead, sleeping or offline laptop runs no
  watchdog. The GitHub copy covers those cases.
- The GitHub copy depends on GitHub starting the run. Queue time is normally seconds, but it is not
  guaranteed, and it does not run at all during a GitHub Actions outage.
- If RunPod's API refuses or drops every call for the whole window, nothing outside RunPod can stop
  the pod. The watchdog alerts, keeps trying for 900 s, then exits 2. After that, only the owner in
  the RunPod console can stop it.
- A 404 means "not found or not accessible to the caller" (REST v2 OpenAPI). If the watchdog never
  saw the pod with its key, it exits 3 and alerts instead of reporting success.
- The billing start and the delete-to-billing-stop latency are not measured. The estimate assumes
  billing starts no earlier than the create call and stops once the pod reads as gone.
