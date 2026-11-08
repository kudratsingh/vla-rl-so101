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

At launch the watchdog first confirms that its key sees the pod under a `vla-rl-so101*` name,
re-reading for up to 120 s (log event `confirmed`). If no source shows the pod (wrong id, or not
visible to the key), it alerts and exits 3 at once. If both REST APIs refuse the key (401/403), it
exits 5 at once. It never deletes a pod it has not confirmed. A layer counts as armed only once it
has confirmed (step 3).

A read counts as "gone" only when two of REST v2, REST v1 and GraphQL agree (404, status
`TERMINATED`, or a null GraphQL `pod`). A 404 also means "not accessible to the caller", and one
API can glitch. The watchdog polls every 60 s and exits 0 early only if the pod deleted itself and
every read over at least 120 s shows it gone. At `DEADLINE` - 400 s it tries, in this order: REST
v2 `DELETE https://api.runpod.io/v2/pods/{id}`, then REST v1 `DELETE
https://rest.runpod.io/v1/pods/{id}`, then GraphQL `podTerminate`. A GraphQL reply counts only when
it has no `errors`, and a DELETE 404 is not an acceptance. After an accepted delete that left the
pod in place, every retry sends all three. The pod is verified gone after two consecutive reads at
least 10 s apart; an accepted delete is not enough. Each delete has a 15 s total timeout and each
read an 8 s one, so a worst-case cycle takes about 128 s and three cycles fit inside the 400 s lead.
If the pod is not verified gone by `DEADLINE`, the watchdog alerts (macOS notification and bell, or
a GitHub `::error::` annotation, which emails the owner when the run fails). It keeps trying for
900 s from the later of `DEADLINE` and its own start of termination (at least one full attempt,
also after the Mac wakes from sleep), then exits 2.

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
3. **GitHub environment secret**, not a repo secret. A repo secret can be read by a workflow on
   any branch pushed to the repo; an environment secret whose environment only allows `main`
   cannot. Create the environment before the first dispatch (a run that names a missing
   environment creates it without the branch rule):
   ```zsh
   gh api -X PUT repos/kudratsingh/vla-rl-so101/environments/pod-watchdog \
     -F 'deployment_branch_policy[protected_branches]=false' \
     -F 'deployment_branch_policy[custom_branch_policies]=true'
   gh api -X POST repos/kudratsingh/vla-rl-so101/environments/pod-watchdog/deployment-branch-policies \
     -f name=main -f type=branch
   gh secret set RUNPOD_API_KEY --repo kudratsingh/vla-rl-so101 --env pod-watchdog   # prompts
   gh secret list --repo kudratsingh/vla-rl-so101   # no repo-level RUNPOD_API_KEY should be listed
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
   CAP=13800; T_REQ=$(date +%s); DEADLINE=$((T_REQ + CAP)); echo "$T_REQ $DEADLINE"
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
3. **Start both watchdogs at once**, before anything else (`CAP` is 5940 for the 4090 pod, 13800 or
   12000 for the A100 pod with the paired clip probe (12660 / 10860 without it, `CLIP_GROUPS=0`;
   a pod launched at those smaller caps skips the clip probe in-pod so the lr sweep keeps its
   time; docs/STATUS.md):
   ```zsh
   gh workflow run pod-watchdog.yml --repo kudratsingh/vla-rl-so101 \
     -f pod_id=<id> -f deadline_epoch=$DEADLINE -f cap_s=$CAP
   bash scripts/pod/watchdog_mac.sh <id> $DEADLINE $CAP   # exit 0 only after "confirmed"
   RUN=$(gh run list --repo kudratsingh/vla-rl-so101 --workflow pod-watchdog.yml -L 1 \
     --json databaseId --jq '.[0].databaseId')
   gh run view "$RUN" --repo kudratsingh/vla-rl-so101 --json status,jobs \
     --jq '.status, (.jobs[0].steps[] | select(.name == "Confirm pod visible") | .conclusion)'
   ```
   Both layers are armed only when the Mac launcher printed `watchdog CONFIRMED the pod` (exit 0)
   and the GitHub run is `in_progress` with the "Confirm pod visible" step `success`. Allow ~3 min
   for the GitHub step. If neither confirms, or either reports a wrong id (Mac exit 4, GitHub step
   `failure` after exit 3/4/5), delete the pod through the connector now: the outside layer is not
   watching it. If only one confirms, say so to the owner before going on.
   Each watchdog warns if it starts more than 180 s after `T_req` or after the pod's `createdAt`.
   With `cap_s` set, each one also tightens its deadline to `createdAt` + cap when that is earlier.
4. **Read the pod back** through the connector. The start command must equal the file, the env must
   hold `SPEND_CAP_SECONDS`, `POD_DEADLINE_EPOCH` and `PUBLIC_KEY`, and the volume must be at
   `/workspace`. Note `createdAt`.
5. **Over SSH**, `cat /root/backstop_status` must read `KEY_OK http=200 timer=armed
   cap_from=creation`. Anything else means: delete the pod through the connector now. The watchdogs
   then see it gone and exit 0.
6. **HF token** to `/root/.hf_token` over SSH (`umask 077`), never as pod env or on a command line.
7. **Launch the job**: `nohup bash scripts/pod/<job>.sh > /workspace/job.log 2>&1 &`.
8. **Approval gates**: post the probe numbers. On the owner's explicit go, write THIS pod's
   approval file over SSH, from the repo root on the pod:
   `bash -c '. /root/runpod.env; . scripts/pod/approval_lib.sh; approval_write "owner go <date>: <what>"'`.
   It writes `/workspace/APPROVED.<RUNPOD_POD_ID>`. The gate accepts only that file, and only if
   it was written after the pod's first boot. A stale file from an earlier pod on a reused volume,
   or the old unscoped `/workspace/APPROVED` (`touch` no longer works), is ignored, and the job log
   names it (`scripts/pod/approval_lib.sh`).
9. **End**: both watchdogs log `verified_gone` (exit 0). Check with `gh run view` and
   `runs/watchdog/<id>.jsonl`. Write the run record and spend row.

Deleting a pod early through the connector needs no change to the watchdogs. They have seen the
pod, so they verify it is gone (two sources, for at least 120 s) and exit 0. To stop one by hand, do it only after the pod is
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
- A 404 means "not found or not accessible to the caller" (REST v2 OpenAPI). "Gone" therefore needs
  two sources to agree, and a pod the key never saw gives exit 3 at launch instead of a silent watch.
- If GraphQL answers a missing pod with an `errors` body (its shape for a valid key is not
  verified), "gone" rests on REST v2 and v1 both saying 404. If only one source can answer, the
  watchdog cannot verify the pod gone and ends with exit 2 and an alert, even if the pod is gone.
- Never verified against the real RunPod API: only against the loopback fake in
  `tests/test_pod_watchdog.py`.
- The billing start and the delete-to-billing-stop latency are not measured. The estimate assumes
  billing starts no earlier than the create call and stops once the pod reads as gone.
