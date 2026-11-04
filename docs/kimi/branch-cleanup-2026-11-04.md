# Temporary branch cleanup - 2026-11-04

Status: planned; deletion results will be recorded after verification.

The user requested removal of unnecessary branches. Each selected branch head
is reachable from a branch that is being retained. No unique commit history is
removed. Default branches, PR64288, its separate synchronization fix, the active
validation branch, IPAT and branches with independent work are retained.

| Repository | Selected branch | Original head | Retained branch containing the head |
| --- | --- | --- | --- |
| ajith-sirra-amd/vllm | `checks/fp8-native-tests-4920807` | `577c08da05d35b23c4dee34b61cf609d77010dbf` | `checks/pr64288-rendered-docs-20261102` |
| ajith-sirra-amd/vllm | `checks/pr64288-asm-sync-review-20261104` | `e2393d702a3736dffdfbfc7eb25abbabe484a1df` | `perf/rocm-fp8-kv-proj-fmha` |
| ajith-sirra-amd/vllm | `checks/pr64288-quickfix-candidate-20261104` | `8a55b8eacf9020a0b421a3c50a927c4a40e61604` | `perf/rocm-fp8-kv-proj-fmha` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/native-concurrency-sweep-20261102` | `d67152dac62f6f19a42a49c72bc2ae98b7747de1` | `checks/pr64288-focused-review-20261104` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/native-fp8-blas-replay-20261102` | `b56b28bb1fc7302f8f35095341165ddea89de4fa` | `checks/native-fp8-binary-scale-20261102` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/native-fp8-channel-balance-20261102` | `fa0e06998fb840ce2212a51c7212bb6373f69abf` | `checks/native-fp8-binary-scale-20261102` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/native-fp8-new-nightly-20261102` | `c047b417fe471c7a3e913544ebd289343f82bd5c` | `checks/pr64288-focused-review-20261104` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/native-fp8-nope-rotation-20261102` | `ef9ebefa7fc46bc33966c3d4fae3802b88eb6b30` | `checks/native-fp8-binary-scale-20261102` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/native-profile-analysis-20261102` | `00dedf3614dc6d5a1a7649c1a9d23e5fb1094199` | `checks/native-profile-analysis-run-20261104` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/native-run-queue-20261102` | `b359e9a3cd5d0ea7b8244c64df5bb6119a71ae6a` | `checks/pr64288-focused-review-20261104` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/native-steady-profile-20261102` | `9da21ee27541876ead88045459ecbf0e74e7c670` | `checks/pr64288-focused-review-20261104` |
| ajith-sirra-amd/InferenceMAX_Rocm_Team | `checks/pr59124-direct-benchmark-20261102` | `190f103aad8911bff3a31a2ee0cbc12ed9070ce9` | `checks/pr64288-focused-review-20261104` |

To restore a deleted name, create `refs/heads/<selected branch>` in the named
repository at its recorded original head. The retained branch keeps that commit
reachable. No workflow runs or artifacts are deleted.
