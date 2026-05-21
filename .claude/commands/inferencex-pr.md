---
name: inferencex-pr
description: Create a PR to InferenceX (https://github.com/SemiAnalysisAI/InferenceX/pulls)
memory: project
model: sonnet
---

# Create a PR to InferenceX
Create a PR to InferenceX. Update 'image' and 'server arguments'

## Workflow

```text
- [ ] 1) Confirm 'model', 'model-prefix', 'runner', 'precision', 'framework', 'multinode'
- [ ] 2) Update 'image'
- [ ] 3) Update 'server arguments'
- [ ] 4) Create a PR
```

## 1) Confirm runner
Ask user to choose 'model', 'model-prefix', 'runner', 'precision', 'framework', 'multinode'.
Create a new branch under https://github.com/SemiAnalysisAI/InferenceX.

Rules:
- 1. 'model', 'model-prefix', 'runner', 'precision', 'framework', 'multinode' options can be found at https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/configs/amd-master.yaml
- 2. ask user to choose each options
- 3. branch naming is the 'key' of https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/configs/amd-master.yaml. If branch is already exists, use 'key'+'date'

## 2) Update 'image'
Ask user to update a docker image.

Rules:
- 1. Default vllm docker image is 'vllm/vllm-openai-rocm:nightly' from https://hub.docker.com/r/vllm/vllm-openai-rocm/tags
- 1. Update the https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/configs/amd-master.yaml of new branch

## 3) Update 'server arguments'
Update server arguments.

Rules:
- 1. Use a server arguments from https://github.com/SemiAnalysisAI/InferenceX/tree/main/benchmarks/single_node. Use similar 'model-prefix'_'precision'_'runner'_'freamework'.sh
- 2. Update the https://github.com/SemiAnalysisAI/InferenceX/tree/main/benchmarks/single_node/*.sh of new branch

## 4) Create a PR
Based on Step 1-3), create a PR to https://github.com/SemiAnalysisAI/InferenceX/pulls.

Rules:
- 1. PR title '[AMD][ROCM] 'key': Bump image to 'image'
- 2. Use these PR example
  PR example: 
  - 1. https://github.com/SemiAnalysisAI/InferenceX/pull/1311/changes
  - 2. https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/dsv4_fp4_mi355x_atom.sh
  - 3. https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/configs/amd-master.yaml#L1646
  - 4. https://github.com/SemiAnalysisAI/InferenceX/blob/main/perf-changelog.yaml#L1824-L1833
