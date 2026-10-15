# Kimi C96 profiling evidence

Baseline worker statistics and 100% kernel-share reports are retained here.
The supplied trace directory is pit2-p03-g03; baseline workflow runner was
mi355x-amd_pit2_p03_g07. The folder-to-run identity is not independently proven.
All eight supplied traces matched their stats exactly. Full raw kernel traces
remain local (approximately 4 GB total) and are not ordinary Git blobs here.

Baseline workflow: https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/38696489924
Corrective LSE smoke: https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/38718539330
The latter workflow was green but invalid: all eight workers failed numerical
validation. Read server.log for the first-context comparisons.

Shares use summed GPU kernel duration, not wall-clock or critical-path time.
