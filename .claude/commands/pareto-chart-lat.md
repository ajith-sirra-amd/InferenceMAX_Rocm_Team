---
name: pareto-chart-lat
description: create a pareto frontier chart 
memory: project
model: opus
---

# parteo-chart create
create a pareto frontier chart from a given json file

## Workflow

```text
- [ ] 1) Ask users what json file to use. Json file can be multiple
- [ ] 2) Create 2 charts for each Json file. left: a tput_per_gpu pareto frontier chart, middle: a ttft bar chart, right: a cache hit stacked bar chart
- [ ] 3) For multiple Json files, arrange them vertically in subplots. 
- [ ] 4) For pareto frontier chart, x-axis is 'p90_e2el' and unit is (s), y-axis is 'tput_per_gpu' and unit is (tok/sec). do not use log scale, also connect a pareto frontier line 
- [ ] 5) For pareto frontier chart, each plot market shows this label [c='conc',gpus='tp']
- [ ] 6) For ttft bar chart, x-axis is 'conc' and unit is (size), y-axis is 'p90_ttft' and unit is (sec). do not use log scale
- [ ] 7) For cache hit stacked bar chart, x-axis is 'conc' and unit is (size), y-axis is the value of "total" of "stats" key from "source": "local_compute", "source": "local_cache_hit", "source": "external_kv_transfer" from ../agentic*/aiperf_artifacts/server_metrics_export.json in the same folder as the Json file
- [ ] 8) title shows 'hw', 'model', 'precision', 'framework'
- [ ] 9) show 'offloading': 'none' and 'offloading' : 'lmcache' and 'offloading' : 'hicache' in different colors