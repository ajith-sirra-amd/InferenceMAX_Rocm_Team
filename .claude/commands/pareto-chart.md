---
name: pareto-chart
description: create a pareto frontier chart 
memory: project
model: opus
---

# parteo-chart create
create a pareto frontier chart from a given json file

## Workflow

```text
- [ ] 1) Ask users what json file to use. Json file can be multiple
- [ ] 2) Create 2 charts for each Json file. left: a tput_per_gpu pareto frontier chart and right: a ttft bar chart 
- [ ] 3) For multiple Json files, arrange them vertically in subplots. 
- [ ] 4) For pareto frontier chart, x-axis is 'mean_intvty' and unit is (tok/s/user), y-axis is 'tput_per_gpu' and unit is (tok/sec). do not use log scale, also connect a pareto frontier line only for increasing 'tput_pet_gpu' points
- [ ] 5) For pareto frontier chart, each plot market shows this label [c='conc',gpus='tp']
- [ ] 6) For ttft bar chart, x-axis is 'conc' and unit is (size), y-axis is 'mean_ttft' and unit is (sec). do not use log scale
- [ ] 7) title shows 'hw', 'model', 'precision', 'framework'
- [ ] 8) show 'offloading': 'none' and 'offloading' : 'lmcache' and 'offloading' : 'hicache' in different colors