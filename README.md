# DADS Experiment

### Install
``` shell
pip install requirements.txt
```

### Run
To run default experiment 1.1 on DADS
``` shell
python3 benchmark.py dads 1-1
```
or
``` shell
./run.sh
```

To specify algorithm and dataset setting, please:

choose a certain "**algo**" in 
"**dads**", "**ssad**", "**deepSAD**", "**supervised**", "**unsupervised**", "**vime**", "**devnet**", "**dplan**", 
"**static_stoc**", "**dynamic_stoc**";

choose a certain "**setting**" in "**1-1**", "**1-2**", "**2-1**", "**2-1**".

Then execute: 
``` shell
python benchmark.py algo setting
```


To change model configuration of a certain algorithm "**algo**", please switch to directory **/config/algo/model_config.toml**.