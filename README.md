# DADS Experiment

### Install
``` shell
pip install requirements.txt
```

### Run
To run default experiment 1.1 on DADS
``` shell
python benchmark.py dads
```

To change semi-supervised algorithm, please choose a certain "**algo**" in "**dads**", "**ssad**", "**deepSAD**", "**supervised**", "**unsupervised**", "**vime**", "**devnet**", and execute
``` shell
python benchmark.py algo
```

To change experiment configuration of a certain algorithm "**algo**", please switch to directory **/config/algo/data_config.toml**.
Experiment parameters of setting1.1, setting1.2 and setting2 are respectively annotated. Uncomment the chosen setting and annotate others.

To change model configuration of a certain algorithm "**algo**", please switch to directory **/config/algo/model_config.toml**.