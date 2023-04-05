## Schrodinger Bridges for Anomaly Detection

# Requirements

[Tensorflow 2.9.0](https://www.tensorflow.org/)

# Data

Data files are the taken from the [LHC Olympics 2020](https://zenodo.org/record/4536624). To access directly the high level features used in the work, you can download the preprocessed filed from [this folder](https://cernbox.cern.ch/index.php/s/Bo1QFzUojLQ5Eae).

# Training

To train the bridge you can run:

```bash
python lhco.py
```

# Plotting

We can verify how well the diffusion process worked by running the plot script as:

```bash
python plot_lhco.py --stage 1
```
where the ```--stage``` flags selects which IPF iteration to load.