<a name="readme-top"></a>

# TPhaseNet: Phase picking for regional seismic data

Code related to the submitted paper **Deep learning models for regional phase detection on seismic stations in Northern Europe and the European Arctic**.

Tested setup for installation of required packages  :

```
conda create -n test python=3.10.12
conda activate test
pip install -r requirements.txt
```

To train a model for phase detection run :

```
python train.py
```

All training and model parameters can be changed in config.yaml.

This is a simply training example on a cpu. It is recommend to train on gpu.
To train on gpu you have to adapt the call of train.py for example using docker.

Due to limited space on GitHub, this example uses a dummy training data set with only a few events (tf/data).
Currently, to train you own model, you must generate your own data. The data structure can be explored by loading
the dummy training data files. We plan to make a larger dataset available.

The models used in the paper are also too large for GitHub and will be provided here soon:

[Zenodo repository](https://www.doi.org/10.5281/zenodo.11231543)

To pick arrivals in continuous data of station ARA0 first download the models, store tjem in tf/output/models,
adjust pred_config.yaml, and then run:

```
python predict.py
```


<!-- CONTACT -->
## Contact

Andreas Köhler - andreas.kohler@norsar.no - [ORCID](https://orcid.org/0000-0002-1060-7637)

Erik B. Myklebust - [ORCID](https://orcid.org/0000-0002-3056-2544)


Project Link: [https://github.com/NorwegianSeismicArray/tphasenet](https://github.com/NorwegianSeismicArray/tphasenet)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Models are built with [TensorFLow](https://www.tensorflow.org/)
* ARCES waveform data are available via the [Norwegian EIDA node](https://eida.geo.uib.no/webdc3/)
* Reviewed seismic event bulletins from which the input data labels were obtained are available from the [Finish National Seismic Network](https://www.seismo.helsinki.fi/bulletin/list/norBull.html
) and [NORSAR](http://www.norsardata.no/NDC/bulletins/regional/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

                        
