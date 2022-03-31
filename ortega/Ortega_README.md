# HYGRIP data-set
The Hybrid Dynamic Grip data-set repository accompanying the paper <br>
> *HYGRIP: Full-stack characterisation of neurobehavioural signals (fNIRS, EEG, EMG,
>force & breathing) during a bimanual grip force control task*
> by Pablo Ortega, Tong Zhao and A Aldo Faisal
> Brain and Behaviour Lab, Imperial College London
> www.FaisalLab.org


- For a **quick set-up** to run the notebook `presentation.ipynb` containing the 
brief analysis presented in the paper refer to the "Quick set-up" section.
- For further details in **data collection** refer to sections below or read the
[publication](url).
- For further details on how to **exploit the data-set** for your own purposes 
refer to the "Exploiting the data-set" section and check commented classes and 
functions in `utils`.

> _Special **thanks to** all the anonymous volunteers and their time and open libraries 
> developers that have contributed to this project_

## Quick set-up

### Clone the repo and download data-set
1. In the terminal, go to the directory `<YOURDIR>` where you want the repo to be 
cloned <br>
`$ cd <YOURDIR>`

2. Clone the repo in that directory (will create `<YOURDIR>/hygrip`) <br>
`$ git clone https://gitlab.doc.ic.ac.uk/bbl/hygrip.git`

3. [Download from private link](https://figshare.com/s/00507be12a74f233be0d) the data-set `hygrip.h5` 
(4.33GB) into `<YOURDIR>/hygrip`<br>
(Note: the **public link** DOI [10.6084/m9.figshare.12383639](10.6084/m9.figshare.12383639)
will be made available in case of publication)


### Set-up with `conda`
First you will need to install [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).<br>
After cloning the repo, the file `environment.yml` should 
be available in `<YOURDIR>/hygrip`.<br>
Once `conda` is installed and the repo cloned:

1. Create an environment with all required dependencies to run the notebook
using* <br>
`$ conda env create -f environment.yml`

2. Activate the environment using* <br>
`$ conda activate hygrip` 

3. Launch the notebook <br>
`$ jupyter-lab presentation.ipynb`

*For Windows users using the `conda-prompt` the `conda` part of the command
needs to be ommited. 

## Custom set-up

### python
If you don't want to install all dependencies and create a new environment you 
just need to manually install the dependencies for `utils` manually.
The dependencies can be installed via your preferred packaging manager 
(`conda`, `pip`, etc.):<br> 
1. `h5py`
2. `numpy`
3. `scipy`
4. `scikitlearn`
5. `matplotlib` 

To run the notebook `jupyter-lab` or `jupyter` are  also needed.

### matlab
Since the data-set is a standalone H5 file this can be loaded in matlab directly
using `h5read` ([learn more](https://uk.mathworks.com/help/matlab/ref/h5read.html)).
We recommend to go first through the "Quick set-up" section to run the notebook
since it contains a through explanation of the data-set organisation and 
exploitation.

## HYGRIP description
The Hybrid Dynamic Grip (HYGRIP) data-set provides a complete data-set, that 
comprises the task signal (a dynamically varying force signal that is displayed 
visually), co-located brain signals in form of electroencephalography (EEG) and 
near-infrared spectroscopy (NIRS) that captures a higher spatial resolution 
oxygenation cortical response, the muscle activity of the grip muscles, the 
force generated at the grip sensor, as well as confounding sources, such as 
breathing and eye movement activity during a motor task. 
In total, 14 right-handed subjects performed a uni-manual dynamic grip force
task within 25-50% of each hand's maximum voluntary contraction. <br>

HYGRIP is intended as a benchmark with two open challenges and research 
questions for grip-force decoding:
1.  The exploitation and fusion of data from brain signals spanning very 
 different time-scales, as EEG changes about three orders of magnitude faster 
 than NIRS.
2. The decoding of whole brain signals associated with the use of each hand and 
the extent to which models share features for each hand, or conversely, are 
different for each hand.

The data-set provided as a single hard-disk file (HDF) has undergone very little
processing to avoid biasing future analyses. 
Preprocessing amounts to down-sampling to reduce storage space and the 
formatting of data, recorded events and other meta-data from different devices 
so that all data followed the same format regardless of their origin.

## HYGRIP organisation
The data-set follows a tree organisation in three levels.
In first and third levels the data-set contains meta-data in string format 
that can be accessed via the attributes of the level.
The first level is the data-set itself and the shared attributes across subjects
measures, e.g. sampling frequencies and units, and other information such as 
the channel grid disposition and a template of hybrid sensor 3D coordinates over
the scalp.
In the second level, the data-set is organised in one group per subject indexed 
by their anonymized ID (i.e. fourteen groups with keys `A` to `N`) and contain 
no attributes.
In the third level, each group contains a subgroup for each measure (e.g. keys 
`frc` for force and `eeg` for EEG) containing the data in numeric format and an 
attribute called `events` containing the times at which an instruction is given 
during the recording (e.g. 'relax', 'left-hand', 'right-hand') and the 'begin 
and 'end' time-stamps indicating the beginning and end of the recording session,
also numeric.
In particular, the `events` group contains a second numeric attribute `MVC` 
containing the maximum voluntary contraction value for each hand.

## Exploiting the data-set
The data-set (`hygrip.h5`) is a standalone HDF that can be loaded in `python`: 
`h5py`, `pandas`; `matlab`: `h5read`; and other software supporting H5 files.

The accompanying code in `utils` aims to provide a quick overview of the data-set
organisation and the data themselves. And it is designed around the class
`Pipeline` in `utils/pipelines.py`.

The class `Pipeline` is a callable object that is instantiated with a list of 
callable objects or _"pipeline steps"_.
```
step0 = APipelineStep(args, kwargs)
step1 = AnotherPipelineStep(args, kwargs)
my_pipe = Pipeline([step0, step1], name="FOO")
```
When it is called it just iteratively apply to processes contained in its list 
and returns the result.

`Pipeline` classes can be named and printed directly to easily trace back the
preprocessing applied to the data
```
[In]: print(my_pipe)
[Out]: Pipeline(FOO)
            APipelineStep(args, kwargs)
            AnotherPipelineStep(args, kwargs)
```

_"Pipeline steps"_ need to strictly follow the following convention:
```
class APipelineStep(object):
    """ Pipeline process
    Does something
    """
    def __init__(self, param1, param2):
        """ Parameters that will be needed during the call and define the processing step.
        """
        self._param1 = param1
        self._param2 = param2

    def __repr__(self):
        """ Representation of the processing step that will be used to print
        """
        return f"APipelineStep(param1={self._param1}, param2={self._param[2]})\n"

    def __call__(self, data, events, sfreq, **kwargs):
        """ Executed when instance of object is called.
            Admits additional keyworded arguments in case they are required.
            This allows for particular inputs to be exploited at a particular 
            step if the key is contained in `kwargs`.
        """
        # Do something using self._param1, and self._param2
        data = foo(data, sfreq, self._param1)
        events = foo1(events, sfreq, self._param2)
        return data, events, sfreq
```

The fundamental _"Pipeline steps"_ are provided in `utils/pipelines.py` 
and `utils/ica.py` (for modularity and due to its extension ICA related 
processing is contained in its own module **thanks to scikit-learn.org**).
In `utils/plots.py` the functions to produce the following plots are provided 
and are more specific of the data to plot.
Of special interest are the 2D grid plots for EEG and NIRS 
(`plot_eeg` and `plot_hb`) where it can be seen how to use "scalp_chan_grid".
With the provided structure it should be easy to implement ones 
own _"Pipeline steps"_ and expand the analysis.

**For the machine-learning community**<br>
Since data is loaded as numpy arrays
it would be easy to convert them to any popular machine learning library.
The less obvious and sensitive part is the separation of training, validation 
and test sets.
The most obvious way of splitting the data-set would be to load all subjects 
data and then randomly separate the examples.
If the purpose is classification, then gathering per labels would help having
balanced ammount of labels per split.
A less obvious and more challenging way of splitting the data is to perform a 
k-fold validation accross subjects. In this case the structure of the data-set 
is convenient since it will allow to gather all data for all subjects, except the 
one that is left out and would be used for testing (and so on for each fold).
This will test the average generalisation of a training procedure to unseen
subjects for which differences in physiologies, recording positions,
behaviours... should be accounted (or not, if it doesn't generalise) 
by the model.
