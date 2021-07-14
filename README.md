# Predicting Reliability of Power Electronics Devices for Extreme Conditions with Machine Learning Algorithms

This repo contains all of the necessary scripts to both recreate the results presented in the original paper and use the code as a base to create your own predictions. 

##  Initialization and Requirements

You can initialize this repository by running

```bash
git clone https://github.com/cao826/suli-summer.git
```

To use this repository, please ensure your system meets the requirements below.

```bash
numpy==1.19.1
scipy==1.4.1
tqdm==4.59.0
pandas==1.1.0
matplotlib==3.2.2
tensorflow==2.2.0
scikit_learn==0.24.2
```

## Layout
This project has four subdirectories:

1. encoder-decoder-version: This subdirectory is where you can predict stress test response curves using an LSTM based encoder-decoder model
2. traditional-ml-version: Predicts stress response curves with Gradient Boosting models and evaluates them according to all of the different methodologies discussed in the paper.
3. Multimodal LSTM models: Do we still want to keep this? I only mention these models tangentially in the paper, and none of their results are shown, although I probably do want to include that (?)
4. recreate-results: This module contains a script which you can run to recreate the results presented in the paper. 
5. direct-pf-prediction: This sub-directory allows the user to predict the pass/fail status of a device directly. It also generates a box plot of accuracies by manufacturer, and both forms of PCA.
## Accessing Data

The data needed for this project is included in the repository, in the ```data``` directory. 
There are two datassets there: 

1. modeling-data.csv: This file the data formatted for Gradient Boosting models.
2. modeling-data-encoder-decoder-format.csv: The same data, but formatted for the encoder-decoder models.

Originally, this data contained information about the manufacturers of the power devices tested. 
The data has been modified so that the manufacturers are not identifiable. 
(Should I add more details about this?)

## Automate/ running the scripts

Every script in the project requires the user to provide a path to the appropriate data. 
Make sure to provide the correctly formatted data for each script. 
The proper filenames for each are included below

### Running the Gradient Boosting Script

In the ```traditional-ml-version``` direcotry, run 

```bash
python gradboost_script PATH-TO-DATA/modeling-data.csv
```

### Running the Encoder-Decoder Script

In the ```encoder-decoder-version``` subdirectory, run

```bash
python encoder_decoder_script.py PATH-TO-DATA/modeling-data-encoder-decoder-format.csv
```

### Recreating our results

In the ```recreate-results``` subdirectory, run

```bash
python script-to-compute-results.py PATH-TO-DATA/modeling-data.csv
```

### Driect P/F Prediction

In the ```direct-pf-prediction``` subdirectory, run

```bash
python nscript.py PATH-TO-DATA/ndevice-pipeline.xlsx
```
