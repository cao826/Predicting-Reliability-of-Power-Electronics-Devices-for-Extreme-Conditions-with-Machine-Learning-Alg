# Traditional ML readme file (Final title TBA)

This directory containes the resources and scripts for training and inference with Gradient Boosted models. 

## Running the code

The code is designed to both completely reproduce the results presented in the paper, and be used to create new predictions and results. 

### Reproducing the paper's results

The paper's results are based on predicted curves created in the latest training and inference run. Those predicted curves are saved to a file, and TODO we need to figure out how to distribute them. 
Reproducing the results exactly will require that you download or access those saved predicted curves in some way, and then provide a path to them in the script titled ```script-to-compute-results-based.py```. 
Specifically, change line number 9:

```python
predicted_curves_path = "/Users/..." #change to the path to OUR predicted curves in your system
```
and line number 14:

```python
data_path = "/Users/..." # change to the path to the modeling data in your local system.
```

### Generating new predicted curves and computing results with them

To generate a new set of predicted curves and compute results based on them, first, change the line number 6 in the file ```script-to-generate-xgboost-predicted-curves.py```: 

```python
path = "/Users/..."
```

to the path to the modeling data in your local system. 

Then, change the paths in ```script-to-compute-results-based.py``` as such: 

First, change line 9:

```python
path = "/Users/..." #Change this to the path to the NEW predicted curves you computed.
```

Then, change line 14: 

```python
path = "/Users/..." #Change this to the path to the modeling data in your system
```

