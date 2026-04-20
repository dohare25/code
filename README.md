# Mantle reservoir classification code

This folder contains the full code used in the study on
machine-learning classification of mantle reservoirs (DM, EM1, EM2,
and HIMU) from MORB and OIB trace-element data.

## Files

- `AutoML_pipeline.py`  
  Single end-to-end script that:
  1. Loads the processed training and test data from `../data/`
  2. Runs H2O AutoML with the settings used in the manuscript
  3. Evaluates the final model and exports predictions/metrics

  The script run_pipeline.py expects the processed training dataset at ../data/training_data.csv and writes outputs to ../outputs/. These paths are defined
  relative to the script and should work unchanged after cloning the repository.

  See the comments inside `run_pipeline.py` for detailed descriptions of
  each processing and modeling step.

- `classifier.py`  
  Command-line tool that uses the trained H2O AutoML model exported in
  the paper to classify new MORB/OIB samples. It supports:
    - Batch classification from a CSV file with the six required
      features (U/Pb, Nb, Th/U, La, Rb, Ba)
    - Interactive classification of a single sample via manual input  
  The script loads the saved model from the `saved_model/` subdirectory
  and outputs predicted mantle reservoir labels (DM, EM1, EM2, HIMU)
  plus per-class probabilities.

## Requirements

- Python 3.19.9
- H2O and standard scientific Python libraries

## License

The code is released under the MIT License. See the `LICENSE` file in
the repository root for details.
