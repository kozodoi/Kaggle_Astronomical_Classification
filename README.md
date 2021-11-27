# PLAsTiCC Astronomical Classification

Top-5% solution to the [PLAsTiCC Astronomical Classification](https://www.kaggle.com/c/PLAsTiCC-2018) Kaggle competition on the classification of astronomical signal sources.

![cover](https://i.postimg.cc/fyy2hLHr/space-cover.jpg)


## Summary

This project works with a large data set containing the telescope data from different astronomical sources such as planets, supernovae and others. Using the time series of the objects brightness referred to as light curves and available object meta-data, we preprocess the data and build a classification pipeline with LightGBM and MLP models. Our solution represents a stacking ensemble of multiple base models for galactic and extragalactic objects. The solution reaches the top-2% of the Kaggle competition leaderboard.


## Project structure

The project has the following structure:
- `codes/`:  Jupyter notebooks with Python codes for modeling and ensembling
- `input/`: preprocessed input data, including train and test sets
- `output/`: figures produced when executing the modeling notebooks
- `oof_preds_joao/`: out-of-fold predictions produced by Joao's models
- `oof_preds_nik/`: out-of-fold predictions produced by Nikita's models
- `preds_stacking/`: test set predictions produced by the stacking ensemble
