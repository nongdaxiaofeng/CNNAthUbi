# CNNAthUbi: ubiquitination site prediction for Arabidopsis Thaliana

This is a project for predicting ubiquitination sites in Arabidopsis Thaliana using convolutional neural networks (CNNs).

The binary encoding and amino acid properties are respectively as inputs of the CNNs to build models, which are named CNN_Binary and CNN_Property.

The dataset directory stores the training dataset and test dataset.

aaindex31 is the file of 31 amino acid properties.

pred_site_property.zip and pred_site_binary.zip are the files of predicted ubiquitination sites in Arabidopsis by CNN_Binary and CNN_Property.

The two files fpr_tpr_thr_binary and fpr_tpr_thr_property list the prediction thresholds, true positive rates (sensitivity), and false positive rates in five-fold cross-validation for CNN_Binary and CNN_Property respectively.

AraUbiSite is a previously developed ubiquitination site prediction tool for A. thaliana. AraUbiSite_cv_score.txt and AraUbiSite_ind_score.txt are files of prediction scores in five-fold cross-validation and independent test, which are provided by its webserver.

cv_ind_test.py performs five-fold cross validation and independent test for CNN_Binary and CNN_Property and plot ROC curves of them and AraUbiSite.

We use a simple grid search scheme to optimize hyper parameters that make the cross-validation perform best. The grid_search.py file performs grid search.

In order to run the program properly, the user needs to install python software.

The python libraries including numpy, keras, pandas, tensorflow and matplotlib are also required to install.

Before running the programms, all the files should be downloaded.

To run cv_ind_test.py, use the command below:
python cv_ind_test.py

To run grid_search.py, use the command below:
python grid_search.py
