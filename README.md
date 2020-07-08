# CNNAthUbi:Source code to build ubiquitination prediction model for Arabidopsis Thaliana
CNNAthUbi.py implements five-fold cross-validation and independent test.
proteome_score.7z stores the prediction scores of lysine residues in Arabidopsis Thaliana proteome. The scores are predicted by the model, which is trained using the four datasets (pos_train_dataset, pos_neg_dataset, neg_train_dataset, neg_test_dataset).
Model.h5 is a trained CNNArabUbi model using the four datasets (pos_train_dataset, pos_neg_dataset, neg_train_dataset, neg_test_dataset) as training set.
sequences.fasta is an example of protein sequence file in fasta format.
predictor.py is a script using the trained CNNArabUbi model to predict ubiquitination sites in protein sequnces which are displayed as fasta format in a file.
