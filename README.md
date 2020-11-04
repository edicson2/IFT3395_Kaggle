# IFT3395 - Classification d'articles
Pour rouler les codes, il faut avoir installé scikit-learn, numpy, matplotlib.pyplot, pandas


Classifieur de Bayes :

 - nécessite train.csv et test.csv pour importer les données d'entraînement et les données de test
 - nécessite common_english_words.txt qui est un fichier des mots communs de la langue anglaise qui ne sont pas tenus en compte
 - le code à exécuter est Naives_Bayes_w_final_predictions.ipnyb
 - produit tous les résultats voulus dont les predictions finales qui ont été soumises sur Kaggle (predictions_bayes_naif_test_set.csv)
 - produit aussi un graph qui teste l'hyperparamètre alpha (influence_alpha.png)


SVM : 

 - nécessite aussi train.csv et test.csv
 - code à rouler : svm_model.py
 - produit les prédictions sous : submission.csv
 - produit un graphique qui regarde les hyperparamètres C et gamma (influence_C_gamma.png)
