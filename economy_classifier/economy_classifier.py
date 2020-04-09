import numpy as np
import joblib
from sklearn.datasets import load_files
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


def article_classify(article: str, model_path: str, convect_path: str, diff_coef=.1) -> list:
    """
    Gets an article text and model path and countvector path (which is used for transforming the article), 
    then classifies the article with that model and 
    returns list of string name of categories which the article belongs to
    
    params model_path: saved and trained classification model
    params convect_path: saved and filled countvector
    params diff_coef: the difference between the max category result and the others
    """
    
    categ_dict = np.array(['agricult', 'crypto', 'energy', 'metals'])
    model = joblib.load(model_path) 
    convect = joblib.load(convect_path)
    X_test = convect.transform([article])
    
    y_pred = model.predict_proba(X_test).reshape(-1)
    result = categ_dict[y_pred > (y_pred.max() - diff_coef)]
    
    return result


s = open(r'StockMoon_categories/crypto/Speaker Update Bitstamp’s Miha Grčar Joins Barcelona Trading Conference.txt', 'rt').read()

# сюда вставь свой путь к модели классификации и к векторам которые я кинул
model = r'economy_classifier_RF.sav'
vectors = r'economy_classifier_CV.sav'

article_classify(s, model, vectors)
