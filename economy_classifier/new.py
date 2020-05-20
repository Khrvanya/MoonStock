import numpy as np
import joblib
import os


class Node:
    def __init__(self, name: str, model_path=None, vectorizer_path=None, children_dict=None):
        if children_dict is None:
            children_dict = {}
        self.name = name
        self.__children_dict = children_dict  # dict of Nodes as values and Nodes.name as keys
        self.__saved_model = model_path
        self.__saved_vectorizer = vectorizer_path

    def get_children_dict(self):
        return self.__children_dict

    def get_model(self):
        return self.__saved_model

    def get_vectorizer(self):
        return self.__saved_vectorizer


def make_node_structure() -> Node:
    """
    Makes a Node structure, which looks like a tree
    Returns the tree root
    """

    agricult = Node('argicult', joblib.load(os.path.join(os.path.abspath(os.curdir),
                                             'finance/agricult/agricult_classifier_RF.sav')),
                    joblib.load(os.path.join(os.path.abspath(os.curdir),
                                 'finance/agricult/agricult_classifier_CV.sav')),
                    {'cattle': Node('cattle'), 'corn': Node('corn'),
                     'soybean': Node('soybean'), 'sugar': Node('sugar')})

    crypto = Node('crypto', joblib.load(os.path.join(os.path.abspath(os.curdir),
                                         'finance/crypto/crypto_classifier_RF.sav')),
                  joblib.load(os.path.join(os.path.abspath(os.curdir),
                               'finance/crypto/crypto_classifier_CV.sav')),
                  {'bitcoin': Node('bitcoin'), 'dash': Node('dash'),
                   'ethereum': Node('ethereum'), 'litecoin': Node('litecoin'),
                   'monero': Node('monero'), 'ripple': Node('ripple'), 'zash': Node('zash')})    

    energy = Node('energy', joblib.load(os.path.join(os.path.abspath(os.curdir),
                                         'finance/energy/energy_classifier_RF.sav')),
                  joblib.load(os.path.join(os.path.abspath(os.curdir),
                               'finance/energy/energy_classifier_CV.sav')),
                  {'brent crude': Node('brent crude'), 'coal': Node('coal'),
                   'crude oil': Node('crude oil'), 'natural gas': Node('natural gas')})

    metals = Node('metals', joblib.load(os.path.join(os.path.abspath(os.curdir),
                                         'finance/metals/metals_classifier_RF.sav')),
                  joblib.load(os.path.join(os.path.abspath(os.curdir),
                               'finance/metals/metals_classifier_CV.sav')),
                  {'gold': Node('gold'), 'iron': Node('iron'),
                   'platinum': Node('platinum'), 'silver': Node('silver')})

    finance = Node('finance', joblib.load(os.path.join(os.path.abspath(os.curdir),
                                           'finance/finance_classifier_RF.sav')),
                   joblib.load(os.path.join(os.path.abspath(os.curdir),
                                'finance/finance_classifier_CV.sav')),
                   {'agricult': agricult, 'crypto': crypto,
                    'energy': energy, 'metals': metals})

    return finance


def article_classification(article: str, node: Node, diff_coef=.1) -> list:
    """
    Gets an article text and model path and countvector path (which is used for transforming the article), 
    then classifies the article with that model and 
    Returns list of string name of categories which the article belongs to

    params article: the article that has to be classified
    params node: has filled countvector and trained classification model
    params diff_coef: the difference between the max category result and the others
    """

    x_test = node.get_vectorizer().transform([article])
    y_pred = node.get_model().predict_proba(x_test).reshape(-1)

    result = np.array(list(node.get_children_dict()))[y_pred > (y_pred.max() - diff_coef)]

    return result


def article_classification_tree(article: str, node=make_node_structure(), diff_coef=.1) -> dict:
    """
    Classifies an article above all the tree, which you can see in node structure
    Gets Node structure, article, diff_coef, which is metioned above
    Returns a dictionary, which is used rucursively to create categories and 
    subcategories and ... which this article is in

    params article:  the text article that has to be classified
    params node: a tree structure that classifies the article
    params diff_coef: the difference between the max category result and the others
    """

    if not node.get_children_dict():
        return {}

    diction = {}
    for category in article_classification(article, node, diff_coef):
        diction[category] = article_classification_tree(article, node.get_children_dict()[category])

    return diction


article_path = r'../categories_articles/energy/‘I love energy,’ says trader who sees tremendous buying opportunity in the space.txt'
f = open(article_path, 'r')
f = f.read()

print(article_classification_tree(f))

# поменяй статью 

# качай всю папку economy_classifier