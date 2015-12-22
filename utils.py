from functools import reduce
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold


class Data:
    def __init__(self, test=False):
        data = pd.read_json('train.json')
        self.recipes = data.ingredients.values

        self.y = data.cuisine.values
        self.y_model = LabelEncoder()
        self.y = np.array(self.y_model.fit_transform(self.y), dtype=np.int8)
        # Hot coded y. For each row there is only one 1 on y_i position.
        self.y_hc = np.array(np.zeros(
            (self.y.shape[0], len(self.y_model.classes_))), dtype=np.int8)
        for n, index in enumerate(self.y):
            self.y_hc[n, index] = 1

        # Get ingredient description map: ingredient -> ingredient_id
        self.ids = {}
        indices = iter(range(10**6))
        for recipe in self.recipes:
            for ingredient in recipe:
                if ingredient not in self.ids:
                    self.ids[ingredient] = next(indices)

        # For each recipe make text description (document), connecting all
        # ingredients in one line
        documents = []
        for recipe in self.recipes:
            document = reduce(lambda a, b: a + ' ' + b, recipe, '')
            documents.append(document)

        # We have 4 data models.

        # Ingredient model is simply encoded ingredients,
        # where for recipe i, vector x has 1 values only in positions,
        # corresponding to ingredients, containing in i_th recipe.
        ingredient_model = CountVectorizer(vocabulary=self.ids)
        # Word model is a bag of words, based on text description of recipes.
        word_model = CountVectorizer()
        # TFIDF model, based on text description of recipes.
        tfidf_word_model = TfidfVectorizer()

        self.ingredient_data = ingredient_model.fit_transform(documents)
        self.word_data = word_model.fit_transform(documents)
        self.tfidf_word_data = tfidf_word_model.fit_transform(documents)

        self.documents = documents

        self.cv4 = StratifiedKFold(self.y, 4, random_state=7)
        self.cv10 = StratifiedKFold(self.y, 10, random_state=7)

        if test:
            self.data_test = pd.read_json('test.json')
            self.recipes_test = self.data_test.ingredients.values
            self.test_ids = self.data_test.id.values
            documents_test = []
            for recipe in self.recipes_test:
                document = reduce(lambda a, b: a + ' ' + b, recipe, '')
                documents_test.append(document)

            self.ingredient_test = ingredient_model.transform(documents_test)
            self.word_test = word_model.transform(documents_test)
            self.tfidf_word_test = tfidf_word_model.transform(documents_test)

    def _cross_val_predict(self, model, x, y, cv):
        predictions = []
        for train, test in cv:
            model.fit(x[train], y[train])
            d = model.predict_proba(x[test])
            if d.ndim == 1:
                d = d.reshape((d.shape[0], 1))
            predictions.append(d)

        prediction = np.empty((y.shape[0], predictions[0].shape[1]))
        for p, (train, test) in zip(predictions, cv):
            prediction[test] = p
        return prediction

    def predict_cv(self, x_train, y_train, x_test, model, model_name):
        print('predicting for train set')
        y_train_proba = self._cross_val_predict(model, x_train, y_train,
                                                self.cv10)
        np.save(model_name + '_train', y_train_proba)
        print('predicting for test set')
        model.fit(x_train, y_train)
        y_test = model.predict_proba(x_test)
        np.save(model_name + '_test', y_test)
