import numpy as np
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_SEED = 5


class NaiveBayes(object):
    def __init__(self):
        pass

    def likelihood_ratio(self, ratings_stars):  # [5pts]
        '''
        Args:
            rating_stars is a python list of numpy arrays that is length <number of labels> x 1
            
            Example rating_stars for Five-label NB model:
    
            ratings_stars = [ratings_1, ratings_2, ratings_3, ratings_4, ratings_5] -- length 5

            ratings_1: N_ratings_1 x D
                where N_ratings_1 is the number of reviews that gave an Amazon
                product a 1-star rating and D is the number of features (we use the word count as the feature)
            ratings_2: N_ratings_2 x D
                where N_ratings_2 is the number of reviews that gave an Amazon
                product a 2-star rating and D is the number of features (we use the word count as the feature)
            ratings_3: N_ratings_3 x D
                where N_ratings_3 is the number of reviews that gave an Amazon
                product a 3-star rating and D is the number of features (we use the word count as the feature)
            ratings_4: N_ratings_4 x D
                where N_ratings_4 is the number of reviews that gave an Amazon
                product a 4-star rating and D is the number of features (we use the word count as the feature)
            ratings_5: N_ratings_5 x D
                where N_ratings_5 is the number of reviews that gave an Amazon
                product a 5-star rating and D is the number of features (we use the word count as the feature)
            
            If you look at the end of this python file, you will see a docstring that contains more examples!
            
        Return:
            likelihood_ratio: <number of labels> x D matrix of the likelihood ratio of different words for different class of speeches
        '''
        return np.array([(np.sum(i, axis=0) + 1) / (np.sum(i) + len(i[0])) for i in ratings_stars])

    def priors_prob(self, ratings):  # [5pts]
        '''
        Args:
            rating_stars is a python list of numpy arrays that is length <number of labels> x 1
            
            Example rating_stars for Five-label NB model:
    
            ratings_stars = [ratings_1, ratings_2, ratings_3, ratings_4, ratings_5] -- length 5

            ratings_1: N_ratings_1 x D
                where N_ratings_1 is the number of reviews that gave an Amazon
                product a 1-star rating and D is the number of features (we use the word count as the feature)
            ratings_2: N_ratings_2 x D
                where N_ratings_2 is the number of reviews that gave an Amazon
                product a 2-star rating and D is the number of features (we use the word count as the feature)
            ratings_3: N_ratings_3 x D
                where N_ratings_3 is the number of reviews that gave an Amazon
                product a 3-star rating and D is the number of features (we use the word count as the feature)
            ratings_4: N_ratings_4 x D
                where N_ratings_4 is the number of reviews that gave an Amazon
                product a 4-star rating and D is the number of features (we use the word count as the feature)
            ratings_5: N_ratings_5 x D
                where N_ratings_5 is the number of reviews that gave an Amazon
                product a 5-star rating and D is the number of features (we use the word count as the feature)
            
            If you look at the end of this python file, you will see a docstring that contains more examples!
            
        Return:
            priors_prob: 1 x <number of labels> matrix where each entry denotes the prior probability for each class
        '''
        ratings = np.array(ratings)
        result = []
        total = 0
        for i in range(len(ratings)):
            result.append(np.sum(ratings[i]))
            total += np.sum(np.sum(ratings[i]))
        return np.array(result) / total

    # [5pts]
    def analyze_star_rating(self, likelihood_ratio, priors_prob, X_test):
        '''
        Args:
            likelihood_ratio: <num labels> x D matrix of the likelihood ratio of different words for different class of news
            priors_prob: 1 x <num labels> matrix where each entry denotes the prior probability for each class
            X_test: N_test x D bag of words representation of the N_test number of news that we need to analyze its sentiment
        Return:
             1 x N_test list where each entry is a class label specific for the Naive Bayes model
        '''
        return np.array([np.argmax(np.prod(np.power(likelihood_ratio, X_test[i, :]), axis=1) * priors_prob) for i in range(len(X_test))])


'''
ADDITIONAL EXAMPLES for ratings_stars

ratings_stars: Python list that contains the labels per corresponding Naive Bayes models.

The length of ratings will change depending on which Naive Bayes model we are training.
You are highly encouraged to use a for-loop to iterate over ratings!
------------------------------------------------------------------------------------------------------------------------
Two-label NB model:
ratings_stars = [ratings_less_than_or_equal_to_2, ratings_greater_or_equal_to_3] -- length 2

ratings_less_than_or_equal_to_2: N_ratings_less_than_or_equal_to_2 x D
    where N_ratings_less_than_or_equal_to_2 is the number of reviews that gave an Amazon
    product a 1 or 2-star rating and D is the number of features (we use the word count as the feature)

ratings_greater_or_equal_to_3: N_ratings_greater_or_equal_to_3 x D
    where N_ratings_greater_or_equal_to_3 is the number of reviews that gave an Amazon
    product a 3, 4, or 5-star rating and D is the number of features (we use the word count as the feature)
------------------------------------------------------------------------------------------------------------------------
Three-label NB model:
ratings_stars = [ratings_less_than_or_equal_to_2, ratings_3, ratings_greater_or_equal_to_4] -- length 3

ratings_less_than_or_equal_to_2: N_ratings_less_than_or_equal_to_2 x D
    where N_ratings_less_than_or_equal_to_2 is the number of reviews that gave an Amazon
    product a 1 or 2-star rating and D is the number of features (we use the word count as the feature)

ratings_3: N_ratings_3 x D
    where N_ratings_3 is the number of reviews that gave an Amazon
    product a rating a 3-star and D is the number of features (we use the word count as the feature)

ratings_greater_or_equal_to_4: N_ratings_greater_or_equal_to_4 x D
    where N_ratings_greater_or_equal_to_4 is the number of reviews that gave an Amazon
    product a 4 or 5-star rating and D is the number of features (we use the word count as the feature)
------------------------------------------------------------------------------------------------------------------------
Four-label NB model:
ratings_stars = [ratings_less_than_or_equal_to_2, ratings_3, ratings_4, ratings_5] -- length 4

ratings_less_than_or_equal_to_2: N_ratings_less_than_or_equal_to_2 x D
    where N_ratings_less_than_or_equal_to_2 is the number of reviews that gave an Amazon
    product a 1 or 2-star rating and D is the number of features (we use the word count as the feature)

ratings_3: N_ratings_3 x D
    where N_ratings_3 is the number of reviews that gave an Amazon
    product a 3-star rating and D is the number of features (we use the word count as the feature)

ratings_4: N_ratings_4 x D
    where N_ratings_4 is the number of reviews that gave an Amazon
    product a 4-star rating and D is the number of features (we use the word count as the feature)

ratings_5: N_ratings_5 x D
    where N_ratings_5 is the number of reviews that gave an Amazon
    product a 5-star rating and D is the number of features (we use the word count as the feature)
------------------------------------------------------------------------------------------------------------------------
Five-label NB model:
ratings_stars = [ratings_1, ratings_2, ratings_3, ratings_4, ratings_5] -- length 5

ratings_1: N_ratings_1 x D
    where N_ratings_1 is the number of reviews that gave an Amazon
    product a 1-star rating and D is the number of features (we use the word count as the feature)

ratings_2: N_ratings_2 x D
    where N_ratings_2 is the number of reviews that gave an Amazon
    product a 2-star rating and D is the number of features (we use the word count as the feature)

ratings_3: N_ratings_3 x D
    where N_ratings_3 is the number of reviews that gave an Amazon
    product a 3-star rating and D is the number of features (we use the word count as the feature)

ratings_4: N_ratings_4 x D
    where N_ratings_4 is the number of reviews that gave an Amazon
    product a 4-star rating and D is the number of features (we use the word count as the feature)

ratings_5: N_ratings_5 x D
    where N_ratings_5 is the number of reviews that gave an Amazon
    product a 5-star rating and D is the number of features (we use the word count as the feature)
------------------------------------------------------------------------------------------------------------------------

*** Note, the variables inside the list are just placeholders. Do not reference with these variable names! ***
'''