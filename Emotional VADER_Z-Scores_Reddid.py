#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pandas as pd 
import numpy as np
import random


# In[3]:


# coding: utf-8
# Author: M. Wozniak

import os
import re
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from io import open

# ##Constants##

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
N_SCALAR = -0.74

NEGATE =     ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs

BOOSTER_DICT =     {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR, 
     "completely": B_INCR, "considerable": B_INCR, "considerably": B_INCR,
     "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormous": B_INCR, "enormously": B_INCR,
     "entirely": B_INCR, "especially": B_INCR, "exceptional": B_INCR, "exceptionally": B_INCR, 
     "extreme": B_INCR, "extremely": B_INCR,
     "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR, "frackin": B_INCR, "fracking": B_INCR,
     "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR, 
     "fuckin": B_INCR, "fucking": B_INCR, "fuggin": B_INCR, "fugging": B_INCR,
     "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR, 
     "incredible": B_INCR, "incredibly": B_INCR, "intensely": B_INCR, 
     "major": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
     "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
     "so": B_INCR, "substantially": B_INCR,
     "thoroughly": B_INCR, "total": B_INCR, "totally": B_INCR, "tremendous": B_INCR, "tremendously": B_INCR,
     "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utter": B_INCR, "utterly": B_INCR,
     "very": B_INCR,
     "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
     "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
     "less": B_DECR, "little": B_DECR, "marginal": B_DECR, "marginally": B_DECR,
     "occasional": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
     "scarce": B_DECR, "scarcely": B_DECR, "slight": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
     "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}

# check for sentiment laden idioms that do not contain lexicon words (future work, not yet implemented)
SENTIMENT_LADEN_IDIOMS = {"cut the mustard": 2, "hand to mouth": -2,
                          "back handed": -2, "blow smoke": -2, "blowing smoke": -2,
                          "upper hand": 1, "break a leg": 2,
                          "cooking with gas": 2, "in the black": 2, "in the red": -2,
                          "on the ball": 2, "under the weather": -2}

# check for special case idioms and phrases containing lexicon words
SPECIAL_CASES = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "badass": 1.5, "bus stop": 0.0,
                 "yeah right": -2, "kiss of death": -1.5, "to die for": 3, 
                 "beating heart": 3.1, "broken heart": -2.9 }


# #Static methods# #

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE)
    for word in neg_words:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "n't" in word:
                return True
    '''if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True'''
    return False


def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped

class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt",anger_lex="anger-scores.txt", 
                 anticipation_lex="anticipation-scores.txt", disgust_lex="disgust-scores.txt", fear_lex="fear-scores.txt",
                 joy_lex="joy-scores.txt", sadness_lex="sadness-scores.txt", surprise_lex="surprise-scores.txt", 
                 trust_lex="trust-scores.txt"):
#     def __init__(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt",anger_lex="anger-updated.txt", 
#              anticipation_lex="anticipation-updated.txt", disgust_lex="disgust-updated.txt", fear_lex="fear-updated.txt",
#              joy_lex="joy-updated.txt", sadness_lex="sadness-updated.txt", surprise_lex="surprise-updated.txt", 
#              trust_lex="trust-updated.txt"):
            
        
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        
        lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
        with codecs.open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        emoji_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), emoji_lexicon)
        with codecs.open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()
        
        anger_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), anger_lex)
        with codecs.open(anger_full_filepath, encoding='utf-8') as f:
            self.anger_full_filepath = f.read()
        self.anger_lexicon = self.make_anger_dict()
        
        anticipation_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), anticipation_lex)
        with codecs.open(anticipation_full_filepath, encoding='utf-8') as f:
            self.anticipation_full_filepath = f.read()
        self.anticipation_lexicon = self.make_anticipation_dict()

        disgust_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), disgust_lex)
        with codecs.open(disgust_full_filepath, encoding='utf-8') as f:
            self.disgust_full_filepath = f.read()
        self.disgust_lexicon = self.make_disgust_dict()

        fear_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), fear_lex)
        with codecs.open(fear_full_filepath, encoding='utf-8') as f:
            self.fear_full_filepath = f.read()
        self.fear_lexicon = self.make_fear_dict()

        joy_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), joy_lex)
        with codecs.open(joy_full_filepath, encoding='utf-8') as f:
            self.joy_full_filepath = f.read()
        self.joy_lexicon = self.make_joy_dict()

        sadness_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), sadness_lex)
        with codecs.open(sadness_full_filepath, encoding='utf-8') as f:
            self.sadness_full_filepath = f.read()
        self.sadness_lexicon = self.make_sadness_dict()

        surprise_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), surprise_lex)
        with codecs.open(surprise_full_filepath, encoding='utf-8') as f:
            self.surprise_full_filepath = f.read()
        self.surprise_lexicon = self.make_surprise_dict()

        trust_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), trust_lex)
        with codecs.open(trust_full_filepath, encoding='utf-8') as f:
            self.trust_full_filepath = f.read()
        self.trust_lexicon = self.make_trust_dict()
        
    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
            if not line:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.rstrip('\n').split('\n'):
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict
    
    def make_anger_dict(self):
        """
        Convert anger lexicon file to a dictionary
        """
        anger_dict = {}
        for line in self.anger_full_filepath.rstrip('\n').split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            anger_dict[word] = float(measure)
        return anger_dict
    
    def make_anticipation_dict(self):
        """
        Convert anticipation lexicon file to a dictionary
        """
        anticipation_dict = {}
        for line in self.anticipation_full_filepath.rstrip('\n').split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            anticipation_dict[word] = float(measure)
        return anticipation_dict
    
    def make_disgust_dict(self):
        """
        Convert disgust lexicon file to a dictionary
        """
        disgust_dict = {}
        for line in self.disgust_full_filepath.rstrip('\n').split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            disgust_dict[word] = float(measure)
        return disgust_dict
    
    def make_fear_dict(self):
        """
        Convert fear lexicon file to a dictionary
        """
        fear_dict = {}
        for line in self.fear_full_filepath.rstrip('\n').split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            fear_dict[word] = float(measure)
        return fear_dict
    
    def make_joy_dict(self):
        """
        Convert joy lexicon file to a dictionary
        """
        joy_dict = {}
        for line in self.joy_full_filepath.rstrip('\n').split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            joy_dict[word] = float(measure)
        return joy_dict
    
    def make_sadness_dict(self):
        """
        Convert sadness exicon file to a dictionary
        """
        sadness_dict = {}
        for line in self.sadness_full_filepath.rstrip('\n').split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            sadness_dict[word] = float(measure)
        return sadness_dict
    
    def make_surprise_dict(self):
        """
        Convert surprise lexicon file to a dictionary
        """
        surprise_dict = {}
        for line in self.surprise_full_filepath.rstrip('\n').split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            surprise_dict[word] = float(measure)
        return surprise_dict
    
    def make_trust_dict(self):
        """
        Convert trust lexicon file to a dictionary
        """
        trust_dict = {}
        for line in self.trust_full_filepath.rstrip('\n').split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            trust_dict[word] = float(measure)
        return trust_dict
    
    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence/scores, negative value are negative
        valence/scores.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for chr in text:
            if chr in self.emojis:
                # get the textual description
                description = self.emojis[chr]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr
                prev_space = chr == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)
        words_and_emoticons = sentitext.words_and_emoticons

        sentiments = []
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        anger, anticipation, disgust, fear, joy, sadness, surprise, trust = [[] for emotion in range(len(emotions))]
#         dic = {emotion:0 for emotion in emotions}
    
        for i, item in enumerate(words_and_emoticons):
            valence = 0
            anger_score, anticipation_score, disgust_score, fear_score, joy_score, sadness_score,            surprise_score, trust_score = [0 for emotion in range(len(emotions))]
            # check for vader_lexicon words that may be used as modifiers or negations
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)                    
                anger.append(anger_score)
                anticipation.append(anticipation_score)
                disgust.append(disgust_score)
                fear.append(fear_score)
                joy.append(joy_score)
                sadness.append(sadness_score)
                surprise.append(surprise_score)
                trust.append(trust_score)
                continue
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of"):
                sentiments.append(valence)                    
                anger.append(anger_score)
                anticipation.append(anticipation_score)
                disgust.append(disgust_score)
                fear.append(fear_score)
                joy.append(joy_score)
                sadness.append(sadness_score)
                surprise.append(surprise_score)
                trust.append(trust_score)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments, self.lexicon)            
            anger = self.sentiment_valence(anger_score, sentitext, item, i, anger, self.anger_lexicon)
            anticipation = self.sentiment_valence(anticipation_score, sentitext, item, i, anticipation, self.anticipation_lexicon)
            disgust = self.sentiment_valence(disgust_score, sentitext, item, i, disgust, self.disgust_lexicon)
            fear = self.sentiment_valence(fear_score, sentitext, item, i, fear, self.fear_lexicon)
            joy = self.sentiment_valence(joy_score, sentitext, item, i, joy, self.joy_lexicon)
            sadness = self.sentiment_valence(sadness_score, sentitext, item, i, sadness, self.sadness_lexicon)
            surprise = self.sentiment_valence(surprise_score, sentitext, item, i, surprise, self.surprise_lexicon)
            trust = self.sentiment_valence(trust_score, sentitext, item, i, trust, self.trust_lexicon)

        sentiments = self._but_check(words_and_emoticons, sentiments)
        anger = self._but_check(words_and_emoticons, anger)
        anticipation = self._but_check(words_and_emoticons, anticipation)
        disgust = self._but_check(words_and_emoticons, disgust)
        fear = self._but_check(words_and_emoticons, fear)
        joy = self._but_check(words_and_emoticons, joy)
        sadness = self._but_check(words_and_emoticons, sadness)
        surprise = self._but_check(words_and_emoticons, surprise)
        trust = self._but_check(words_and_emoticons, trust)

        valence_dict = self.score_valence(sentiments, anger, anticipation, disgust, fear, joy, sadness, surprise, trust, text)

        return valence_dict

    def sentiment_valence(self, valence, sentitext, item, i, sentiments, lexicon):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in lexicon:
            # get the sentiment valence 
            valence = lexicon[item_lowercase]
                
            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == "no" and i != len(words_and_emoticons)-1 and words_and_emoticons[i + 1].lower() in self.lexicon:
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() == "no")                or (i > 1 and words_and_emoticons[i - 2].lower() == "no")                or (i > 2 and words_and_emoticons[i - 3].lower() == "no" and words_and_emoticons[i - 1].lower() in ["or", "nor"] ):
                valence = lexicon[item_lowercase] * N_SCALAR
            
            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in lexicon:
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_and_emoticons, i)

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        return sentiments


    def _least_check(self, valence, words_and_emoticons, i):
        '''FIND BETTER SOLUTION - is it at leas or can be something else? -> it requires 8 functions'''
        # check for negation case using "least"
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon                 and words_and_emoticons[i - 1].lower() == "least":
            if words_and_emoticons[i - 2].lower() != "at" and words_and_emoticons[i - 2].lower() != "very":
                valence = valence * N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon                 and words_and_emoticons[i - 1].lower() == "least":
            valence = valence * N_SCALAR
        return valence
    

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if 'but' in words_and_emoticons_lower:
            bi = words_and_emoticons_lower.index('but')
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    @staticmethod
    def _special_idioms_check(valence, words_and_emoticons, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        onezero = "{0} {1}".format(words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                          words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoone = "{0} {1}".format(words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                           words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwo = "{0} {1}".format(words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASES:
                valence = SPECIAL_CASES[seq]
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
            if zeroone in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroone]
        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                              words_and_emoticons_lower[i + 2])
            if zeroonetwo in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroonetwo]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in BOOSTER_DICT:
                valence = valence + BOOSTER_DICT[n_gram]
        return valence

    @staticmethod
    def _sentiment_laden_idioms_check(valence, senti_text_lower):
        # Future Work
        # check for sentiment laden idioms that don't contain a lexicon word
        idioms_valences = []
        for idiom in SENTIMENT_LADEN_IDIOMS:
            if idiom in senti_text_lower:
                print(idiom, senti_text_lower)
                valence = SENTIMENT_LADEN_IDIOMS[idiom]
                idioms_valences.append(valence)
        if len(idioms_valences) > 0:
            valence = sum(idioms_valences) / float(len(idioms_valences))
        return valence

    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 1 word preceding lexicon word (w/o stopwords)
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == "never" and                     (words_and_emoticons_lower[i - 1] == "so" or
                     words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 2] == "without" and                     words_and_emoticons_lower[i - 1] == "doubt":
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 2 words preceding the lexicon word position
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == "never" and                     (words_and_emoticons_lower[i - 2] == "so" or words_and_emoticons_lower[i - 2] == "this") or                     (words_and_emoticons_lower[i - 1] == "so" or words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 3] == "without" and                     (words_and_emoticons_lower[i - 2] == "doubt" or words_and_emoticons_lower[i - 1] == "doubt"):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 3 words preceding the lexicon word position
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, anger, anticipation, disgust, fear, joy, sadness, surprise, trust,  text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text) 
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)      

            
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0
            
        if anger:
            sum_anger = float(sum(anger))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_anger > 0:
                sum_anger += punct_emph_amplifier
            elif sum_anger < 0:
                sum_anger -= punct_emph_amplifier

            anger_intensity = normalize(sum_anger)
           
        if anticipation:
            sum_anticipation = float(sum(anticipation))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_anticipation > 0:
                sum_anticipation += punct_emph_amplifier
            elif sum_anticipation < 0:
                sum_anticipation -= punct_emph_amplifier

            anticipation_intensity = normalize(sum_anticipation)
            
        if disgust:
            sum_disgust = float(sum(disgust))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_disgust > 0:
                sum_disgust += punct_emph_amplifier
            elif sum_disgust < 0:
                sum_disgust -= punct_emph_amplifier

            disgust_intensity = normalize(sum_disgust)
            
        if fear:
            sum_fear = float(sum(fear))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_fear > 0:
                sum_fear += punct_emph_amplifier
            elif sum_fear < 0:
                sum_fear -= punct_emph_amplifier

            fear_intensity = normalize(sum_fear)
            
        if joy:
            sum_joy = float(sum(joy))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_joy > 0:
                sum_joy += punct_emph_amplifier
            elif sum_joy < 0:
                sum_joy -= punct_emph_amplifier

            joy_intensity = normalize(sum_joy)
            
        if sadness:
            sum_sadness = float(sum(sadness))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_sadness > 0:
                sum_sadness += punct_emph_amplifier
            elif sum_sadness < 0:
                sum_sadness -= punct_emph_amplifier

            sadness_intensity = normalize(sum_sadness)
            
        if surprise:
            sum_surprise = float(sum(surprise))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_surprise > 0:
                sum_surprise += punct_emph_amplifier
            elif sum_surprise < 0:
                sum_surprise -= punct_emph_amplifier

            surprise_intensity = normalize(sum_surprise)
            
        if trust:
            sum_trust = float(sum(trust))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_trust > 0:
                sum_trust += punct_emph_amplifier
            elif sum_trust < 0:
                sum_trust -= punct_emph_amplifier

            trust_intensity = normalize(sum_trust)

        sentiment_dict =             {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4),
             "anger" : round(anger_intensity, 4),
             "anticipation" : round(anticipation_intensity, 4),
             "disgust" : round(disgust_intensity, 4),
             "fear" : round(fear_intensity, 4),
             "joy" : round(joy_intensity, 4),
             "sadness" : round(sadness_intensity, 4),
             "surprise" : round(surprise_intensity, 4),
             "trust" : round(trust_intensity, 4)}

        return sentiment_dict


# # COUNTING Z-SCORES FOR REDDIT COMMENTS TAGGED AND DESCRIBED IN https://arxiv.org/pdf/2005.00547.pdf

# In[8]:


df1 = pd.read_csv('goemotions_1.csv')
df2 = pd.read_csv('goemotions_2.csv')
df3 = pd.read_csv('goemotions_3.csv')

df = pd.concat([df1,df2,df3],axis=0,ignore_index=True)

emo_df = df.loc[:,['text','anger','disgust','fear','joy','sadness','surprise','approval','desire']].copy()

emotions = ['anger','disgust','fear','joy','sadness','surprise','trust','anticipation']
emo_df.columns = ['text', 'anger','disgust','fear','joy','sadness','surprise','trust','anticipation']
emo_df = emo_df[(emo_df.loc[:,emotions]==1).any(axis=1)].copy()
emo_df = emo_df.reset_index(drop=True)
emo_df = emo_df.drop_duplicates()


# In[4]:


'''How many posts elicit only one emotion in the dataset (how many for joy, how many for disgust, etc.)
   How many posts elicit at least that emotion (e.g. joy with #, disgust with #, fear with #, etc?)'''

emotion_counts = dict()

for emotion in emotions:
    emotion_counts[emotion] = []
    at_least_emotion_idx = emo_df[emo_df[emotion]==1].index
    only_emotion_idx = emo_df.iloc[:,1:][(emo_df.loc[:,emo_df.columns != emotion]!=1).all(axis=1) & emo_df[emotion]==1].index
    emotion_counts[emotion] = np.array([len(only_emotion_idx), len(at_least_emotion_idx)])
        
emotion_counts


# In[5]:


# anger_idx = np.array(emo_df[emo_df['anger']==1].index)
# all_idx = np.array(emo_df.index) 

# '''indices of posts which doesn't t contain given emotion'''
# not_anger_idx = [e for e in all_idx if e not in anger_idx]
# not_anger_df = emo_df.iloc[not_anger_idx, :]

# '''split into training and validation set ~ 50/50'''
# anger_train_idx = np.random.choice(not_anger_idx, int(len(not_anger_idx)/2))
# anger_validation_idx = [e for e in not_anger_idx if e not in anger_train_idx]

# anger_train_df = emo_df.iloc[anger_train_idx,:]
# anger_validation_df = emo_df.iloc[anger_validation_idx,:]


# mean_and_std_train = dict()
# analyzer = SentimentIntensityAnalyzer()
# mean_and_std_train['anger'] = []

# temporary_df = anger_train_df['text'].apply(lambda x: analyzer.polarity_scores(x)['anger'])
# mean_and_std_train['anger'] = np.array([np.mean(temporary_df), np.std(temporary_df)])


# emo_validation_df = pd.concat([not_anger_df, anger_validation_df],axis=0,ignore_index=True)

# z_scores = dict()
# mean = mean_and_std_train['anger'][0]
# std = mean_and_std_train['anger'][1]
# z_score = (emo_validation_df['text'].apply(lambda x: (analyzer.polarity_scores(x)['anger']-mean)/std)).sort_values()
# z_scores['anger'] = []
# z_scores['anger'].extend(sorted(z_score.values))



# emo_train_df = pd.concat([not_anger_df, anger_train_df],axis=0,ignore_index=True)

# # z_scores


# In[7]:


analyzer = SentimentIntensityAnalyzer()
mean_and_std_train = dict()
mean_and_std_validation = dict()
z_scores_train = dict()
z_scores_validation = dict()

for emotion in emotions:
    emotion_idx = np.array(emo_df[emo_df[emotion]==1].index)
    all_idx = np.array(emo_df.index) 

    '''indices of posts which doesn't t contain given emotion'''
    not_emotion_idx = [e for e in all_idx if e not in emotion_idx]
    not_emotion_df = emo_df.iloc[not_emotion_idx, :]

    '''split idx which doesn't contain given emotion into training and validation set ~ 50/50.
       Creatin corrensponding ddataframes'''
    train_idx = np.random.choice(not_emotion_idx, int(len(not_emotion_idx)/2))
    train_df = emo_df.iloc[train_idx,:]
    
    validation_idx = [e for e in not_emotion_idx if e not in train_idx]
    validation_df = emo_df.iloc[validation_idx,:]
    
    print(emotion, 'idx computed')
    
    ''' Counting Z-scores for training set'''
    mean_and_std_train[emotion] = []
    temporary_df_t = train_df['text'].apply(lambda x: analyzer.polarity_scores(x)[emotion])
    mean_and_std_train[emotion] = np.array([np.mean(temporary_df_t), np.std(temporary_df_t)])
    
    print(emotion, 'train: mean and std computed')

    emo_and_validation_df = pd.concat([not_emotion_df, validation_df],axis=0,ignore_index=True)
    
    mean_t = mean_and_std_train[emotion][0]
    std_t = mean_and_std_train[emotion][1]
    z_score_t = (emo_and_validation_df['text'].apply(lambda x: (analyzer.polarity_scores(x)[emotion]-mean_t)/std_t)).sort_values()
    z_scores_train[emotion] = []
    z_scores_train[emotion].extend(sorted(z_score_t.values))
    
    print(emotion, 'train: z-scores computed')
    
    ''' Counting Z-scores for validation set'''
    mean_and_std_validation[emotion] = []
    temporary_df_v = validation_df['text'].apply(lambda x: analyzer.polarity_scores(x)[emotion])
    mean_and_std_validation[emotion] = np.array([np.mean(temporary_df_v), np.std(temporary_df_v)])
    
    print(emotion, 'validation: mean and std computed')

    emo_and_train_df = pd.concat([not_emotion_df, train_df],axis=0,ignore_index=True)
    
    mean_v = mean_and_std_validation[emotion][0]
    std_v = mean_and_std_validation[emotion][1]
    z_score_v = (emo_and_train_df['text'].apply(lambda x: (analyzer.polarity_scores(x)[emotion]-mean_v)/std_v)).sort_values()
    z_scores_validation[emotion] = []
    z_scores_validation[emotion].extend(sorted(z_score_v.values))
    
    print(emotion, 'validation: z-scores computed')


# In[8]:


mean_and_std_train


# In[9]:


mean_and_std_validation


# In[10]:


z_scores_train


# In[11]:


z_scores_validation


# In[12]:


'''how many post labeled with z-score higher than 1.95 are present in the dataset'''

tp_count = dict()
for emotion in emotions:
    arr_train = np.array(z_scores_train[emotion])
    tp_count[emotion + '_train'] = len(arr_train[abs(arr_train>1.96)])
    
    arr_validation = np.array(z_scores_validation[emotion])
    tp_count[emotion + '_validation'] = len(arr_validation[abs(arr_validation>1.96)])

tp_count


# In[ ]:


# mean_and_std = dict()
# analyzer = SentimentIntensityAnalyzer()

# for emotion in emotions[:2]:
#     mean_and_std[emotion] = []
#     temporary_df = emo_df[emo_df[emotion]==1]['text'].apply(lambda x: analyzer.polarity_scores(x)[emotion])
#     mean_and_std[emotion] = np.array([np.mean(temporary_df), np.std(temporary_df)]) 
    
# mean_and_std


# In[ ]:


# z_scores = dict()

# for emotion in emotions[:2]:
#     idx = np.random.choice(emo_df[emo_df[emotion]==1].index,50)
#     mean = mean_and_std[emotion][0]
#     std = mean_and_std[emotion][1]
#     z_score = (emo_df.loc[idx,'text'].apply(lambda x: (analyzer.polarity_scores(x)[emotion]-mean)/std)).sort_values()
#     z_scores[emotion] = []
#     z_scores[emotion].extend(sorted(z_score.values))

# z_scores


# In[ ]:


# data = { 'Emotion': ['anger'], 'Mean':mean_and_std['anger'][0], 'Std':mean_and_std['anger'][1], 'Min z-score': z_scores['anger'][:1], 'Max z-score':z_scores['anger'][-1:]}  
# df = pd.DataFrame(data)  

# for emotion in emotions[1:]:
#     data = { 'Emotion': emotion, 'Mean':mean_and_std[emotion][0], 'Std':mean_and_std[emotion][1], 'Min z-score': z_scores[emotion][:1], 'Max z-score':z_scores[emotion][-1:]}  
#     temp_df = pd.DataFrame(data)  
#     df = df.append(temp_df)
    
# df

