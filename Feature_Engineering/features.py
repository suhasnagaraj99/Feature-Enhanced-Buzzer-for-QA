# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json
from eval import normalize_answer
from eval import rough_compare
import nltk
nltk.download('stopwords')
nltk.download('words')
# from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict
from nltk.corpus import words
from nltk.corpus import stopwords
import string
# import spacy
# spacy.cli.download("en_core_web_md")

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """

        question -- The JSON object of the original question, you can extract metadata from this such as the category

        run -- The subset of the question that the guesser made a guess on

        guess -- The guess created by the guesser

        guess_history -- Previous guesses (needs to be enabled via command line argument)

        other_guesses -- All guesses for this run
        """


        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        
        guess_length = log(len(guess))
        
        guess_word_length = log(len(guess.split()))

        run_word_length = log(len(run.split()))
        
        run_char_length = log(len(run))
                        
        yield ("guess", guess_length)  
        yield ("run_word", run_word_length)
        yield ("run_char", run_char_length) 
        yield ("guess_word", guess_word_length)
        

class GuessHistoryFeature(Feature):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        
        guess_counter  = Counter()
        keys = guess_history['gpr'].keys()
        for key in keys:
            guess_counter[guess_history['gpr'][key][0]['guess']] += 1
        
        top=sorted(guess_counter, key=guess_counter.get, reverse=True)[:1]
        # for i, gue in enumerate(top_five):
        #     if gue == guess:
        #         index = i + 1
        if guess in top:
            # yield ("T5_guess_count",1)
            yield ("Top",guess_counter[guess])
            # yield ("T5_guess", -index)
        else:
            yield ("Top", 0)
            # yield ("T5_guess", 0)
        yield ("different_guesses", len(guess_counter.keys()))
        yield ("total", sum(guess_counter.values()))


class ModFrequencyFeature(Feature):
    
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self):                                

        question_source = "../data/qanta.buzztrain.json.gz"                                                      
        if 'json.gz' in question_source:                                    
            with gzip.open(question_source) as infile:                      
                questions = json.load(infile)                               
        else:                                                               
            with open(question_source) as infile:                           
                questions = json.load(infile)
        for ii in questions:       
            try:
                self.counts[self.normalize(ii["page"].lower())] += 1  
            except:
                pass

    def __call__(self, question, run, guess, guess_history, other_guesses=None): 
        if self.normalize(guess) in self.counts:
            yield ("freq", 1/self.counts[self.normalize(guess)]) ##/sum(self.counts.values()))
            # if self.counts[self.normalize(guess)] > 3:
            #     yield ("freq", -1)
            # else:
            #     yield ("freq", 0)
        else:
            yield ("freq", 0)
        

class LastGuessFeature(Feature):
    def __init__(self, name):
        super().__init__(name)
        self.last_guess = []

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        if len(self.last_guess) > 4:
            self.last_guess.append(guess)
            counts = np.sum(np.array(self.last_guess[-5:-4])==guess)
            if counts > 0:
                yield ("last_guess", 1)
            else:
                yield ("last_guess", 0)
        else:
            self.last_guess.append(guess)
            yield ("last_guess", 0)

class SearchFeature(Feature):
    def __init__(self, name):
        super().__init__(name)
        self.stop_words = set(stopwords.words('english'))
     
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        count=0
        run_tk = set(word for word in run.lower().split() if word not in self.stop_words and word not in string.punctuation)
        guess_tk = set(word for word in guess.lower().split() if word not in self.stop_words and word not in string.punctuation)
        for word in guess_tk:
            if word in run_tk:
                count+=1
        if len(guess_tk) == 0:
            yield ("common", 0)        
        else:
            yield ("common", log(1+count/len(guess_tk)))

class DictionaryFeature(Feature):
    def __init__(self, name):
        self.name = name
        self.vocab = set(words.words())

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        local_count1 = 0 
        local_count2=0
        gs = guess.lower().split()
        rs = run.lower().split()
        for g in gs:
            if g in self.vocab:
                local_count1+=1
        for r in rs:
            if r in self.vocab:
                local_count2+=1
        # yield ("guess", log(1+local_count1))
        # yield ("run", log(1+local_count2))
        # yield ("guess", local_count1/len(gs))
        # yield ("run", local_count2/len(rs))
        if len(gs) == 0:
            yield ("guess", 0)
        else:
            yield ("guess", local_count1/len(gs))
        if len(rs) == 0:
            yield ("run", 0)
        else:
            yield ("run", local_count2/len(rs))

# class POSFeature(Feature):
#     def __init__(self, name):
#         super().__init__(name)
#         self.model = spacy.load("en_core_web_sm")
#     def __call__(self, question, run, guess, guess_history, other_guesses=None):
#         run_key = Counter(word.pos_ for word in self.model(run))
#         guess_key = Counter(word.pos_ for word in self.model(guess))
#         yield ("run_noun", run_key['NOUN'])
#         yield ("run_verb", run_key['VERB'])
#         yield ("guess_noun", guess_key['NOUN'])
#         yield ("guess_verb", guess_key['VERB'])

if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from parameters import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags, guesser_params)
    buzzer = load_buzzer(flags, buzzer_params)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)
