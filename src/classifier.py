
# coding: utf-8

# #### Licenses
# GloVe
# Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/
# 
# Facebookresearch / FastText words embeddings
# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
# 
# @article{bojanowski2016enriching,
#   title={Enriching Word Vectors with Subword Information},
#   author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
#   journal={arXiv preprint arXiv:1607.04606},
#   year={2016}
# }
# 
# License Creative Commons Attribution-Share-Alike License 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)

# In[2]:


import os
import io
import pickle
import numpy as np
from keras.utils import to_categorical

class DataManager:
    
    root_dir_ = '.'
    
    UNKNOWN_WORD = "<UNK>"
    
    
    def __init__(self, root_dir='.', unknown_word='<UNK>'):
        self.root_dir_ = root_dir
        self.UNKNOWN_WORD = unknown_word
        
    def load_dummy_data(self):
        """
        This method makes available some dummy training data.
        """
        X = self.pickle_load(os.path.join('data', '/fr_X.pkl'))
        Y = self.pickle_load("data/fr_Y.pkl")
        vocab_mots = self.pickle_load("data/fr_vocab_mots.pkl")
        vocab_pdd = self.pickle_load("data/fr_vocab_pdd.pkl")
        vocab_liaisons = self.pickle_load("data/fr_vocab_liaisons.pkl")
        return X, Y, vocab_mots, vocab_pdd, vocab_liaisons
        
    def load_dummy_data_2(self):
        """
        This method makes available some dummy training data.
        """
        data = self.pickle_load("data/f1_fr_project_ok_bool.pkl")
        return data['X'], data['Y'], data['vocab_mots'], data['vocab_pdd'], data['vocab_liaisons'] 
    
    def load_data(self, phase='train', lang='fr', featureset='f1'):
        """
        Loads a dataset for a specific lang and feature set, and phase (train/dev/test).
        Note: if unknown word is not present in WORDS or LEMMA vocab, then it is appended.
        
        Parameters
        ----------
        
        phase: str
            'train', 'dev' or 'test'
            
        lang: str
        
        featureset: str
            'f1', 'f2' or 'f3'
        
        """
        name = "{featureset}_{lang}-{phase}".format(lang=lang, featureset=featureset, phase=phase)
        fname = os.path.join(self.root_dir_, 'data', name)
        data = self.pickle_load(fname)
        if data:
            vocabs = {}
            X = np.array(data['X'])
            Y = np.array(data['Y'])
            vocabs['WORDS'] = data['vocab_mots']
            vocabs['POS'] = data['vocab_pdd']
            vocabs['LABELS'] = data['vocab_liaisons']
            if isinstance(vocabs['WORDS'], np.ndarray):
                vocabs['WORDS'] = vocabs['WORDS'].tolist()
            if self.UNKNOWN_WORD not in vocabs['WORDS']:
                vocabs['WORDS'].append(self.UNKNOWN_WORD)
                
            if isinstance(vocabs['POS'], np.ndarray):
                vocabs['POS'] = vocabs['POS'].tolist()
            if isinstance(vocabs['LABELS'], np.ndarray):
                vocabs['LABELS'] = vocabs['LABELS'].tolist()
            if self.UNKNOWN_WORD not in vocabs['LABELS']:
                vocabs['LABELS'].append(self.UNKNOWN_WORD)
                
            if featureset == 'f2' or featureset == 'f3':
                vocabs['MORPHO'] = data['vocab_morpho']
                vocabs['LEMMA'] = data['vocab_lemma']
                if isinstance(vocabs['MORPHO'], np.ndarray):
                    vocabs['MORPHO'] = vocabs['MORPHO'].tolist()
                if self.UNKNOWN_WORD not in vocabs['MORPHO']:
                    vocabs['MORPHO'].append(self.UNKNOWN_WORD)
                if isinstance(vocabs['LEMMA'], np.ndarray):
                    vocabs['LEMMA'] = vocabs['LEMMA'].tolist()
                if self.UNKNOWN_WORD not in vocabs['LEMMA']:
                    vocabs['LEMMA'].append(self.UNKNOWN_WORD)
                          
            print("load_data: loaded X = ", X.shape, ", Y = ", Y.shape, ", vocabs = ", 
                  (''.join("{key} ({len}), ".format(key=k, len=len(vocabs[k])) for k in vocabs.keys())))
            return X, Y, vocabs
        
        return None
    
    def merge_vocabs(self, vocab1, vocab2, data, columns, test_mode=False, verbose=False):
        """
        Merges vocab2 into vocab1, updating accordingly words references in data columns.
        For all words in vocab2 with index idx_vocab2:
        - if it exists in vocab1, then words features of data in columns refering to idx_vocab2 will be replaced by idx_vocab1
        - if it does not exist in vocab1, 
          - If test_mode is False, then it will be appended to vocab1 then replacement will be done as above in data
          - If test_mode is True, then pointers to idx_vocab2 in data columns, will be replaced by pointers to unknown word
            from vocab1 (this may be used to align a test vocab/data to a neural network vocab, as we are not supposed
            to change then neural network vocab once trained on train/dev data)
        When test_mode is True, vocab1 is left unchanged.
        
        Parameters
        ----------
        
        vocab1: list(str)
            A list of words from original vocabulary.
                        
        vocab2: list(str)
            A list of words from vocabulary to be merged into vocab1 (used to build X2).
            
        data: array
            Array containing the words indices to be updated by merge of vocabularies.
            Usually this should be an array with samples as first dimension (rows), then columns for features.

        columns: tuple(int)
            Indices of columns to be updated in data.
            
        test_mode: boolean
            
        Returns:
        
        vocab1: list(str)
            A new vocabulary with missing words appended to vocab1.
            Note: vocab1 is also update in place, meaning this function modified original vocab1.
        
        ! data is updated in place.
        
        """
        
        vocab1_ = vocab1.copy()
        data_ = data.copy()
        
        unknown_idx = -1
        if self.UNKNOWN_WORD in vocab1:
            unknown_idx = vocab1.index(self.UNKNOWN_WORD)
        
        for idx_vocab2, w in enumerate(vocab2):
            # treat vocabs
            if w in vocab1:
                if verbose:
                    print("word [{i}]={w} found in vocab1 at {j}".format(i=idx_vocab2, w=w, j=vocab1.index(w)))
                idx_vocab1 = vocab1.index(w)
            else:
                if verbose:
                    print("word [{i}]={w} not found in vocab1".format(i=idx_vocab2, w=w))
                if not test_mode:
                    vocab1.append(w)
                    idx_vocab1 = len(vocab1) - 1
                elif unknown_idx is not -1:
                    # if in test mode, we associate missing word to "unknown word"
                    idx_vocab1 = unknown_idx
                else:
                    print("merge_vocabs: word [{i}]={w} not found in vocab1 but no unknown word '{unk}' defined in vocab1"
                         .format(i=idx_vocab2, w=w, unk=self.UNKNOWN_WORD))
                
            # replace all references in data
            for i in range(len(data_)):
                if len(data_.shape) == 1:
                    if data_[i] == idx_vocab2:
                        if verbose:
                            print("Replacing word [{i}]= {idx}, {w} index {fro} to {to}"
                                    .format(i=i, idx=idx_vocab2, w=vocab2[idx_vocab2], 
                                            fro=idx_vocab2, to=idx_vocab1))
                        data[i] = idx_vocab1
                        #Replacing word [796]= 1, LEFT_advcl index 1 to 2
                        
                else:
                    for idx, j in enumerate(columns):
                        if data_[i][j] == idx_vocab2:
                            if verbose:
                                print("Replacing word [{i},{j}]= {idx}, {w} index {fro} to {to}"
                                      .format(i=i, j=j, idx=idx_vocab2, w=vocab2[idx_vocab2], 
                                              fro=idx_vocab2, to=idx_vocab1))
                            data[i][j] = idx_vocab1

        return vocab1, data
        

    # some utilities for saving results
    def safe_pickle_dump(self, filename, obj):
        """
        Serializes an object to file, creating directory structure if it does not exist.
        """
        name = filename
        print("pickle.dump "+name)
        try:
            os.makedirs(os.path.dirname(name), exist_ok=True)
            f = open(name,"wb")
            pickle.dump(obj,f)
            f.close()
        except:
            return False
    
        return True
    
    def pickle_load(self, filename):
        """
        Deserialize an object from a file created with pickle.dump.
        Returns False if this failed.
        """
        name = filename
        print("pickle.load "+name)
        try:
            f = open(name,"rb")
            obj = pickle.load(f)
            f.close()
            return obj
        except Exception as e:
            print(e)
            return None
      
        return None
        
    def load_embeddings(self, lang, type='fasttext'):
        """
        Loads an embeddings file depending on its type and language.
        
        Parameters
        ----------
        
        type: str
            Only "fasttext" is supported.
            
        lang: str
            See load_fasttext_embeddings(lang)
        
        """
        
        return self.load_fasttext_embeddings(lang)
        
    def load_fasttext_embeddings(self, lang):
        """
        Loads a fasttext embedding, chosen depending on lang parameter provided.
        File expected as root_dir_/data/embeddings/facebookresearch/wiki.{lang}.vec
        (or as root_dir_/cache/wiki.{lang}.vec.pkl if already loaded once)
        
        Parameters
        ----------
        
        lang: str
            One of 'fr', 'ja', 'en', 'nl' (or additional ones depending on embeddings present on disk).
        
        """
        data_dict = {}
        apax = ""
        
        pickle_fname = "wiki.{lang}.vec.pkl".format(lang=lang)
        pickle_ffname = os.path.join(self.root_dir_, 'cache', pickle_fname)
             
        if os.path.isfile(pickle_ffname):
            data_dict = self.pickle_load(pickle_ffname)
            print("Embedding for {lang} loaded from {fname}".format(lang=lang, fname=pickle_ffname))
        else: 
            fname = "wiki.{lang}.vec".format(lang=lang)
            data_file = os.path.join(self.root_dir_, 'data', 'embeddings', 'facebookresearch', fname)
        
            fin = io.open(data_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
            n, d = map(int, fin.readline().split())
            
            for line in fin:
                tokens = line.rstrip().split(' ')
                data_dict[tokens[0]] = list(map(float, tokens[1:]))
            print("Embedding for {lang} loaded from {fname}".format(lang=lang, fname=data_file))
            # save embeddings as array format to improve speed next time
            self.safe_pickle_dump(pickle_ffname, data_dict)
        
        # adds unknown word if required, using embedding of apax word
        if self.UNKNOWN_WORD not in data_dict.keys():
            apax = list(data_dict.keys())[-1] # python does not guarantee order of dict keys but normally should be ok
            print("debug apax ", apax)
            data_dict[self.UNKNOWN_WORD] = data_dict[apax]

        print("load_fasttext_embeddings: loaded ", len(data_dict), " words vectors, apax '", apax, "'")
        return data_dict, apax
    
    def get_words_to_match_for_embeddings(self, word):
        """
        Returns a list with same word, word lowercased, word lowercased and dashes removed, 
        then all this plus \xa0 removed.
        
        Parameters
        ----------
        
        word: str
            Word to be transformed.
            
        Returns: list
            List of transformed word forms.
        """
        return [word, word.lower(), word.lower().replace('-', ''), word.lower().replace('-', '').replace('\\xa0', ' ')]
    
    def align_embeddings(self, vocab, embeddings, augment_vocab=True, max_vocab_size=-1):
        """
        Generates aligned embeddings from original embeddings, and a vocabulary of words.
        Words from vocabulary may not exist in original embeddings, in this case a random vector is generated.
        Words matching is done as (by priority) : exact match, then case insensitively, then with dash ('-') removed.
        Exception: unknown word is matched with lowest frequency word from embeddings corpus (with fasttext, it is the last
        one from the list).
        
        Parameters
        ----------
        
        vocab: array
            An array containing each word in the vocabulary.
            
        embeddings: dict
            A dict object with words as keys and their embeddings (as a vector array) as values.
            
        augment_vocab: boolean
            If True, then all words from embeddings not existing in vocab, are appended to vocab (up to max_vocab_size).
            
        max_vocab_size: int
            Maximum length of resulting vocab if augment_vocab is True and if max_vocab_size < original vocab length.
            If -1 then there is no limit.
            Note: resulting vocab size can't be < original vocab size, whatever the value of max_vocab_size.
            
        Returns
        -------
        
        aligned_embeddings: list
            An array containing:
          - for words from vocab found in embeddings, the corresponding embedding at same index as in vocab.
          - for words not found in embeddings, a random vector, at same index as in vocab.
          - if augment_vocab is True, all remaining words from embeddings (not found in vocab) are added after
            len(vocab)
        
        words_not_found: list
            An array containing indices (based on vocab) of words not found in embeddings and replaced by random
            values.
            
        words_matched: list
            An array of strings of words based on vocab, as they were matched in embeddings.
            For example if lowercased word from vocab was matched, then lowercase version of this word will be found
            in this table (whereas the original case sensitive word will remain as is in vocab array)
        
        """
        dim_embedding = len(embeddings[list(embeddings.keys())[0]]) # find length from value of 'first' key
        
        print("align_embeddings: aligning embeddings ({elen},{edim}) with vocab ({vlen}) using at most {max}"
             .format(elen=len(embeddings), edim=dim_embedding, vlen=len(vocab), max=max_vocab_size))
        
        cur_size = len(vocab) # to avoid computing vocab len at each iteration
        
        # first append missing embeddings to vocab, if required, to limit unknown words
        if augment_vocab:
            for embedding_word in embeddings.keys():
                # adds missing word only up to max vocab size, if used
                if max_vocab_size is not -1 and cur_size > max_vocab_size:
                    break
                elif embedding_word not in vocab:
                    vocab.append(embedding_word)
                    cur_size = cur_size + 1
#                    if cur_size % 100 == 0:
#                        print("debug embed cur_size ", cur_size)
        
            print("align_embeddings: new augmented vocab size : ", len(vocab))
        
        aligned_embeddings = np.zeros((len(vocab), dim_embedding))
        words_not_found = []
        words_matched = [None] * len(vocab)
        for idx_mot, mot in enumerate(vocab):
            words_to_match = self.get_words_to_match_for_embeddings(mot)
            for word_to_match in words_to_match:
                if word_to_match in embeddings.keys():
                    aligned_embeddings[idx_mot] = embeddings[word_to_match]
                    words_matched[idx_mot] = word_to_match
                    break
            if words_matched[idx_mot] is None:
                words_not_found.append(idx_mot)
                words_matched[idx_mot] = mot
                aligned_embeddings[idx_mot] = np.random.rand(dim_embedding)
        
        print("align_embeddings: new embeddings shape {shap}, words not found {wnf}, words found {wf}"
             .format(shap=aligned_embeddings.shape, wnf=len(words_not_found), wf=len(words_matched)))
        return aligned_embeddings, words_not_found, words_matched


# In[3]:


import os
import pickle
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, Concatenate, Embedding, concatenate, Flatten, Dropout
from keras.engine.input_layer import Input
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

class DependencyClassifier:
    
    UNKNOWN_WORD = '<UNK>' 
    
    MAX_VOCAB_SIZE = -1
    
    models_ = {}
    networks_ = {}
    path_ = '.'
    
    current_model_ = None
    current_network_ = None
    
    embeddings_ = None
    embeddings_for_words_ = None
    embeddings_for_lemmas_ = None
    
    dm_ = DataManager()
    
    def __init__(self, path='.', max_vocab_size=-1, unknown_word='<UNK>'):
        """
        Class to handle neural networks for TAL - TBP AE purpose.
        
        Parameters
        ----------
        
        path: str
            Root path to find files (path/.), scripts (path/.), cache (path/cache), embeddings (path/data/embeddings/...) ...
            Current path by default.
            
        max_vocab_size: int
            Maximum length of a vocabulary - hence for an embedding matrix used in a network.
            This can be set to limit amount of memory used by vocabs and embeddings.
            In practice, original pre-trained embeddings vectors are truncated up to this length, considering
            that embeddings used (fasttext) are ordered from most frequent to least frequent word.
            -1 (default value) means no limitation.
        
        """
        if path != '.':
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path_ = path
        self.MAX_VOCAB_SIZE = max_vocab_size
        self.UNKNOWN_WORD = '<UNK>'
        self.dm_ = DataManager(self.path_)
        
    def get_model(self, model_name):
        """
        Returns an existing TAL model, or None.
        
        Parameters
        ----------
        
        model_name: str
            Name of the model.
        """
        if model_name not in self.models_.keys():
            print("Model {m} not found".format(m=model_name))
            return None
        return self.models_[model_name]
    
    def get_current_model(self):
        return self.models_[self.current_model_]

    def create_model(self, model_name, lang, featureset, use_forms, vocabs, vocabs_dev, vocabs_test, 
                     status=False, test_status=False, embeddings_file=None):
        """
        Creates a new TAL model.
        
        Parameters
        ----------
        
        model_name: str
            Name of this model.
            
        lang: str
            'fr', 'ja', 'nl', 'en'
            
        featureset: str
            'f1', 'f2' or 'f3'
            
        use_forms: boolean
            If true then both words forms are added on top of the features.
            
        vocabs, vocabs_dev, vocabs_test: dict
            Vocabs for learning task, with keys 'WORDS', 'POS', 'MORPHO' and/or 'LEMMA' (or 'LABELS' for targets).
        
        status: boolean
            If a network was prepared, created and trained for this model.
            
        test_status: boolean
            If test results were produced for this model (ie conllu test file)
        
        """
        model = {'name': model_name,
                'lang': lang,
                'featureset': featureset,
                'use_forms': use_forms,
                'vocabs': vocabs,
                'vocabs_dev': vocabs_dev,
                'vocabs_test': vocabs_test,
                'status': status,
                'test_status': test_status,
                'embeddings_file': embeddings_file}
        print("featureset ", featureset)
        #print("create model ", model)
        if model_name not in self.models_.keys():
            self.models_[model_name] = model
        
        self.current_model_ = model_name
        
        return model
            
    def remove_model(self, model_name):
        """
        """
        if model_name in self.models_.keys():
            if self.current_model_ == model_name:
                self.current_model_ = None
            return self.models_.pop(model_name, None)
    
    def save_model(self, model_name):
        """
        """
        if model_name in self.models_:
            self.dm_.safe_pickle_dump(os.path.join(self.path_, model_name + '-model.pkl'), self.models_[model_name])
        else:
            print("Model " + model_name + " not found")
            
    def load_model(self, model_name):
        """
        """
        model_conf = self.dm_.pickle_load(os.path.join(self.path_, model_name + '-model.pkl'))
        if model_conf is not None:
            self.create_model(model_name=model_conf['name'], lang=model_conf['lang'], featureset=model_conf['featureset'], 
                              vocabs=model_conf['vocabs'], vocabs_dev=model_conf['vocabs_dev'],
                              vocabs_test=model_conf['vocabs_test'], use_forms=model_conf['use_forms'], 
                              status=model_conf['status'], test_status=model_conf['test_status'])
            self.current_model_ = model_name
            return True
        return False
            
    def set_model_done(self, model_name):
        if model_name in self.models_:
            self.models_[model_name]['status'] = True
        
    def set_model_undone(self, model_name):
        if model_name in self.models_:
            self.models_[model_name]['status'] = True
            
    def get_model_status(self, model_name):
        if model_name in self.models_:
            return self.models_[model_name]['status']
        else:
            return False

    def set_model_test_done(self, model_name):
        if model_name in self.models_:
            self.models_[model_name]['test_status'] = True
        
    def set_model_test_undone(self, model_name):
        if model_name in self.models_:
            self.models_[model_name]['test_status'] = True
            
    def get_model_test_status(self, model_name):
        if model_name in self.models_:
            return self.models_[model_name]['test_status']
        else:
            return False
        
    def create_network(self, network_name, model_name, nb_classes, dropout=False, hidden_dim=200):
        """
        Creates a keras architecture appropriate for a TAL model.
        Note: a number of parameters comes from the TAL model.
        
        Parameters
        ----------
        
        nb_classes: int
            Number of classes recognized by the network.
            
        dropout: boolean
            Whether to add dropout layers or not.
            
        hidden_dim: int
            The size of the hidden layers.
            (network will consist of a first layer of hidden_dim and a second layer of hidden_dim/2 neurons)
            
        """
        net_model = None
        
        if model_name not in self.models_.keys():
            print("Model {m} not found".format(m=model_name))
            return None
        
        model = self.models_[model_name]
        featureset = model['featureset']
        use_forms = model['use_forms']
        lang = model['lang']
        vocabs = model['vocabs']
        
        net_model = Model()
                       
        #if embeddings != None:
            
        embedsw = self.embeddings_for_words_
        embedsl = self.embeddings_for_lemmas_
        
        if embedsw is not None:
            dim_embeddings = embedsw.shape[1]
        else:
            dim_embeddings = 300
        
        concat_layers = []
        input_layers = []
        
        if use_forms:
            if embedsw is not None:
            
                if len(vocabs['WORDS']) != embedsw.shape[0]:
                    print("Words vocab size {v} must equal embeddings length {e}".format(v=len(vocabs['WORDS']), 
                                                                               e=embedsw.shape[0]))
                    return None
            
                # Pretrained Embedding layer for 2 words
                input_word1 = Input(shape=(1,), dtype='int32', name='word_input_1')
                embeddings_w1 = Embedding(
                    input_dim=len(vocabs['WORDS']), 
                    output_dim=dim_embeddings, 
                    weights=[embedsw], 
                    input_length=1)(input_word1)
                embeddings_w1 = Flatten()(embeddings_w1)
                
                # Embedding layer for second word
                input_word2 = Input(shape=(1,), dtype='int32', name='word_input_2')
                embeddings_w2 = Embedding(input_dim=len(vocabs['WORDS']), 
                                     output_dim=dim_embeddings, 
                                     weights=[embedsw], 
                                     input_length=1)(input_word2)
                #embeddings_2 = embeddings_layer(input_word2) # sharing weights between both words embeddings
                embeddings_w2 = Flatten()(embeddings_w2)
            
            else:
                # Embedding layer for 2 words
                input_word1 = Input(shape=(1,), dtype='int32', name='word_input_1')
                embeddings_w1 = Embedding(
                    input_dim=len(vocabs['WORDS']), 
                    output_dim=dim_embeddings, 
                    input_length=1)(input_word1)
                embeddings_w1 = Flatten()(embeddings_w1)
                
                # Embedding layer for second word
                input_word2 = Input(shape=(1,), dtype='int32', name='word_input_2')
                embeddings_w2 = Embedding(
                    input_dim=len(vocabs['WORDS']), 
                    output_dim=dim_embeddings, 
                    input_length=1)(input_word2)
                embeddings_w2 = Flatten()(embeddings_w2)
                
            concat_layers.append(embeddings_w1)
            concat_layers.append(embeddings_w2)
            input_layers.append(input_word1)
            input_layers.append(input_word2)
            
        if 'LEMMA' in vocabs.keys():
            if featureset is not 'f1' and embedsl is not None:
                # we must define also inputs and embeddings for lemmas
                if len(vocabs['LEMMA']) != embedsl.shape[0]:
                    print("Lemma vocab size {v} must equal embeddings length {e}".format(v=len(vocabs['LEMMA']), 
                                                                                   e=embedsl.shape[0]))
                    return None
            
                # Pretrained Embedding layer for 2 lemmas
                input_lemma1 = Input(shape=(1,), dtype='int32', name='lemma_input_1')
                embeddings_l1 = Embedding(
                    input_dim=len(vocabs['LEMMA']), 
                    output_dim=dim_embeddings, 
                    weights=[embedsl], 
                    input_length=1)(input_lemma1)
                embeddings_l1 = Flatten()(embeddings_l1)
                
                # Embedding layer for second word
                input_lemma2 = Input(shape=(1,), dtype='int32', name='lemma_input_2')
                embeddings_l2 = Embedding(
                    input_dim=len(vocabs['LEMMA']), 
                    output_dim=dim_embeddings, 
                    weights=[embedsl], 
                    input_length=1)(input_lemma2)
                embeddings_l2 = Flatten()(embeddings_l2)
            
                concat_layers.append(embeddings_l1)
                concat_layers.append(embeddings_l2)
                input_layers.append(input_lemma1)
                input_layers.append(input_lemma2)
            
            elif featureset is not 'f1':
            
                # Embedding layer for 2 lemmas
                input_lemma1 = Input(shape=(1,), dtype='int32', name='lemma_input_1')
                embeddings_l1 = Embedding(
                    input_dim=len(vocabs['LEMMA']), 
                    output_dim=dim_embeddings, 
                    input_length=1)(input_lemma1)
                embeddings_l1 = Flatten()(embeddings_l1)
                    
                # Embedding layer for second word
                input_lemma2 = Input(shape=(1,), dtype='int32', name='lemma_input_2')
                embeddings_l2 = Embedding(
                    input_dim=len(vocabs['LEMMA']), 
                    output_dim=dim_embeddings, 
                    input_length=1)(input_lemma2)
                embeddings_l2 = Flatten()(embeddings_l2)  
            
                concat_layers.append(embeddings_l1)
                concat_layers.append(embeddings_l2)
                input_layers.append(input_lemma1)
                input_layers.append(input_lemma2)
            
        
        # define input for additional features
        # note: dist is restricted to [0 ... 7] so 8 values
        if featureset == 'f1':
            """ S.0.POS
                B.0.POS
                DIST"""
            dim_features = len(vocabs['POS']) * 2 + 8
        elif featureset == 'f2':
            """ S.0.POS
                S.0.LEMMA   # embeddings
                S.0.MORPHO
                S.-1.POS
                B.0.POS
                B.0.LEMMA   # embeddings
                B.0.MORPHO
                B.-1.POS
                B.1.POS
                DIST"""
            dim_features = len(vocabs['POS']) * 4  + len(vocabs['MORPHO']) * 2 + 8
        else:
            """ S.0.POS
                S.0.LEMMA   # embeddings
                S.0.MORPHO
                S.-1.POS
                B.0.POS
                B.0.LEMMA   # embeddings
                B.0.MORPHO
                B.-2.POS
                B.-1.POS
                B.1.POS
                DIST"""
            # same size as 'f2'
            dim_features = len(vocabs['POS']) * 5 + len(vocabs['MORPHO']) * 2 + 8
        print("  expecting {n} features dim".format(n=dim_features))
        
        # define input for features
        features_input = Input(shape=(dim_features,))
        concat_layers.append(features_input)
        input_layers.append(features_input)
        
        # concatenate inputs if there's more than one
        if len(concat_layers) > 1:
            l = concatenate(concat_layers)
        else:
            l = features_input
            
        #if embeddings is not None:
            # concatenate features and embeddings
            #l = concatenate([embeddings_1, embeddings_2, features_input])
        #else:
            #l = features_input
        
        # adding dense layers
        
        l = Dense(hidden_dim)(l)
        l = Activation('relu')(l)
        if dropout:
            l = Dropout(0.15)(l)
        l = Dense(int(hidden_dim / 2))(l)
        l = Activation('relu')(l)
        if dropout:
            l = Dropout(0.15)(l)
        
        l = Dense(nb_classes)(l)
        out = Activation('softmax')(l)
        
        if len(input_layers) > 1:
            net_model = Model(input_layers, out)
        else:
            net_model = Model(features_input, out)
        
        # not sure where to compile the model ...
        net_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
        if net_model:
            self.networks_[network_name] = net_model
            # save initial state of this network (this also saves embedding layers, so we remove them)
            self.save_network(network_name)
            #del self.embeddings_for_words_
            #del self.embeddings_for_lemmas_
            self.current_network_ = network_name
        return net_model

    def save_network(self, network_name):
        if network_name in self.networks_:
            self.networks_[network_name].save(os.path.join(self.path_, network_name + '-net.h5'))
        else:
            print("Net " + network_name + " not found")
            
    def load_network(self, network_name):
        """
        Loads a keras network model (.h5, architecture and weights) from disk.
        """
        fname = os.path.join(self.path_, network_name + '-net.h5')
        print("trying to load", fname)
        if os.path.isfile(fname):
            try:
                self.networks_[network_name] = load_model(fname)
                self.current_network_ = network_name
                print("succesfully loaded network ", network_name, "from file", fname)
                return True
            except Exception as e:
                print("Could not load keras model because ", str(e))
                return False
        
        return False
    
    def get_network(self, network_name):
        if network_name in self.networks_:
            return self.networks_[network_name]
        else:
            return None
        
    def get_current_network(self):
        if self.current_network_ in self.networks_:
            return self.networks_[self.current_network_]

    def remove_network(self, network_name):
        """
        Removes a network model from DependencyClassifier (not from disk !), if it exists it is returned by this method.
        """
        if network_name in self.networks_.keys():
            self.current_network_ = None
            return self.networks_.pop(model_name, None)    
    
    def preprocess_embeddings(self, model_name, augment_vocabs=True):
        """
        Preprocess embeddings for training a network for this model.
        Notes:
        - should be run once BEFORE creating a network, if you want to use pre-trained embeddings to train this network, 
          if not embeddings will be completely learned during training
        - loaded embeddings should be deleted before loading/running a new network to avoid exhausting memory.
        - if called again for another model embeddings will be REPLACED
        - for already created network embedded remain and will be saved along with the network (no need to call this if
          you load a saved network)
            - at network creation, once net model is saved to disk with keras model.save(), loaded embeddings are deleted
              to free memory
            - to manually remove loaded embeddings, delete attributes embeddings_, embeddings_for_words_ and embeddings_for_lemmas_
        
        Parameters
        ----------
        
        model_name: str
            Name of the TAL model.
            
        Returns: nothing
        
        """
        
        if model_name not in self.models_.keys():
            print("Model {m} not found".format(m=model_name))
            return None
        model = self.models_[model_name]
        featureset = model['featureset']
        use_forms = model['use_forms']
        lang = model['lang']
        vocabs = model['vocabs']
        
        print("preprocess_embeddings(model",model_name,',augment',augment_vocabs,
              ',f',featureset,',forms',use_forms,',lang',lang,')')
        
        """model f1_fr ,augment True ,f f1 ,forms False ,lang fr )"""
        
        if (self.embeddings_for_words_ is None and use_forms) or (self.embeddings_for_lemmas_ is None and featureset != 'f1'):
            print("self.embeddings_for_words_ is None", self.embeddings_for_words_ is None)
            print("use_forms", use_forms)
            print("self.embeddings_for_lemmas_ is None",self.embeddings_for_lemmas_ is None)
            print("featureset is not 'f1'",featureset is not 'f1')
            print("featureset == 'f1'", featureset == 'f1')
            if self.embeddings_ is None:
                print("preprocess_embeddings: loading original embeddings ...")
                self.embeddings_, self.apax_ = self.dm_.load_embeddings(lang)
        
        if self.embeddings_for_words_ is None and use_forms:
            faligned = os.path.join(self.path_, 'cache', model_name + '-embeddings-forms-aligned.pkl')
            if os.path.isfile(faligned):
                self.embeddings_for_words_ = self.dm_.pickle_load(faligned)
            if self.embeddings_for_words_ is None:
                print("preprocess_embeddings: aligning embeddings for forms ...")
                self.embeddings_for_words_, words_not_found, words_matched = self.dm_.align_embeddings(
                    vocabs['WORDS'], self.embeddings_, augment_vocab=augment_vocabs, max_vocab_size=self.MAX_VOCAB_SIZE)
                self.dm_.safe_pickle_dump(faligned, self.embeddings_for_words_)
                    
        if self.embeddings_for_lemmas_ is None and featureset is not 'f1':
            faligned = os.path.join(self.path_, 'cache', model_name + '-embeddings-lemmas-aligned.pkl')
            if os.path.isfile(faligned):
                self.embeddings_for_lemmas_ = self.dm_.pickle_load(faligned)
            if self.embeddings_for_lemmas_ is None and 'LEMMA' in vocabs.keys():
                print("preprocess_embeddings: aligning embeddings for lemmas ...")
                self.embeddings_for_lemmas_, words_not_found, words_matched = self.dm_.align_embeddings(
                    vocabs['LEMMA'], self.embeddings_, augment_vocab=augment_vocabs, max_vocab_size=self.MAX_VOCAB_SIZE)
                self.dm_.safe_pickle_dump(faligned, self.embeddings_for_lemmas_)
       
        # free some memory - original embeddings should be useless now
        if self.embeddings_ is not None:
            del self.embeddings_
        
        return
    
            
    def preprocess_data(self, model_name, X_train, y_train, X_dev, y_dev, X_test, y_test):
        """
        Preprocess data for training a network for this model.
        - merges vocabs (WORDS) of X_dev into X_train
        - merges vocabs (LEMMA) of X_dev into X_train, if any
        - merges vocabs (MORPHO) of X_dev into X_train, if any
        - merges vocabs (labels) of Y_dev and Y_test into Y_train
        - converts to categorical (one-hot), features of X_* (POS, MORPHO, DIST)
        
        Note: arrays and vocabs are updated in place, so keep a copy of originals if required.
        
        Parameters
        ----------
        
        model_name: str
            TAL model to use for treating data (defines features and vocabs).
            
        X_train, y_train, X_dev, y_dev, X_test, y_test: arrays
            Training, validation and test data to process.
            
        Returns
        -------
        
        X_train, y_train, X_dev, y_dev, X_test, y_test
            
        """
        
        if model_name not in self.models_.keys():
            print("Model {m} not found".format(m=model_name))
            return None
        model = self.models_[model_name]
        featureset = model['featureset']
        use_forms = model['use_forms']
        lang = model['lang']
        vocabs = model['vocabs']
        vocabs_dev = model['vocabs_dev']
        vocabs_test = model['vocabs_test']
        
        # Treat vocabularies
        # Note: X_test/y_test are used only to manually check relevance of the classifier with "fake" arc-eager
        # but they are not used to generate test results
        
        # trim data that was wrongly parsed (non-projective sentences)
        print("preprocess_data: removing non-projective samples ...")
        print("    before ", X_train.shape, y_train.shape)
        X_good = X_train[:, -1] == 1
        X_train = X_train[X_good]
        y_train = y_train[X_good]
        print("    after ", X_train.shape, y_train.shape)
        print("    before ", X_dev.shape, y_dev.shape)
        X_good = X_dev[:, -1] == 1
        X_dev = X_dev[X_good]
        y_dev = y_dev[X_good]
        print("    after ", X_dev.shape, y_dev.shape)
        
        if use_forms:
            print("preprocess_data: merging WORDS vocabs from dev into train ...")
            self.dm_.merge_vocabs(vocab1=vocabs['WORDS'], vocab2=vocabs_dev['WORDS'], data=X_dev, columns=(0, 1))
            # ... vocabs_dev['WORDS'] is now useless
            print("preprocess_data: aligning WORDS vocabs from test into train ...")            
            # handle test set (align vocabs and set unknown words)
            self.dm_.merge_vocabs(vocab1=vocabs['WORDS'], vocab2=vocabs_test['WORDS'], data=X_test, columns=(0, 1),
                                 test_mode=True) # !!!
        
        if 'POS' in vocabs:
            nb_classes_pos = len(np.unique(vocabs['POS']))
        if 'MORPHO' in vocabs:
            print("preprocess_data: merging and aligning MORPHO vocabs from dev into train ...")
            self.dm_.merge_vocabs(vocab1=vocabs['MORPHO'], vocab2=vocabs_dev['MORPHO'], data=X_dev, columns=(4, 8))
            # ... vocabs_dev['MORPHO'] is now useless
            nb_classes_morpho = len(np.unique(vocabs['MORPHO']))
            print("preprocess_data: aligning MORPHO vocabs from test into train ...")
            self.dm_.merge_vocabs(vocab1=vocabs['MORPHO'], vocab2=vocabs_test['MORPHO'], data=X_test, columns=(4, 8),
                                 test_mode=True) # there should be no UNK in this vocab normally !
                                                
        if 'LEMMA' in vocabs:
            print("preprocess_data: merging and aligning LEMMA vocabs from dev into train ...")
            self.dm_.merge_vocabs(vocab1=vocabs['LEMMA'], vocab2=vocabs_dev['LEMMA'], data=X_dev, columns=(3, 7))
            # ... vocabs_dev['LEMMA'] is now useless            
            print("preprocess_data: aligning LEMMA vocabs from test into train ...")
            self.dm_.merge_vocabs(vocab1=vocabs['LEMMA'], vocab2=vocabs_test['LEMMA'], data=X_test, columns=(3, 7),
                                 test_mode=True) # !!!

        print("preprocess_data: merging and aligning LABEL vocabs from dev into train ...")
        self.dm_.merge_vocabs(vocab1=vocabs['LABELS'], vocab2=vocabs_dev['LABELS'], data=y_dev, columns=(0,))
        print("preprocess_data: merging and aligning LABEL vocabs from test into train ...")
        self.dm_.merge_vocabs(vocab1=vocabs['LABELS'], vocab2=vocabs_test['LABELS'], data=y_test, columns=(0,),
                             test_mode=True) # there should be no UNK in this vocab !
        nb_classes = len(vocabs['LABELS'])
            
        # convert some features to one-hot encoding
        
        nb_classes_dist = 8
        
        
        print("preprocess_data: converting {f} features to one-hot encoding...".format(f=featureset))
        
        if featureset == 'f1':
            """
            2 S.0.POS
            3 B.0.POS
            4 DIST"""
            cats_pos1 = to_categorical(X_train[:, 2], num_classes=nb_classes_pos)
            cats_pos2 = to_categorical(X_train[:, 3], num_classes=nb_classes_pos)
            # dist is positive for arc eager
            cats_dist = to_categorical(np.abs(X_train[:, 4]), num_classes=nb_classes_dist)
            if use_forms:
                X_train = np.column_stack((X_train[:, 0], X_train[:, 1], cats_pos1, cats_pos2, cats_dist))
            else:
                X_train = np.column_stack((cats_pos1, cats_pos2, cats_dist))
            
            cats_pos1 = to_categorical(X_dev[:, 2], num_classes=nb_classes_pos)
            cats_pos2 = to_categorical(X_dev[:, 3], num_classes=nb_classes_pos)
            # dist is positive for arc eager
            cats_dist = to_categorical(np.abs(X_dev[:, 4]), num_classes=nb_classes_dist)
            if use_forms:
                X_dev = np.column_stack((X_dev[:, 0], X_dev[:, 1], cats_pos1, cats_pos2, cats_dist))
            else:
                X_dev = np.column_stack((cats_pos1, cats_pos2, cats_dist))
            
            cats_pos1 = to_categorical(X_test[:, 2], num_classes=nb_classes_pos)
            cats_pos2 = to_categorical(X_test[:, 3], num_classes=nb_classes_pos)
            # dist is positive for arc eager
            cats_dist = to_categorical(np.abs(X_test[:, 4]), num_classes=nb_classes_dist)
            if use_forms:
                X_test = np.column_stack((X_test[:, 0], X_test[:, 1], cats_pos1, cats_pos2, cats_dist))
            else:
                X_test = np.column_stack((cats_pos1, cats_pos2, cats_dist))
            
        elif featureset == 'f2': # 'f2' and 'f3' featuresets have same structure
            """ 
            2 S.0.POS
            3 S.0.LEMMA
            4 S.0.MORPHO
            5 S.-1.POS (f2, or S.1.POS for f3)
            6 B.0.POS
            7 B.0.LEMMA
            8 B.0.MORPHO
            9 B.-1.POS
            10 B.1.POS
            11 DIST"""                
            cats_pos1    = to_categorical(X_train[:, 2], num_classes=nb_classes_pos)
            cats_morpho1 = to_categorical(X_train[:, 4], num_classes=nb_classes_morpho)
            cats_pos2    = to_categorical(X_train[:, 5], num_classes=nb_classes_pos)
            cats_pos3    = to_categorical(X_train[:, 6], num_classes=nb_classes_pos)
            cats_morpho2 = to_categorical(X_train[:, 8], num_classes=nb_classes_morpho)
            cats_pos4    = to_categorical(X_train[:, 9], num_classes=nb_classes_pos)
            cats_pos5    = to_categorical(X_train[:, 10], num_classes=nb_classes_pos)
            cats_dist    = to_categorical(np.abs(X_train[:, 11]), num_classes=nb_classes_dist)
            X_train = np.column_stack((X_train[:, 3], X_train[:, 7], cats_pos1, 
                                      cats_morpho1, cats_pos2, cats_pos3,
                                cat_trains_morpho2, cats_pos4, cats_pos5, cats_dist))
            
            cats_pos1    = to_categorical(X_dev[:, 2], num_classes=nb_classes_pos)
            cats_morpho1 = to_categorical(X_dev[:, 4], num_classes=nb_classes_morpho)
            cats_pos2    = to_categorical(X_dev[:, 5], num_classes=nb_classes_pos)
            cats_pos3    = to_categorical(X_dev[:, 6], num_classes=nb_classes_pos)
            cats_morpho2 = to_categorical(X_dev[:, 8], num_classes=nb_classes_morpho)
            cats_pos4    = to_categorical(X_dev[:, 9], num_classes=nb_classes_pos)
            cats_pos5    = to_categorical(X_dev[:, 10], num_classes=nb_classes_pos)
            cats_dist    = to_categorical(np.abs(X_dev[:, 11]), num_classes=nb_classes_dist)
            X_dev = np.column_stack((X_dev[:, 3], X_dev[:, 7], cats_pos1,
                                      cats_morpho1, cats_pos2, cats_pos3,
                               cat_trains_morpho2, cats_pos4, cats_pos5, cats_dist))
            

            cats_pos1    = to_categorical(X_test[:, 2], num_classes=nb_classes_pos)
            cats_morpho1 = to_categorical(X_test[:, 4], num_classes=nb_classes_morpho)
            cats_pos2    = to_categorical(X_test[:, 5], num_classes=nb_classes_pos)
            cats_pos3    = to_categorical(X_test[:, 6], num_classes=nb_classes_pos)
            cats_morpho2 = to_categorical(X_test[:, 8], num_classes=nb_classes_morpho)
            cats_pos4    = to_categorical(X_test[:, 9], num_classes=nb_classes_pos)
            cats_pos5    = to_categorical(X_test[:, 10], num_classes=nb_classes_pos)
            cats_dist    = to_categorical(np.abs(X_test[:, 11]), num_classes=nb_classes_dist)
            X_test = np.column_stack((X_test[:, 3], X_test[:, 7], cats_pos1, 
                                      cats_morpho1, cats_pos2, cats_pos3,
                               cats_morpho2, cats_pos4, cats_pos5, cats_dist))
            
        elif featureset == 'f3':
            """ 
            2 S.0.POS
            3 S.0.LEMMA
            4 S.0.MORPHO
            5 S.-1.POS    (S.1.POS does not exist)
            6 B.0.POS
            7 B.0.LEMMA
            8 B.0.MORPHO
            9 B.-2.POS    (added for f3)
            10 B.-1.POS
            11 B.1.POS
            12 DIST"""                
            cats_pos1    = to_categorical(X_train[:, 2], num_classes=nb_classes_pos)
            cats_morpho1 = to_categorical(X_train[:, 4], num_classes=nb_classes_morpho)
            cats_pos2    = to_categorical(X_train[:, 5], num_classes=nb_classes_pos)
            cats_pos3    = to_categorical(X_train[:, 6], num_classes=nb_classes_pos)
            cats_morpho2 = to_categorical(X_train[:, 8], num_classes=nb_classes_morpho)
            cats_pos4    = to_categorical(X_train[:, 9], num_classes=nb_classes_pos)
            cats_pos5    = to_categorical(X_train[:, 10], num_classes=nb_classes_pos)
            cats_pos6    = to_categorical(X_train[:, 11], num_classes=nb_classes_pos)
            cats_dist    = to_categorical(np.abs(X_train[:, 12]), num_classes=nb_classes_dist)
            X_train = np.column_stack((X_train[:, 3], X_train[:, 7], cats_pos1, cats_morpho1, 
                                       cats_pos2, cats_pos3, cat_trains_morpho2, cats_pos4, 
                                       cats_pos5, cats_pos6, cats_dist))
            
            cats_pos1    = to_categorical(X_dev[:, 2], num_classes=nb_classes_pos)
            cats_morpho1 = to_categorical(X_dev[:, 4], num_classes=nb_classes_morpho)
            cats_pos2    = to_categorical(X_dev[:, 5], num_classes=nb_classes_pos)
            cats_pos3    = to_categorical(X_dev[:, 6], num_classes=nb_classes_pos)
            cats_morpho2 = to_categorical(X_dev[:, 8], num_classes=nb_classes_morpho)
            cats_pos4    = to_categorical(X_dev[:, 9], num_classes=nb_classes_pos)
            cats_pos5    = to_categorical(X_dev[:, 10], num_classes=nb_classes_pos)
            cats_pos6    = to_categorical(X_dev[:, 11], num_classes=nb_classes_pos)
            cats_dist    = to_categorical(np.abs(X_dev[:, 12]), num_classes=nb_classes_dist)
            X_dev = np.column_stack((X_dev[:, 3], X_dev[:, 7], cats_pos1, 
                                      cats_morpho1, cats_pos2, cats_pos3,
                                cat_trains_morpho2, cats_pos4, cats_pos5, cats_pos6, cats_dist))
            

            cats_pos1    = to_categorical(X_test[:, 2], num_classes=nb_classes_pos)
            cats_morpho1 = to_categorical(X_test[:, 4], num_classes=nb_classes_morpho)
            cats_pos2    = to_categorical(X_test[:, 5], num_classes=nb_classes_pos)
            cats_pos3    = to_categorical(X_test[:, 6], num_classes=nb_classes_pos)
            cats_morpho2 = to_categorical(X_test[:, 8], num_classes=nb_classes_morpho)
            cats_pos4    = to_categorical(X_test[:, 9], num_classes=nb_classes_pos)
            cats_pos5    = to_categorical(X_test[:, 10], num_classes=nb_classes_pos)
            cats_pos6    = to_categorical(X_test[:, 11], num_classes=nb_classes_pos)
            cats_dist    = to_categorical(np.abs(X_test[:, 12]), num_classes=nb_classes_dist)
            X_test = np.column_stack((X_test[:, 3], X_test[:, 7], cats_pos1, 
                                      cats_morpho1, cats_pos2, cats_pos3,
                               cats_morpho2, cats_pos4, cats_pos5, cats_pos6, cats_dist))            
                        
        print("preprocess_data: converting LABELs to one-hot encoding ...")
        y_train = to_categorical(y_train, num_classes=nb_classes)
        y_dev = to_categorical(y_dev, num_classes=nb_classes)
        y_test = to_categorical(y_test, num_classes=nb_classes)
                
        return X_train, y_train, X_dev, y_dev, X_test, y_test
    

    def process_test_data(self, X_test):
        """
        Process one sample data in order to fit it to the keras model.
        Format should be a vector with same list of items (POS, MORPHO, LEMMA, optionnally FORM ...), but with their
        VALUES (and not their index in a vocab).
        If there are additional columns in input they will not be used (nor returned).
        
        Parameters
        ----------
        
        X_test: array/list
        
        Returns
        -------
        
        transformed X_test: array/list that can be passed to predict
        
        """
        
        """
        w1[FORM , POS , GOV , LABEL]
        w2[FORM , POS , GOV , LABEL]
        
        f2
        FORM , POS , GOV , LABEL, LEMMA , MORPHO
        """
        
        #print("process_data({data})".format(data=X_test))
        
        model = self.get_current_model()
        featureset = model['featureset']
        use_forms = model['use_forms']
        lang = model['lang']
        vocabs = model['vocabs']
        vocabs_dev = model['vocabs_dev']
        vocabs_test = model['vocabs_test']
        
        nb_classes = len(vocabs['LABELS'])
        nb_classes_pos = len(vocabs['POS'])
        if 'MORPHO' in vocabs.keys():
            nb_classes_morpho = len(vocabs['MORPHO'])
        nb_classes_dist = 8
        
        unknown_idx = -1
        unknown_lemma_idx = -1
        unknown_morpho_idx = -1
        if self.UNKNOWN_WORD in vocabs['WORDS']:
            unknown_idx = vocabs['WORDS'].index(self.UNKNOWN_WORD)
        if 'LEMMA' in vocabs.keys():
            if self.UNKNOWN_WORD in vocabs['LEMMA']:
                unknown_lemma_idx = vocabs['LEMMA'].index(self.UNKNOWN_WORLD)
        if 'MORPHO' in vocabs.keys():
            if self.UNKNOWN_WORD in vocabs['MORPHO']:
                unknown_morpho_idx = vocabs['MORPHO'].index(self.UNKNOWN_WORLD)
        
        # align to vocabs indices (the ones known by the keras network)
        if featureset == 'f1':
            """
            2 S.0.POS
            3 B.0.POS
            4 DIST"""
            cats_pos1 = to_categorical(vocabs['POS'].index(X_test[2]), num_classes=nb_classes_pos)
            cats_pos2 = to_categorical(vocabs['POS'].index(X_test[3]), num_classes=nb_classes_pos)           
            # dist is positive for arc eager
            cats_dist = to_categorical(np.abs(X_test[4]), num_classes=nb_classes_dist)
            
            if use_forms:
                if X_test[0] in vocabs['WORDS']:
                    w1_idx = vocabs['WORDS'].index(X_test[0])
                else:
                    w1_idx = unknown_idx
                if X_test[1] in vocabs['WORDS']:
                    w2_idx = vocabs['WORDS'].index(X_test[1])
                else:
                    w2_idx = unknown_idx
                    
                X_test = np.concatenate((w1_idx, w2_idx, cats_pos1, cats_pos2, cats_dist))
            else:
                X_test = np.concatenate((cats_pos1, cats_pos2, cats_dist))
            
        elif featureset == 'f2':
            """ 
            2 S.0.POS
            3 S.0.LEMMA
            4 S.0.MORPHO
            5 S.-1.POS (f2, or S.1.POS for f3)
            6 B.0.POS
            7 B.0.LEMMA
            8 B.0.MORPHO
            9 B.-1.POS
            10 B.1.POS
            11 DIST"""        
            cats_pos1    = to_categorical(vocabs['POS'].index(X_test[2]), num_classes=nb_classes_pos)
            if X_test[4] in vocabs['MORPHO']:
                m_idx = vocabs['MORPHO'].index(X_test[4])
            else:
                m_idx = unknown_morpho_idx
            cats_morpho1 = to_categorical(m_idx, num_classes=nb_classes_morpho)
            cats_pos2    = to_categorical(vocabs['POS'].index(X_test[5]), num_classes=nb_classes_pos)
            cats_pos3    = to_categorical(vocabs['POS'].index(X_test[6]), num_classes=nb_classes_pos)
            if X_test[8] in vocabs['MORPHO']:
                m_idx = vocabs['MORPHO'].index(X_test[8])
            else:
                m_idx = unknown_morpho_idx            
            cats_morpho2 = to_categorical(m_idx, num_classes=nb_classes_morpho)
            cats_pos4    = to_categorical(vocabs['POS'].index(X_test[9]), num_classes=nb_classes_pos)
            cats_pos5    = to_categorical(vocabs['POS'].index(X_test[10]), num_classes=nb_classes_pos)
            cats_dist    = to_categorical(np.abs(X_train[11]), num_classes=nb_classes_dist)
            if X_test[3] in vocabs['LEMMA']:
                l1_idx = vocabs['LEMMA'].index(X_test[3])
            else:
                l1_idx = unknown_lemma_idx
            if X_test[7] in vocabs['LEMMA']:
                l2_idx = vocabs['LEMMA'].index(X_test[7])
            else:
                l2_idx = unknown_lemma_idx
            # lemmas (embeddings) are set at beginning of inputs
            X_test = np.concatenate((l1_idx, l2_idx, cats_pos1, cats_morpho1, cats_pos2, cats_pos3,
                                       cat_morpho2, cats_pos4, cats_pos5, cats_dist))
            
        elif featureset == 'f3':
            """ 
            2 S.0.POS
            3 S.0.LEMMA
            4 S.0.MORPHO
            5 S.-1.POS    (S.1.POS does not exist)
            6 B.0.POS
            7 B.0.LEMMA
            8 B.0.MORPHO
            9 B.-2.POS    (added for f3)
            10 B.-1.POS
            11 B.1.POS
            12 DIST"""  
            cats_pos1    = to_categorical(vocabs['POS'].index(X_test[2]), num_classes=nb_classes_pos)
            if X_test[4] in vocabs['MORPHO']:
                m_idx = vocabs['MORPHO'].index(X_test[4])
            else:
                m_idx = unknown_morpho_idx
            cats_morpho1 = to_categorical(m_idx, num_classes=nb_classes_morpho)
            cats_pos2    = to_categorical(vocabs['POS'].index(X_test[5]), num_classes=nb_classes_pos)
            cats_pos3    = to_categorical(vocabs['POS'].index(X_test[6]), num_classes=nb_classes_pos)
            if X_test[8] in vocabs['MORPHO']:
                m_idx = vocabs['MORPHO'].index(X_test[8])
            else:
                m_idx = unknown_morpho_idx            
            cats_morpho2 = to_categorical(m_idx, num_classes=nb_classes_morpho)
            cats_pos4    = to_categorical(vocabs['POS'].index(X_test[9]), num_classes=nb_classes_pos)
            cats_pos5    = to_categorical(vocabs['POS'].index(X_test[10]), num_classes=nb_classes_pos)
            cats_pos6    = to_categorical(vocabs['POS'].index(X_test[11]), num_classes=nb_classes_pos)
            cats_dist    = to_categorical(np.abs(X_train[12]), num_classes=nb_classes_dist)
            if X_test[3] in vocabs['LEMMA']:
                l1_idx = vocabs['LEMMA'].index(X_test[3])
            else:
                l1_idx = unknown_lemma_idx
            if X_test[7] in vocabs['LEMMA']:
                l2_idx = vocabs['LEMMA'].index(X_test[7])
            else:
                l2_idx = unknown_lemma_idx
            # lemmas (embeddings) are set at beginning of inputs
            X_test = np.concatenate((l1_idx, l2_idx, cats_pos1, cats_morpho1, cats_pos2, cats_pos3,
                                       cat_morpho2, cats_pos4, cats_pos5, cats_pos6, cats_dist))            
            
        X_test = X_test.reshape((1, len(X_test)))
        #print("process_test_data ->", X_test)
        return X_test
    
    def get_label(self, y_pred):
        y_pred_idx = np.argmax(y_pred)
        return self.get_current_model()['vocabs']['LABELS'][y_pred_idx]
    
    def print_data(self, model_name, X_, Y, idx, vocab_type):
        if model_name not in self.models_.keys():
            print("Model {m} not found".format(m=model_name))
            return None
        
        model = self.models_[model_name]
        featureset = model['featureset']
        lang = model['lang']
        vocabs = model[vocab_type]
        
        X = X_[idx]
        
        print("feat", featureset)
            
        if featureset == 'f1':
            print(vocabs['WORDS'][X[0]], vocabs['WORDS'][X[1]], 
                  vocabs['POS'][X[2]], vocabs['POS'][X[3]], X[4], '==>', vocabs['LABELS'][Y[idx]])

        elif featureset == 'f2':
            print(vocabs['WORDS'][X[0]], vocabs['WORDS'][X[1]], 
                  vocabs['POS'][X[2]], vocabs['LEMMA'][X[3]], vocabs['MORPHO'][X[4]],
                  vocabs['POS'][X[5]], vocabs['POS'][X[6]] ,vocabs['LEMMA'][X[7]], vocabs['MORPHO'][X[8]],
                  vocabs['POS'][X[9]], vocabs['POS'][X[10]],  X[11], '==>', vocabs['LABELS'][Y[idx]])
        elif featureset == 'f3':
            print(vocabs['WORDS'][X[0]], vocabs['WORDS'][X[1]], 
                  vocabs['POS'][X[2]], vocabs['LEMMA'][X[3]], vocabs['MORPHO'][X[4]],
                  vocabs['POS'][X[5]], vocabs['POS'][X[6]] ,vocabs['LEMMA'][X[7]], vocabs['MORPHO'][X[8]],
                  vocabs['POS'][X[9]], vocabs['POS'][X[10]], vocabs['POS'][X[11]], X[11], '==>', vocabs['LABELS'][Y[idx]])

            
    def print_status(self):
        print()
        print("=== DependencyClassifier = max_vocab_size {max} unknown form {unk} ===".format(max=self.MAX_VOCAB_SIZE,
                                                                                             unk=self.UNKNOWN_WORD))
        for model in self.models_:
            print("   {name} -- {mod}".format(name=model, mod=["{k}:{v},".format(k=k,v=v) for k,v in self.models_[model].items()]))
        for net in self.networks_:
            print(net)
            
        print()
    


# In[4]:


import time
import generate_data as aegen


def main(epochs=10, max_vocab_size=100000, unknown_word='<UNK>'):

    dm = DataManager('../')
    # define a manager with hard limit of 100000 for vocabularies lengths
    dep_classifier = DependencyClassifier(path='../', max_vocab_size=100000, unknown_word='<UNK>')
    
    test_info = {'fr': os.path.join("..", "UD_French-GSD", "fr_gsd-ud-test.conllu"),       
                 'nl' : os.path.join("..", "UD_Dutch-LassySmall", "nl_lassysmall-ud-test.conllu") ,              
                 'en' : os.path.join("..", "UD_English-LinES", "en_lines-ud-test.conllu") ,                              
                 'ja' : os.path.join("..", "UD_Japanese-GSD", "ja_gsd-ud-test.conllu")
                }

    for lang in ['fr', 'nl', 'en', 'ja']:
    
        for featureset in ['f1', 'f1-forms', 'f2', 'f3']:
            
            test_results_fname = "{featureset}_{lang}_results.conllu".format(featureset=featureset, lang=lang)
    
            if os.path.isfile(test_results_fname):
                print("{f} already exists, skipping this task".format(f=test_results_fname))
                break
    
            # === CREATE NEURAL NETWORK ===
            print()
            print("========= {lang} === {featureset} ==========".format(lang=lang, featureset=featureset))
            print()
            print("===== CREATING NEURAL NETWORK =====")
            print()
            t = time.time()
            print("= Loading data...")
            feat = featureset
            if featureset == 'f1-forms':
                feat = 'f1'
            X_train, y_train, vocabs_train = dm.load_data('train', lang, feat)
            X_dev, y_dev, vocabs_dev = dm.load_data('dev', lang, feat)
            X_test, y_test, vocabs_test = dm.load_data('test', lang, feat)

            print("  ... loaded data in ", time.time() - t)

            nb_classes = len(vocabs_train['LABELS'])

            model_name = "{featureset}_{lang}".format(featureset=featureset, lang=lang)
            
            if featureset is 'f1-forms':
                model_name = model_name + '-forms'

            # create a TAL model for this combination
            exists = dep_classifier.load_model(model_name)
            if exists and dep_classifier.get_model_status(model_name):
                print("==> Neural network for this model already prepared")
            else:
                dep_classifier.create_model(model_name, lang, feat, featureset is 'f1-forms', 
                                            vocabs_train, vocabs_dev, vocabs_test)
                dep_classifier.save_model(model_name)

            if dep_classifier.get_model_test_status(model_name):
                print("==> Test conllu file already generated, skipping...")
                break
                
            t = time.time()

            print("= Pre-processing data ...")
            # preprocess X/Y data
            """mod = dep_classifier.get_current_model()
            idx_test = 40
            w = mod['vocabs_test']['WORDS'][X_test[idx_test][0]]
            print("test sample ", idx_test, 'id ', X_test[idx_test][0], '=', w)
            idx = mod['vocabs']['WORDS'].index(w)
            for x in range(len(X_train)):
                if X_train[x][0] == idx:
                    idx_train = x
                    break
            idx = mod['vocabs_dev']['WORDS'].index(w)
            for x in range(len(X_dev)):
                if X_dev[x][0] == idx:
                    idx_dev = x
                    break
            print("x_train", X_train[idx_train])
            print("y_train", y_train[idx_train])
            print("x_dev", X_dev[idx_dev])
            print("y_dev", y_dev[idx_dev])
            print("x_test", X_test[idx_test])
            print("y_test", y_test[idx_test])
            dep_classifier.print_data(model_name, X_train, y_train, idx_train, 'vocabs')
            dep_classifier.print_data(model_name, X_dev, y_dev, idx_dev, 'vocabs_dev')
            dep_classifier.print_data(model_name, X_test, y_test, idx_test, 'vocabs_test')"""
            X_train, y_train, X_dev, y_dev, X_test, y_test = dep_classifier.preprocess_data(model_name, 
                                                                                X_train, y_train, 
                                                                                X_dev, y_dev, 
                                                                                X_test, y_test)

            print("  ... preprocessed data in ", time.time() - t)
            """print("x_train", X_train[idx_train])
            print("y_train", y_train[idx_train])
            print("x_dev", X_dev[idx_dev])
            print("y_dev", y_dev[idx_dev])
            print("x_test", X_test[idx_test])
            print("y_test", y_test[idx_test])   """     
            t = time.time()
            print("= Preprocessing embeddings ...")
            # load pretrained embeddings
            # embeddings have already been processed
            dep_classifier.preprocess_embeddings(model_name, augment_vocabs=True)

            print("  ... preprocessed embeddings in ", time.time() - t)

            t = time.time()
            print("= Creating neural network architecture ...")
            # create a classifier for this TAL model
            network_name = "{model}".format(model=model_name)
            existsnet = dep_classifier.load_network(network_name)
            net = None
            if not existsnet:
                dep_classifier.create_network(network_name, model_name, nb_classes=nb_classes, dropout=True)
                if net is None:
                    print("  ==> could not create network architecture for this model !!!")
            else:
                print("network architecture already exists, loaded from disk")
            net = dep_classifier.get_network(network_name)
            
            print()
            print(net.summary())
            print()

            print("  ... created net in ", time.time() - t)

            #dep_classifier.print_status()
            
            print()
            print("===== TRAINING NEURAL NETWORK =====")
            print()           
        
        
            #@TODO currently if recalled network will be trained again for epochs if it already exists
            print("= Training for {epochs} epochs ...".format(epochs=epochs))
            inputs = []
            inputs_dev = []
            inputs_test = []
            if featureset == 'f1':
                inputs = [X_train]
                inputs_dev = [X_dev]
                inputs_test = [X_test]
            else:
                inputs = [X_train[:, 0], X_train[:, 1], X_train[:, 2:X_train.shape[1]]]
                inputs_dev = [X_dev[:, 0], X_dev[:, 1], X_dev[:, 2:X_dev.shape[1]]]
                inputs_test = [X_test[:, 0], X_test[:, 1], X_test[:, 2:X_test.shape[1]]]
                
            if net is not None:
                history = net.fit(inputs, y_train, validation_data=(inputs_dev, y_dev), batch_size=128, epochs=epochs)
            else:
                print("problem : net is none")
            
            # save the network
            print("= Saving network architecture ...")
            dep_classifier.save_network(network_name)
            
            dep_classifier.set_model_done(model_name)
            dep_classifier.save_model(model_name)            
            
        
            #@TODO plot history ?
            print("= Evaluating ...")
            results = net.evaluate(inputs_test, y_test, batch_size=128)
            print("  ==> (temporary) results for {lang} / {featureset}: ", results)
            
            print()
            print("===== GENERATING TEST RESULTS =====")
            #STUB(dep_classifier)
            
            print("= Calling Arc Eager with new oracle ...")
            if not dep_classifier.get_model_test_status(model_name):
                result_path = os.path.dirname(os.path.abspath(test_info[lang]))
                result_name = os.path.join(result_path, test_results_fname)
                aegen.create_conllu(test_info[lang], feat, result_name, oracle_=dep_classifier)
            
                print("= Setting model as fully done ...")
                # tag this combination as done and save it
                dep_classifier.set_model_test_done(model_name)
                dep_classifier.save_model(model_name)
                
            #dep_classifier.remove_network(network_name)
            #dep_classifier.remove_model(model_name)

main(epochs=1)


# In[80]:


import sys
if True:
    sys.exit(0)
lang='fr'
featureset='f1-forms'

dm = DataManager('../')
# define a manager with hard limit of 100000 for vocabularies lengths
dep_classifier = DependencyClassifier(path='../', max_vocab_size=100000, unknown_word='<UNK>')

print("= Loading data...")
feat = featureset
if featureset == 'f1-forms':
    feat = 'f1'
X_train, y_train, vocabs_train = dm.load_data('train', lang, feat)
X_dev, y_dev, vocabs_dev = dm.load_data('dev', lang, feat)
X_test, y_test, vocabs_test = dm.load_data('test', lang, feat)

nb_classes = len(vocabs_train['LABELS'])

model_name = "{featureset}_{lang}".format(featureset=featureset, lang=lang)

if featureset is 'f1-forms':
    model_name = model_name + '-forms'

# create a TAL model for this combination
exists = dep_classifier.load_model(model_name)
if exists and dep_classifier.get_model_status(model_name):
    print("==> Neural network for this model already prepared")
else:
    print("feat create ", feat)
    dep_classifier.create_model(model_name, lang, feat, featureset == 'f1-forms', 
                                vocabs_train, vocabs_dev, vocabs_test)

    
print("= Pre-processing data ...")
# preprocess X/Y data
mod = dep_classifier.get_current_model()
idx_test = 40
w = mod['vocabs_test']['WORDS'][X_test[idx_test][0]]
print("test sample ", idx_test, 'id ', X_test[idx_test][0], '=', w)
idx = mod['vocabs']['WORDS'].index(w)
for x in range(len(X_train)):
    if X_train[x][0] == idx:
        idx_train = x
        break
idx = mod['vocabs_dev']['WORDS'].index(w)
print (idx)
for x in range(len(X_dev)):
    if X_train[x][0] == idx:
        idx_dev = x
        break
print("x_train", X_train[idx_train])
print("y_train", y_train[idx_train])
print("x_dev", X_dev[idx_dev])
print("y_dev", y_dev[idx_dev])
print("x_test", X_test[idx_test])
print("y_test", y_test[idx_test])
dep_classifier.print_data(model_name, X_train, y_train, idx_train, 'vocabs')
dep_classifier.print_data(model_name, X_dev, y_dev, idx_dev, 'vocabs_dev')
dep_classifier.print_data(model_name, X_test, y_test, idx_test, 'vocabs_test')
X_train, y_train, X_dev, y_dev, X_test, y_test = dep_classifier.preprocess_data(model_name, 
                                                                    X_train, y_train, 
                                                                    X_dev, y_dev, 
                                                                    X_test, y_test)

print("x_train", X_train[idx_train])
print("y_train", y_train[idx_train])
print("x_dev", X_dev[idx_dev])
print("y_dev", y_dev[idx_dev])
print("x_test", X_test[idx_test])
print("y_test", y_test[idx_test])      


# In[82]:


toto = 'f1'
print(toto == 'f1')
print (toto is 'f1')
print(toto is not 'f1')
print (toto != 'f1')


# - il n'y a pas les mots dans les features (seulement les lemmes)
# 
# - ajouter les mots uniquement pour l'expérience en plus
# 
# - embeddings pour les lemmes: il faudrait ajouter tous les mots des embeddings ! (et pas les enlever lors de l'alignement)
# 
# 
