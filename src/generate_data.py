# -*- coding: utf-8 -*-
import copy
from Dicos import *
from Arc_eager import *
import numpy as np
def get_data(filename,feature):
  """

  :param filename: Nom du fichier conllu
  :param feature: feature utilisé
  :return: X , Y
  """
  try:
    conlluFile = open(filename, encoding='utf-8')
  except IOError:
    print('cannot open', conlluFilename)
    exit(1)

  phrase = [] 
  X = []
  Y = []
  
  go = 0
  fa = 0

  count_proj = 0
  index = 0
  for ligne in conlluFile:
    if ligne[0] == '\n': #Nouvelle phrase
      index = 0
      pa = copy.copy(phrase)
      
      transition = get_couple(phrase)
      proj = is_proj(transition)
      #print("proj" if proj else "non proj")
      
      if proj : count_proj += 1 
      
      couple = verif_arcs(phrase,feature)
      
      arcs,x,y = parser(phrase,feature,proj)
      if proj :
        for z in x:
          z += [1]
      else: 
        for z in x:
          z += [0]
      X += x
      Y += y

      if ( not verif_couple(couple,arcs) ):
        fa += 1
      else :
        go += 1

   
      phrase = []
      
      #sys.exit("Error message")
    
    if ligne[0] != '\n' and ligne[0] != '#' :
      tokens = ligne.split("\t")
      if ( "-" not in tokens[0]):
        if ( feature == "f1"):
        # FORM , POS , GOV , LABEL
          phrase.append([tokens[1],tokens[3],tokens[6],tokens[7],index])
        elif ( feature == "f2" or feature == "f3"):
        # FORM , POS , GOV , LABEL, LEMMA , MORPHO
          phrase.append([tokens[1],tokens[3],tokens[6],tokens[7],tokens[2],tokens[5],index])
        index += 1
       
  print("True  : ",go , " / False : " , fa)
  print("Proj : ", count_proj)
  return X,Y


def create_conllu(filename, feature,result_name,oracle=None):
  """

  :param filename: Nom du fichier conllu
  :param feature:  Feature utilisé
  :param result_name: Nom du fichier conllu resultant
  :param oracle_: Utilisation du reseau ou non
  :return: Créer le fichier conllu ( result_name )
  """
  try:
    conlluFile = open(filename, encoding='utf-8')
  except IOError:
    print('cannot open', filename)
    exit(1)

  result_conllu = open(result_name, "w",encoding='utf-8')
  phrase = []
  phrase_all = []
  X = []
  Y = []

  transition = get_couple(phrase)
  proj = is_proj(transition)
  go = 0
  fa = 0

  count_proj = 0
  index = 0


  for ligne in conlluFile:
    if ligne[0] == '\n':  # Nouvelle phrase
      index = 0

      arcs, x, y ,phrase_all = parser(phrase, feature,proj,phrase_all=phrase_all,oracle=oracle)

      for tok in phrase_all:
          result_conllu.write("\t".join(str(x) for x in tok))


      result_conllu.write("\n")


      phrase = []
      phrase_all = []
    if ligne[0] == '#':

        #print(str(ligne))

        result_conllu.write(ligne)
    if ligne[0] != '\n' and ligne[0] != '#':
      tokens = ligne.split("\t")

      if ("-" not in tokens[0]):
        if (feature == "f1"):
          # FORM , POS , GOV , LABEL
          phrase.append([tokens[1], tokens[3], tokens[6], tokens[7], index])
        elif (feature == "f2" or feature == "f3"):
          # FORM , POS , GOV , LABEL, LEMMA , MORPHO
          phrase.append([tokens[1], tokens[3], tokens[6], tokens[7], tokens[2], tokens[5], index])

        index += 1
      phrase_all.append(tokens)

  result_conllu.close()




def get_unique(X,Y):
  b = []
  for i in range(len(X)-1):
    b.append(X[i] + [Y[i]])


  c = np.unique(b, axis=0)
  X = []
  Y = []

  for j in range(len(c)-1):

    X.append([c[j][0],c[j][1],c[j][2],c[j][3],c[j][4]])
    Y.append(c[j][5])
    
  return X,Y
def verif_arcs(phrase,features):
  couple = []
  for i in range(len(phrase)):
    
    if ( phrase[i][3] == "root" ):
      if (features == "f1"):
        couple.append((["Root","Root",0,"Root"],phrase[i]))
      else : 
        couple.append((["Root","Root",0,"Root","Root","Root"],phrase[i]))
    else :
      couple.append((phrase[int(phrase[i][2])-1],phrase[i]))
 
  return couple

def verif_couple(couple,arcs):

  for i in range(len(couple)):
    if (couple[i] not in arcs):
      return False
  
  return True




def convert_x_index(x,vocab,pdd,lemma,morpho,feature):
  if (feature == "f1"):
    for i in range(len(x)):

      x[i][0] = vocab.index(x[i][0])
      x[i][1] = vocab.index(x[i][1])
      x[i][2] = pdd.index(x[i][2])
      x[i][3] = pdd.index(x[i][3])
      x[i][4] = int(x[i][4])
  elif feature == "f2":
    for i in range(len(x)):
      x[i][0] = vocab.index(x[i][0])
      x[i][1] = vocab.index(x[i][1])
      x[i][2] = pdd.index(x[i][2])
      x[i][3] = lemma.index(x[i][3])
      x[i][4] = morpho.index(x[i][4])
      x[i][5] = pdd.index(x[i][5])
      x[i][6] = pdd.index(x[i][6])
      x[i][7] = lemma.index(x[i][7])
      x[i][8] = morpho.index(x[i][8])
      x[i][9] = pdd.index(x[i][9])
      x[i][10] = pdd.index(x[i][10])
      x[i][11] = int(x[i][11])
  elif feature == "f3":
    for i in range(len(x)):
      x[i][0] = vocab.index(x[i][0])
      x[i][1] = vocab.index(x[i][1])
      x[i][2] = pdd.index(x[i][2])
      x[i][3] = lemma.index(x[i][3])
      x[i][4] = morpho.index(x[i][4])
      x[i][5] = pdd.index(x[i][5])
      x[i][6] = pdd.index(x[i][6])
      x[i][7] = lemma.index(x[i][7])
      x[i][8] = morpho.index(x[i][8])
      x[i][9] = pdd.index(x[i][9])
      x[i][10] = pdd.index(x[i][10])
      x[i][11] = pdd.index(x[i][11])
      x[i][12] = int(x[i][12])
  return x

def convert_y_index(y,vocab):
  for i in range(len(y)):
    y[i] = vocab.index(y[i])
    
  return y

def get_all_data(filename,feature="f1"):
  # Input : conll file
  # Output : X  = [ mot1 , mot2 , pdd1 , pdd2, dist]
  #          Y  = [Liaison mots1 - mot2]
  #          mots = vocabulaires des mots
  #          pdd  = vocabulaires des parties de discours
  #          lemma = vocabulaire des lemmes
  #          morpho = vocabulaire des morphos
  #          vocab_y = vocabulaire des liaisons
  mots, pdd,lemma,morpho = create_vocab(filename)
  X,Y = get_data(filename,feature)
  vocab_y = np.unique(Y)
  X_unique_index = convert_x_index(X,mots,pdd,lemma,morpho,feature)
  Y_unique_index = convert_y_index(Y,vocab_y.tolist())
  
  if (feature == "f1"):
    return X_unique_index, Y_unique_index , mots, pdd, vocab_y
  if  (feature == "f2" or feature == "f3"):
    return X_unique_index, Y_unique_index , mots, pdd, lemma, morpho, vocab_y

def create_vocab(filename):
  mcd = (('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))


  dicos = Dicos(mcd)
  mots,pdd,lemma,morpho = dicos.populateFromConlluFile(filename, verbose=False)
  dicos.lock()
  mots += ["Root"] + ["N..U..L..L"]
  pdd += ["Root"] + ["N..U..L..L"]
  lemma += ["Root"] + ["N..U..L..L"]
  morpho += ["Root"] + ["N..U..L..L"]
  
  return mots,pdd,lemma,morpho