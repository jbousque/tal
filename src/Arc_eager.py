from random import randint
import numpy as np
from generate_data import *


def left_arc(stack, buffer, arcs, tab_head, index_phrase, needs):
    arcs.append((buffer[0], stack[len(stack) - 1]))

    stack.pop(len(stack) - 1)
    return stack, buffer, arcs


def right_arc(stack, buffer, arcs, tab_head, index_phrase, needs):
    arcs.append((stack[len(stack) - 1], buffer[0]))
    word = buffer.pop(0)  # Retire le premier mot du buffer
    stack.append(word)

    return stack, buffer, arcs


def shift(stack, buffer):
    word = buffer.pop(0)
    stack.append(word)
    return stack, buffer


def reduce(stack):
    stack.pop(len(stack) - 1)
    return stack


def oracle(w1, w2, phrase, couples,tab_head):
    """

    :param w1: Mot du stack
    :param w2: mot du buffer
    :param phrase: Liste de mots
    :param couples: Liste de couple ( x , y )
    :return:
    """

    if (int(w2[2]) == phrase.index(w1)):
        return "RIGHT_" + str(w2[3]), 2

    elif (int(w1[2]) == phrase.index(w2)):

        return "LEFT_" + str(w1[3]), 1

    elif (w1 in tab_head):
        if (verif_reduce(phrase.index(w1), phrase.index(w2), couples, len(phrase) - 1)):
            return "REDUCE", 4
        else:
            return "SHIFT", 3
    else:

        return "SHIFT", 3


def verif_reduce(index, index2, couples, taille):  ## A OPTIMISER

    possible = []
    for i in range(index2, taille + 1):
        possible.append((index, i))
    for (a, b) in possible:
        if (a, b) in couples: return False

    return True


def listing(sentence):
    res = []
    for i in range(len(sentence)):
        res.append(i + 1)
    return res


def print_result(res, sentence):
    sentence = ["root"] + sentence
    for i in range(len(res)):
        w1, w2 = res[i]
        print(sentence[w1], ' -> ', sentence[w2])


def get_need(phrase):
    need = []
    [need.append(i[2]) for i in phrase]
    return need


def parser(phrase, feature,proj,phrase_all=None,oracle_=None):
    """

    :param phrase: Liste de mots sous forme de tokens ( Spécifique au features )
    :param feature: Feature utilisé ( f1 | f2 | f3 )
    :param phrase_all:  Liste de mots sous forme de tokens ( Tout les tokens )
    :param oracle_: Utilisation du réseau ou non
    :return: arcs : Liste des arcs créée par le parser
    :return: X : Liste des X ( Spécifique au features )
    :return: Y : Liste des Y
    :return: phrase_all : Nouvelle liste de mots avec modifications du GOV et LABEL
    """
    couples = get_couple(phrase)
    tab_head = []
    buffer = phrase
    needs = get_need(phrase)
    if (feature == "f1"):
        root = ["Root", "Root", 0, "Root"]
    elif (feature == "f2" or feature == "f3"):
        root = ["Root", "Root", 0, "Root", "Root", "Root"]
    phrase = [root] + phrase
    index_phrase = [i[0] for i in phrase]
    arcs = []
    stack = []
    stack.append(root)  ## root
    if phrase_all != None :
        for i in range(len(phrase_all)):
            if ("-" not in phrase_all[i][0]):
                phrase_all[i][6] = 0  # GOV
                phrase_all[i][7] = "_"  # Lab
    X = []
    Y = []

    # Rajout du premier SHIFT
    mot_null = "N..U..L..L"
    root = "Root"
    if (feature == "f1"):
        X.append([mot_null,  # FORM mot 1
                  root,  # FORM mot 2
                  mot_null,  # POS mot 1
                  root, 0])  # FORM mot 2
    if (feature == "f2"):
        X.append([mot_null,  # FORM mot 1
                  root,  # FORM mot 2
                  mot_null,  # POS mot 1
                  mot_null,  # LEMMA mot 1
                  mot_null,  # MORPHO mot 1
                  mot_null,  # POS mot 1
                  root,  # POS mot 2
                  root,  # LEMMA mot 2
                  root,  # MORPHO mot 2
                  mot_null,
                  buffer[0][1],
                  0])
    if (feature == "f3"):
        X.append([mot_null,  # FORM mot 1
                  root,  # FORM mot 2
                  mot_null,  # POS mot 1
                  mot_null,  # LEMMA mot 1
                  mot_null,  # MORPHO mot 1
                  mot_null,  # POS mot 1
                  root,  # POS mot 2
                  root,  # LEMMA mot 2
                  root,  # MORPHO mot 2
                  mot_null,
                  mot_null,
                  buffer[0][1],
                  0])
    Y.append("SHIFT")

    # Début du parcours du buffer / stack
    stack_done = []
    buffer_done = []

    stack_before_pos = "N..U..L..L"
    buffer_before_pos = "N..U..L..L"
    buffer_after_pos = "N..U..L..L"
    buffer_before_before_pos = "N..U..L..L"
    while ((len(buffer) != 0) and (len(stack) != 0)):

        if (stack[len(stack) - 1] == root):
            dist = abs(0 - phrase.index(buffer[0]))
        elif (buffer[0] == root):
            dist = abs(phrase.index(stack[len(stack) - 1]))
        else:
            dist = abs(phrase.index(stack[len(stack) - 1]) - phrase.index(buffer[0]))

        if dist > 7: dist = 7

        if (feature == "f1"):
            X.append([stack[len(stack) - 1][0],  # FORM mot 1
                      buffer[0][0],  # FORM mot 2
                      stack[len(stack) - 1][1],  # POS mot 1
                      buffer[0][1], dist])  # POS mot 2
        if (feature == "f2"):

            if (len(stack_done) >= 1):
                stack_before_pos = stack_done[len(stack_done) - 1]
            else:
                stack_before_pos = "N..U..L..L"

            stack_done.append(stack[len(stack) - 1][1])

            if (len(buffer_done) >= 1):
                buffer_before_pos = buffer_done[len(buffer_done) - 1]
            else:
                buffer_before_pos = "N..U..L..L"

            buffer_done.append(buffer[0][1])

            if (len(buffer) >= 2):
                buffer_after_pos = buffer[1][1]
            else:
                buffer_after_pos = "N..U..L..L"

            X.append([stack[len(stack) - 1][0],  # FORM mot 1
                      buffer[0][0],  # FORM mot 2
                      stack[len(stack) - 1][1],  # POS mot 1
                      stack[len(stack) - 1][4],  # LEMMA mot 1
                      stack[len(stack) - 1][5],  # MORPHO mot 1
                      stack_before_pos,  # POS mot -1
                      buffer[0][1],  # POS mot 2
                      buffer[0][4],  # LEMMA mot 2
                      buffer[0][5],  # MORPHO mot 2
                      buffer_before_pos,
                      buffer_after_pos,
                      dist])
        if (feature == "f3"):

            if (len(stack_done) >= 1):
                stack_before_pos = stack_done[len(stack_done) - 1]
            else:
                stack_before_pos = "N..U..L..L"

            stack_done.append(stack[len(stack) - 1][1])

            if (len(buffer_done) >= 1):
                buffer_before_pos = buffer_done[len(buffer_done) - 1]
            else:
                buffer_before_pos = "N..U..L..L"

            if (len(buffer_done) >= 2):
                buffer_before__before_pos = buffer_done[len(buffer_done) - 2]
            else:
                buffer_before__before_pos = "N..U..L..L"

            buffer_done.append(buffer[0][1])

            if (len(buffer) >= 2):
                buffer_after_pos = buffer[1][1]
            else:
                buffer_after_pos = "N..U..L..L"

            X.append([stack[len(stack) - 1][0],  # FORM mot 1
                      buffer[0][0],  # FORM mot 2
                      stack[len(stack) - 1][1],  # POS mot 1
                      stack[len(stack) - 1][4],  # LEMMA mot 1
                      stack[len(stack) - 1][5],  # MORPHO mot 1
                      stack_before_pos,  # POS mot -1
                      buffer[0][1],  # POS mot 2
                      buffer[0][4],  # LEMMA mot 2
                      buffer[0][5],  # MORPHO mot 2
                      buffer_before__before_pos,  # b-2
                      buffer_before_pos, # b-1
                      buffer_after_pos, #b1
                      dist])


        if oracle_ == None:
            Y_actu, gold = oracle(stack[len(stack) - 1], buffer[0], phrase,couples,tab_head)
        else :
            Y_actu, gold = oracle_test(stack[len(stack) - 1], buffer[0],dist,oracle_,proj,stack_before_pos,buffer_before_before_pos,buffer_before_pos,buffer_after_pos)
        if phrase_all != None :
            if "RIGHT" in Y_actu or "LEFT" in Y_actu :
                gov_lab = Y_actu.split("_")
                gov = gov_lab[0]
                lab = gov_lab[1]
                index1 = phrase.index(stack[len(stack) - 1])
                index2 = phrase.index(buffer[0])
                if index2 > 0:
                    if "RIGHT" in gov :
                        all_index = [ i[0] for i in phrase_all ]
                        index_encours = all_index.index(str(index2))
                        phrase_all[index_encours][6] = index1# GOV
                        phrase_all[index_encours][7] =  lab# Lab
                if index1 > 0:
                    if "LEFT" in gov :
                        all_index = [ i[0] for i in phrase_all ]
                        index_encours = all_index.index(str(index1))
                        phrase_all[index_encours][6] = index2  # GOV
                        phrase_all[index_encours][7] = lab  # Lab





        Y.append(Y_actu)

        if (gold == 1):
            tab_head.append(stack[len(stack) - 1])
            stack, buffer, arcs = left_arc(stack, buffer, arcs, tab_head, index_phrase, needs)
        elif (gold == 2):
            tab_head.append(buffer[0])
            stack, buffer, arcs = right_arc(stack, buffer, arcs, tab_head, index_phrase, needs)
        elif (gold == 3):
            stack, buffer = shift(stack, buffer)
        elif (gold == 4):
            stack = reduce(stack)

    # Rajout des reduce réstant dans le stack.
    for i in range(len(stack)):
       j = len(stack)-1 - i
       if (feature == "f1"):
         X.append([stack[j][0],  # FORM mot 1
                   mot_null,  # FORM mot 2
                   stack[j][1],  # POS mot 1
                   mot_null, 0])  # FORM mot 2
       if (feature == "f2"):
         if (i == 0):
           stack_before_pos = stack_done[len(stack_done) - 1]
         else:
           stack_before_pos = stack[j+1][1]


         X.append([stack[j][0],  # FORM mot 1
                   mot_null,  # FORM mot 2
                   stack[j][1],  # POS mot 1
                   stack[j][4],  # LEMMA mot 1
                   stack[j][5],  # MORPHO mot 1
                   stack_before_pos,  # POS mot -1
                   mot_null,  # POS mot 2
                   mot_null,  # LEMMA mot 2
                   mot_null,  # MORPHO mot 2
                   mot_null,
                   mot_null,
                   0])
       if (feature == "f3"):
           if (i == 0):
               stack_before_pos = stack_done[len(stack_done) - 1]
           else:
               stack_before_pos = stack[j + 1][1]

           X.append([stack[j][0],  # FORM mot 1
                     mot_null,  # FORM mot 2
                     stack[j][1],  # POS mot 1
                     stack[j][4],  # LEMMA mot 1
                     stack[j][5],  # MORPHO mot 1
                     stack_before_pos,  # POS mot -1
                     mot_null,  # POS mot 2
                     mot_null,  # LEMMA mot 2
                     mot_null,  # MORPHO mot 2
                     mot_null,
                     mot_null,
                     mot_null,
                     0])
       Y.append("REDUCE")


    # Retour
    if phrase_all != None:
        return arcs, X, Y , phrase_all
    else :
        return arcs, X, Y


def get_couple(phrase):
    """

    :param phrase: Liste de mots sous forme de tokens
    :return: Liste de couple ( x , y )
    """
    couple = []
    for i in range(len(phrase)):
        couple.append((int(phrase[i][2]), i + 1))

    return couple


def is_proj(couple):
    """

    :param couple: Liste de couple ( x , y )
    :return: True si la liste est project , False sinon
    """
    for (x, y) in couple:
        for (x1, y1) in couple:
            # print((x,y),' ',(x1,y1))
            if (x1, y1) != (x, y):
                # if x1 == x:
                # if not(y1 in range(0,y)) : return False
                # if y1 == y:
                # if not(x1 in range(x,taille)) : return False

                if (x < y):

                    if x1 in np.arange(x + 1, y):
                        if not (y1 in np.arange(x, y + 1)): return False

                    if x1 not in np.arange(x, y + 1):
                        if y1 in np.arange(x + 1, y): return False

                    if y1 in np.arange(x + 1, y):
                        if x1 not in np.arange(x, y + 1): return False

                    if y1 not in np.arange(x, y + 1):
                        if x1 in np.arange(x + 1, y): return False

                if (y < x):

                    if x1 in np.arange(y + 1, x):
                        if not (y1 in np.arange(y, x + 1)): return False

                    if x1 not in np.arange(y, x + 1):
                        if x1 in np.arange(y + 1, x): return False

                    if y1 in np.arange(y + 1, x):
                        if not (y1 in np.arange(y, x + 1)): return False

                    if y1 not in np.arange(y, x + 1):
                        if x1 in np.arange(y + 1, x): return False

    return True

def oracle_test(w1, w2 ,dist, oracle,proj,stack_before_pos,buffer_before_before_pos,buffer_before_pos,buffer_after_pos):
    """

    :param w1: mot stack 
              feature1 => w1[FORM , POS , GOV , LABEL]
              feature2 => w2[FORM , POS , GOV , LABEL, LEMMA , MORPHO]
    :param w2: mot buffer
    :param dist: distance entre w1 et w2 
    :param oracle:


    :return: y : prediction convertie par rapport au vocab ( ex : RIGHT_det )
    :return: gold : 1 = left / 2 = right / 3 = shift / 4 = reduce
    """

    if proj : proj_int = 1
    else : proj_int = 0
    keras_model = oracle.get_current_network()

    model = oracle.get_current_model()
    featureset = model['featureset']

    X_test_i = []
    gold = 0


    # Convertir les donnnées
    if featureset == 'f1':

        x_test = []
        x_test.append(w1[0])  # FORM 1
        x_test.append(w2[0])  # FORM 2
        x_test.append(w1[1])  # POS 1
        x_test.append(w2[1])  # POS 2
        x_test.append(dist)  # DIST

    elif featureset == 'f2':
        # FORM , POS , GOV , LABEL, LEMMA , MORPHO
        x_test = []
        x_test.append(w1[0])  # FORM 1
        x_test.append(w2[0])  # FORM 2
        x_test.append(w1[1])  # POS 1
        x_test.append(w1[4])  # LEMMA 1
        x_test.append(w1[5])  # MORPHO 1
        x_test.append(stack_before_pos) # S.-1 POS
        x_test.append(w2[1])  # POS 2
        x_test.append(w2[4])  # LEMMA 2
        x_test.append(w2[5])  # MORPHO 2
        x_test.append(buffer_before_pos)# B.-1 POS
        x_test.append(buffer_after_pos)  # B.1 POS
        x_test.append(dist)  # DIST

    elif featureset == 'f3':
        # FORM , POS , GOV , LABEL, LEMMA , MORPHO
        x_test = []
        x_test.append(w1[0])  # FORM 1
        x_test.append(w2[0])  # FORM 2
        x_test.append(w1[1])  # POS 1
        x_test.append(w1[4])  # LEMMA 1
        x_test.append(w1[5])  # MORPHO 1
        x_test.append(stack_before_pos)  # S.-1 POS
        x_test.append(w2[1])  # POS 2
        x_test.append(w2[4])  # LEMMA 2
        x_test.append(w2[5])  # MORPHO 2
        x_test.append(buffer_before_before_pos)  # B.-2 POS
        x_test.append(buffer_before_pos)  # B.-1 POS
        x_test.append(buffer_after_pos)  # B.1 POS
        x_test.append(dist)  # DIST


		
    X_test_i = oracle.process_test_data(x_test)
    # Prédiction
    y_pred = keras_model.predict(X_test_i)


    # Convertir la précition

    y = oracle.get_label(y_pred)

    if "LEFT" in y:
        gold = 1
    if "RIGHT" in y:
        gold = 2
    if "SHIFT" in y:
        gold = 3
    if "REDUCE" in y:
        gold = 4

    return y , gold