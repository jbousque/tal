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


def oracle(w1, w2, phrase, couples):


    if (int(w2[2]) == phrase.index(w1)):
        if erreur: print("ajout de :", w2)
        return "RIGHT_" + str(w2[3]), 2

    elif (int(w1[2]) == phrase.index(w2)):

        if erreur: print(w1, " ", w2)
        return "LEFT_" + str(w1[3]), 1

    elif (w1 in tab_head):
        if (verif_reduce(phrase.index(w1), phrase.index(w2), couples, len(phrase) - 1, erreur)):
            return "REDUCE", 4
        else:
            return "SHIFT", 3
    else:

        return "SHIFT", 3


def verif_reduce(index, index2, couples, taille, erreur):  ## A OPTIMISER
    possible = []
    for i in range(index2, taille + 1):
        possible.append((index, i))
    for (a, b) in possible:
        if (erreur): print((a, b))
        # print((a,b))
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


def parser(phrase, feature, erreur,phrase_all=None,oracle_=None):
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
                #phrase_all[i][6] = index_root  # GOV
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
    Y.append("SHIFT")

    # Début du parcours du buffer / stack
    stack_done = []
    buffer_done = []

    while ((len(buffer) != 0) and (len(stack) != 0)):

        if (stack[len(stack) - 1] == root):
            dist = abs(0 - phrase.index(buffer[0]))
        elif (buffer[0] == root):
            dist = abs(phrase.index(stack[len(stack) - 1]))
        else:
            dist = abs(phrase.index(stack[len(stack) - 1]) - phrase.index(buffer[0]))

        if dist > 7: dist = 7

        if oracle_ == None:
            Y_actu, gold = oracle(stack[len(stack) - 1], buffer[0], phrase,couples)
        else :
            Y_actu, gold = oracle_test(stack[len(stack) - 1], buffer[0],dist,oracle_ )
        if phrase_all != None :
            if "RIGHT" in Y_actu or "LEFT" in Y_actu :
                gov_lab = Y_actu.split("_")
                gov = gov_lab[0]
                lab = gov_lab[1]
                index1 = phrase.index(stack[len(stack) - 1])
                index2 = phrase.index(buffer[0])

                if "RIGHT" in gov :
                    all_index = [ i[0] for i in phrase_all ]
                    index_encours = all_index.index(str(index2))
                    phrase_all[index_encours][6] = index1# GOV
                    phrase_all[index_encours][7] =  lab# Lab
                if "LEFT" in gov :
                    all_index = [ i[0] for i in phrase_all ]
                    index_encours = all_index.index(str(index1))
                    phrase_all[index_encours][6] = index2  # GOV
                    phrase_all[index_encours][7] = lab  # Lab



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

            if (len(stack) >= 2):
                stack_after_pos = stack[len(stack) - 2][1]
            else:
                stack_after_pos = "N..U..L..L"

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
                      stack_after_pos,  # POS mot   1
                      buffer[0][1],  # POS mot 2
                      buffer[0][4],  # LEMMA mot 2
                      buffer[0][5],  # MORPHO mot 2
                      buffer_before_pos,
                      buffer_after_pos,
                      dist])


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
       Y.append("REDUCE")


    # Retour
    if phrase_all != None:
        return arcs, X, Y , phrase_all
    else :
        return arcs, X, Y


def get_couple(phrase):
    couple = []
    for i in range(len(phrase)):
        couple.append((int(phrase[i][2]), i + 1))

    return couple


def is_proj(couple, taille):
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

def oracle_test(w1, w2 ,dist, model):
    """

    :param w1: mot stack 
              feature1 => w1[FORM , POS , GOV , LABEL]
              feature2 => w2[FORM , POS , GOV , LABEL, LEMMA , MORPHO]
    :param w2: mot buffer
    :param dist: distance entre w1 et w2 
    :param model:

    :return: y : prediction convertie par rapport au vocab ( ex : RIGHT_det )
    :return: gold : 1 = left / 2 = right / 3 = shift / 4 = reduce
    """


    # Convertir les donnnées


    # Prédiction


    # Convertir la précition


    return y , gold