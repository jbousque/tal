from random import randint
import numpy as np
from generate_data import *





def left_arc(stack,buffer,arcs,tab_head,index_phrase,needs):
  arcs.append((buffer[0],stack[len(stack)-1]))
  #if ( str(index_phrase.index(stack[len(stack)-1][0]))  not in needs): # Si le dernier mot du stack a deja sont head / stack[len(stack)-1]  in tab_head and
  #  stack.pop(len(stack)-1)
  #else:
  #word = buffer.pop(0) # Retire le premier mot du buffer 
  #stack.append(word)
  stack.pop(len(stack)-1)
  return stack,buffer,arcs
  
def right_arc(stack,buffer,arcs,tab_head,index_phrase,needs):
  arcs.append((stack[len(stack)-1],buffer[0]))
  
  #if ( buffer[0] in tab_head and str(index_phrase.index(buffer[0][0]))  not in needs): # Si le premier mot du buffer à deja sont head  / buffer[0] in tab_head and
    #word = buffer.pop(0)
    #stack.append(word)
  #else :
  word = buffer.pop(0) # Retire le premier mot du buffer 
  stack.append(word)   
  
  return stack,buffer,arcs
  
def shift(stack,buffer):
  word = buffer.pop(0)
  stack.append(word)
  return stack , buffer
  
def reduce(stack):
  stack.pop(len(stack)-1)
  return stack

def oracle(w1,w2,index_phrase,phrase,tab_head,needs,couples,erreur):

  '''#print(w2[2])
  if ( int(w1[2]) == phrase.index(w2) ):
    if ( str(index_phrase.index(w2[0])) in needs ): 
      needs.pop(needs.index(str(index_phrase.index(w2[0]))))
    tab_head.append(w1)
    return "RIGHT_"+str(w1[3]),2,tab_head,needs
  
  elif ( int(w2[2]) == phrase.index(w1)):
    if ( str(index_phrase.index(w1[0])) in needs ):
      needs.pop(needs.index(str(index_phrase.index(w1[0]))))
    tab_head.append(w2)
    return "LEFT_"+str(w2[3]),1,tab_head,needs
  
  elif (w1 in tab_head and str(index_phrase.index(w1[0])) not in needs) : 
    return "REDUCE",4,tab_head,needs
  '''
  if erreur :
    print("w1 : ",int(w1[2]), "w2:",phrase.index(w2))
    print("w2 : ",int(w2[2]), "w1:",phrase.index(w1))
  if ( int(w2[2]) == phrase.index(w1) ):
    tab_head.append(w2)
    if erreur : print("ajout de :", w2)
    return "RIGHT_"+str(w2[3]),2,tab_head,needs
  
  elif ( int(w1[2]) == phrase.index(w2) ):
    tab_head.append(w1)
    if erreur : print(w1 , " ", w2)
    return "LEFT_"+str(w1[3]),1,tab_head,needs
  
  elif (w1 in tab_head ) :
    if ( verif_reduce(phrase.index(w1),phrase.index(w2),couples,len(phrase)-1,erreur)):
      return "REDUCE",4,tab_head,needs
    else:
      return "SHIFT",3,tab_head,needs
  else : 
    
    return "SHIFT",3,tab_head,needs
  

  
def verif_reduce(index,index2,couples,taille,erreur): ## A OPTIMISER
  possible = []
  for i in range(index2,taille+1):
    possible.append((index,i))
  for (a,b) in possible:
    if ( erreur ): print((a,b))
    #print((a,b))
    if (a,b) in couples : return False
    
  return True

def listing(sentence):
  res = []
  for i in range(len(sentence)):
    res.append(i + 1)
  return res

def print_result(res,sentence):
  sentence = ["root"] + sentence
  for i in range(len(res)):
    w1 , w2 = res[i]
    print(sentence[w1],' -> ', sentence[w2])
    
def get_need(phrase):
  need = []
  [need.append(i[2]) for i in phrase]
  return need
    
    
def parser(phrase,feature,erreur):
  
  couples = get_couple(phrase)
  tab_head = []
  buffer = phrase
  needs = get_need(phrase)
  if (feature == "f1"):
    root = ["Root","Root",0,"Root"]
  elif (feature == "f2" or feature == "f3" ):
    root = ["Root","Root",0,"Root","Root","Root"]
  phrase = [root] + phrase
  index_phrase = [i[0] for i in phrase]
  arcs = []
  stack = []
  stack.append(root) ## root
  
  X = []
  Y = []
  
  stack_done =[]
  buffer_done =[]
  
  while ( (len(buffer) != 0) and (len(stack) != 0)):
    #print()
    #print("buffer : ", [i[0] for i in buffer])
    #print("stack : ", [i[0] for i in stack])
    #print(needs)

    Y_actu, gold,tab_head,needs = oracle(stack[len(stack)-1],buffer[0],index_phrase,phrase,tab_head,needs,couples,erreur)
  
  
    if( stack[len(stack)-1] == root ):
      dist = abs(0 -phrase.index(buffer[0]))
    elif (buffer[0] == root ):
      dist = abs(phrase.index(stack[len(stack)-1]))
    else :
      dist = abs(phrase.index(stack[len(stack)-1])-phrase.index(buffer[0]))
    
    if dist > 7 : dist = 7

      
      
    '''
    print(stack[len(stack)-1])
    print()
    print(buffer[0])
    print()
    print(dist)
    '''
    if ( feature == "f1"):
      X.append([stack[len(stack)-1][0], # FORM mot 1
                buffer[0][0],           # POS mot 1 
                stack[len(stack)-1][1], # FORM mot 2
                buffer[0][1],dist])     # FORM mot 2
    if ( feature == "f2"):
      
      if ( len(stack_done) >= 1 ):
        stack_before_pos = stack_done[len(stack_done)-1]
      else :
        stack_before_pos = "N..U..L..L"
        
      stack_done.append(stack[len(stack)-1][1])
        
      if ( len(buffer_done) >= 1 ):
        buffer_before_pos = buffer_done[len(buffer_done)-1]
      else :
        buffer_before_pos = "N..U..L..L"
        
      buffer_done.append(buffer[0][1])
      
      if ( len(buffer) >= 2 ):
        buffer_after_pos = buffer[1][1]
      else :
        buffer_after_pos = "N..U..L..L"
        
      X.append([stack[len(stack)-1][0], # FORM mot 1
                stack[len(stack)-1][1], # POS mot 1
                stack[len(stack)-1][4], # LEMMA mot 1
                stack[len(stack)-1][5], # MORPHO mot 1
                stack_before_pos,       # POS mot -1
                buffer[0][0], # FORM mot 2
                buffer[0][1], # POS mot 2
                buffer[0][4], # LEMMA mot 2
                buffer[0][5], # MORPHO mot 2
                buffer_before_pos,
                buffer_after_pos,
                dist])
    if ( feature == "f3"):
      
      if ( len(stack) >= 2 ):
        stack_after_pos = stack[len(stack)-2][1]
      else :
        stack_after_pos = "N..U..L..L"
        
      if ( len(buffer_done) >= 1 ):
        buffer_before_pos = buffer_done[len(buffer_done)-1]
      else :
        buffer_before_pos = "N..U..L..L"

      buffer_done.append(buffer[0][1])

      if ( len(buffer) >= 2 ):
        buffer_after_pos = buffer[1][1]
      else :
        buffer_after_pos = "N..U..L..L"


      X.append([stack[len(stack)-1][0], # FORM mot 1
          stack[len(stack)-1][1], # POS mot 1
          stack[len(stack)-1][4], # LEMMA mot 1
          stack[len(stack)-1][5], # MORPHO mot 1
          stack_after_pos,       # POS mot 1
          buffer[0][0], # FORM mot 2
          buffer[0][1], # POS mot 2
          buffer[0][4], # LEMMA mot 2
          buffer[0][5], # MORPHO mot 2
          buffer_before_pos,
          buffer_after_pos,
          dist])
      
      
    if (erreur):
      print(stack[len(stack)-1])
      print(buffer[0])
      print()
      print(stack)
      print(buffer)
      print()
      print(Y_actu)
      print()
      print()

    Y.append(Y_actu)
    
    if ( gold == 1 ):
      stack,buffer,arcs = left_arc(stack,buffer,arcs,tab_head,index_phrase,needs)
    elif (gold == 2):
      stack,buffer,arcs = right_arc(stack,buffer,arcs,tab_head,index_phrase,needs)
    elif (gold == 3):
      stack,buffer = shift(stack,buffer)
    elif (gold == 4):
      stack = reduce(stack)
      

  return arcs,X,Y

def get_couple(phrase):
  couple = []
  for i in range(len(phrase)):
    couple.append((int(phrase[i][2]),i+1))
  
  return couple

def is_proj(couple,taille):
  for (x,y) in couple:
    for (x1,y1) in couple:
      #print((x,y),' ',(x1,y1))
      if (x1,y1) != (x,y):
        #if x1 == x:
          #if not(y1 in range(0,y)) : return False
        #if y1 == y:
          #if not(x1 in range(x,taille)) : return False
        
        if ( x < y):

          if x1 in np.arange(x+1,y):
            if not(y1 in np.arange(x,y+1)): return False
            
          if x1 not in np.arange(x,y+1):
            if y1 in np.arange(x+1,y) : return False
            
          if y1 in np.arange(x+1,y):
            if x1 not in np.arange(x,y+1) : return False
            
          if y1 not in np.arange(x,y+1):
            if x1 in np.arange(x+1,y) : return False
            
        if ( y < x ):

          if x1 in np.arange(y+1,x):
            if not(y1 in np.arange(y,x+1)): return False
            
          if x1 not in np.arange(y,x+1):
            if x1 in np.arange(y+1,x) : return False
          
          if y1 in np.arange(y+1,x):
            if not(y1 in np.arange(y,x+1)): return False
            
          if y1 not in np.arange(y,x+1):
            if x1 in np.arange(y+1,x) : return False

  return True