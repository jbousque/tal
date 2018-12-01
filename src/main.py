import Word
import WordBuffer
from generate_data import *
import sys
import pickle
def main(argv):

    if (len(argv) != 3):
        print("Usage python main.py fichier.conllu nomFichierSortie features[\"f1\",\"f2\",\"f3\"]")
        sys.exit(2)
    else :
        conllu = argv[0]
        sortie = "../resultat/"+argv[1]
        features = argv[2]
        if (str(features)  in ["f1_w", "f2_w", "f3_w"]):
            feat = features.split('_')
            create_conllu(conllu, feat[0], sortie)
        elif ( str(features) not in ["f1","f2","f3"]):
            print("Usage python main.py fichier.conllu nomFichierSortie features[\"f1\",\"f2\",\"f3\"]")
            sys.exit(2)
            
    
        if ( features == "f1"):
          X , Y , vocab_mots , vocab_pdd , vocab_liaisons = get_all_data(conllu,feature=features)
          dico = { "X" : X , "Y":Y , "vocab_mots":vocab_mots , "vocab_pdd":vocab_pdd , "vocab_liaisons":vocab_liaisons}
          f = open(sortie,"w+b")
          pickle.dump(dico,f)
          f.close()
        elif (features =="f2" or features == "f3"):
          X , Y , vocab_mots , vocab_pdd , vocab_lemma, vocab_morpho, vocab_liaisons = get_all_data(conllu,feature=features)
          dico = { "X" : X , "Y":Y , "vocab_mots":vocab_mots , "vocab_pdd":vocab_pdd , "vocab_liaisons":vocab_liaisons,
                   "vocab_lemma":vocab_lemma,"vocab_morpho":vocab_morpho}
          f = open(sortie,"w+b")
          pickle.dump(dico,f)
          f.close()

  
if __name__== "__main__":
  main(sys.argv[1:])