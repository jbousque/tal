class WordBuffer:
        def __init__(self, mcd=None):
                self.currentIndex = 0
                self.array = []
                self.length = 0
                self.mcd = mcd
                
        def addWord(self, w):
                self.array.append(w)
                self.length += 1

        def affiche(self, mcd):
                for w in self.array:
                        w.affiche(mcd)

        def getLength(self):
                return self.length
        
        def getCurrentIndex(self):
                return self.currentIndex

        def getWord(self, index):
                return self.array[index]

        def getCurrentWord(self):
                return self.getWord(self.currentIndex)
            
        def nextSentence(self):
                sentence = []
                sentence.append(Word.fakeWord(self.mcd))
                if self.currentIndex == self.length - 1 :
                        return False
                while self.currentIndex < self.length :
                        sentence.append(self.getCurrentWord())
#                        self.getCurrentWord().affiche(self.mcd)
                        if int(self.getCurrentWord().getFeat('EOS')) == 1 :
                                self.currentIndex += 1
                                return sentence
                        else:
                                self.currentIndex += 1
                
        def readFromMcfFile(self, mcfFilename):
                try:
                        mcfFile = open(mcfFilename, encoding='utf-8')
                except IOError:
                        print(mcfFilename, " : ce fichier n'existe pas")
                        exit(1)
                tokens = []
                for ligne in mcfFile:
                        tokens = ligne.split()
                        w = Word()
                        for i in range(0, len(tokens)):
                                if(self.mcd[i][0] == 'GOV'):
                                        w.setFeat(self.mcd[i][0], tokens[i])
                                        w.setFeat('GOVABS', str(self.length + int(tokens[i]))) # compute absolute index of governor
                                else:
                                        w.setFeat(self.mcd[i][0], tokens[i])
                        self.addWord(w)
                mcfFile.close();
                
        def readFromConlluFile(self, conlluFilename):
                conlluFile = open(conlluFilename, encoding='utf-8')
                
                tokens = []
                for ligne in conlluFile:
#                        print(ligne)
                        if ligne[0] == '\n' :
                                self.getWord(self.currentIndex - 1).setFeat('EOS', '1')
                                next
                        elif ligne[0] == '#' :
#                                print("commentaire")
                                next
                        else :
#                                1	Je	il	PRON	_	Number=Sing|Person=1|PronType=Prs	2	nsubj	_	_
#featModel = (('B', 0, 'POS'),('S', 0, 'POS'), ('B', 0, 'GOV'), ('S', 0, 'GOV'), ('B', -1, 'POS'), ('B', 1, 'POS'))
                                tokens = ligne.split("\t")
                                if '-' not in tokens[0]:
                                        w = Word()
                                        for i in range(0, len(tokens)):
                                                if i == 0 :
                                                        w.setFeat('INDEX', tokens[i])
                                                if i == 1 :
                                                        w.setFeat('FORM', tokens[i])
                                                if i == 2 :
                                                        w.setFeat('LEMMA', tokens[i])
                                                if i == 3 :
                                                        w.setFeat('POS', tokens[i])
                                                if i == 4 :
                                                        w.setFeat('X1', tokens[i])
                                                if i == 5 :
                                                        w.setFeat('MORPHO', tokens[i])
                                                if i == 6 :
                                                        w.setFeat('GOV', tokens[i])
                                                if i == 7 :
                                                        w.setFeat('LABEL', tokens[i])
                                                if i == 8 :
                                                        w.setFeat('X2', tokens[i])
                                                if i == 9 :
                                                        w.setFeat('X3', tokens[i])
                                        w.setFeat('EOS', '0')
                                        self.addWord(w)
                conlluFile.close();
                
        def end(self):
                if(self.getCurrentIndex() >= self.getLength()):
                        return True
                else:
                        return False