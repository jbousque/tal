import subprocess
import os


for lang in ['fr', 'nl', 'en', 'ja']: # ja only has a test set ?!
    
    for f in ['f1', 'f2', 'f3']:
        
        for phase in ['train', 'dev', 'test']:
        
            if lang == 'fr':
                
                conllu_name = "..\\data\\ud\\UD_French-GSD\\fr_gsd-ud-{phase}.conllu".format(phase=phase)
            
            elif lang == 'nl':
            
                conllu_name = "..\\data\\ud\\UD_Dutch-LassySmall\\nl_lassysmall-ud-{phase}.conllu".format(phase=phase)
                
            elif lang == 'en':
                
                conllu_name = "..\\data\\ud\\UD_English-LinES\\en_lines-ud-{phase}.conllu".format(phase=phase)
                
            elif lang == 'ja':
                
                conllu_name = "..\\data\\ud\\UD_Japanese-Modern\\ja_modern-ud-{phase}.conllu".format(phase=phase)
            
            output_name = "{f}_{lang}-{phase}".format(f=f, lang=lang, phase=phase)
        
            if os.path.isfile(conllu_name):

                print("Generating data for lang {lang} feature {f} phase {phase}".format(lang=lang, f=f, phase=phase))
            
                out = subprocess.check_output(['python', 
                                     'main.py', #'C:\\IAAA\\tal-github\src\main.py', 
                                     conllu_name,
                                     output_name,
                                     f],
                                    stderr=subprocess.STDOUT,
                                    shell=True)
                print(out)

                
                
            else:
                print("File does not exist, skipping: ", conllu_name)
