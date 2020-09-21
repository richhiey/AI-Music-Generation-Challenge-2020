## http://abcnotation.com/wiki/abc:standard:v2.1
## ---------------------------------------------
## Reference Fields that contain  information about a transcribed tune
## ---------------------------------------------

METADATA_KEYS = {
    'A':'area',
    'B':'book',
    'C':'composer',
    'D':'discography',
    'F':'file url',
    'G':'group',
    'H':'history',
    'I':'instruction',
    'L':'unit note length',
    'm':'macro',
    'N':'notes',
    'O':'origin',
    'P':'parts',
    'Q':'tempo',
    'r':'remark',
    'S':'source',
    's':'symbol line',
    'T':'tune title',
    'U':'user defined',
    'V':'voice',
    'W':'words',
    'w':'words',
    'X':'reference number',
    'Z':'transcription',
}


## Reference fields to be used as conditioning for the symbolic models

CONDITIONAL_KEYS = {
    'K':'key',
    'M':'meter',
    'R':'rhythm',
}

MAX_TIMESTEPS_FOR_ABC_MODEL = 256
MIN_TIMESTEPS_FOR_ABC_MODEL = 64

## Base vocabulary for models
## Inspired from the vocabulary of Folk-RNN models by Sturm et al. 
## 
## This vocaulary is later injected with unique tokens of meter and
## key which will be derived from the datasets in use.
## Thus, Model vocab = Base vocab (ABC tokens)  
##                      + Extra tokens from dataset (Conditional tokens) -> rhythm, key, meter, etc

BASE_ABC_VOCABULARY = ["C,","^C,","D,","^D,","E,","F,","^F,","G,","^G,","A,","^A,","B,","C","^C","D","^D","E","F","^F","G","^G","A","^A","B","c","^c","d","^d","e","f","^f","g","^g","a","^a","b","c'","^c'","d'","^d'","e'","f'","g'","^g'","a'","^a'","B'","_C,","_D,","_E,","_G,","_A,","_B,","_C","_D","_E","_G","_A","_B","_c","_d","_e","_g","_a","_b","_c'","_d'","_e'","_g'","_a'","_b'","=C,","=E,","=F,","=G,","=A,","=B,","=C","=D","=E","=F","=G","=A","=B","=c","=d","=e","=f","=g","=a","=b","=c'","=d'","=e'","=f'","=g'","=a'","=b'","z","|","|:",":|","|]","||","[|","::","|1","|2","(2","(3","(4","(5","(6","(7","(9","/","//","/2","/3","/4","/8","2","3","4","5","6","7","8","9","12","16","3/4","3/2","5/2","7/2","2>","2<","/2>","/2<","4>","<",">","<s>","</s>"]
CONSTRAINT_VOCABULARY = ["NC","|","|:",":|","|]","||","[|","::","|1","|2","(2","(3","(4","(5","(6","(7","(9","/","//","/2","/3","/4","/8","2","3","4","5","6","7","8","9","12","16","3/4","3/2","5/2","7/2","2>","2<","/2>","/2<","4>","<",">","<s>","</s>"]

BASE_KEY_NAMES = {
    'maj': ['C#','F#','B','E','A','D','G','C','F','Bb','Eb','Ab','Db','Gb','Cb'],
    'min': ['A#m','D#m','G#m','C#m','F#m','Bm','Em','Am','Dm','Gm','Cm','Fm','Bbm','Ebm','Abm'],
    'mix': ['G#mix', 'C#Mix','F#Mix','BMix ','EMix ','AMix ','DMix ','BbMix','FMix ','CMix ','GMix ','GbMix','DbMix','AbMix','EbMix'],
    'dor': ['DbDor','AbDor','EbDor','BbDor','FDor ','CDor ','GDor ','DDor ','ADor ','EDor ','BDor ','F#Dor','C#Dor','G#Dor','D#Dor'],
    'loc': ['CLoc','FLoc','BbLoc''GLoc','DLoc','ELoc','ALoc','BLoc','F#Loc','C#Loc','G#Loc','D#Loc','B#Loc','E#Loc','A#Loc'],
    'phr': ['EbPhr','BbPhr','FPhr ','CPhr ','GPhr ','DPhr ','APhr ','EPhr ','BPhr ','F#Phr','C#Phr','G#Phr','D#Phr','A#Phr','E#Phr'],
    'lyd': ['F#Lyd','BLyd ','ELyd ','ALyd ','DLyd ','GLyd ','CLyd ','FLyd ','BbLyd','EbLyd','AbLyd','DbLyd','GbLyd','CbLyd','FbLyd']
}