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

MAX_TIMESTEPS_FOR_ABC_MODEL = 512

## Base vocabulary for models
## Inspired from the vocabulary of Folk-RNN models by Sturm et al. 
## 
## This vocaulary is later injected with unique tokens of meter and
## key which will be derived from the datasets in use.
## Thus, Model vocab = Base vocab (ABC tokens)  
##                      + Extra tokens from dataset (Conditional tokens) -> rhythm, key, meter, etc

BASE_ABC_VOCABULARY = ["C,","^C,","D,","^D,","E,","F,","^F,","G,","^G,","A,","^A,","B,","C","^C","D","^D","E","F","^F","G","^G","A","^A","B","c","^c","d","^d","e","f","^f","g","^g","a","^a","b","c'","^c'","d'","^d'","e'","f'","g'","^g'","a'","^a'","B'","_C,","_D,","_E,","_G,","_A,","_B,","_C","_D","_E","_G","_A","_B","_c","_d","_e","_g","_a","_b","_c'","_d'","_e'","_g'","_a'","_b'","=C,","=E,","=F,","=G,","=A,","=B,","=C","=D","=E","=F","=G","=A","=B","=c","=d","=e","=f","=g","=a","=b","=c'","=d'","=e'","=f'","=g'","=a'","=b'","z","[","]","{","}","(",")","~",".","|","|:",":|","|]","||","[|","::","|1","|2","(2","(3","(4","(5","(6","(7","(9","/2","/3","/4","/8","2","3","4","5","6","7","8","9","12","16","3/4","3/2","5/2","7/2","2>","2<","/2>","/2<","4>","<",">","<s>","</s>"]