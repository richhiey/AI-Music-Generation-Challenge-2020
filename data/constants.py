## http://abcnotation.com/wiki/abc:standard:v2.1
## ---------------------------------------------
## Reference Fields that contain  information about a transcribed tune

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

MAX_TIMESTEPS_FOR_ABC_MODEL = 511