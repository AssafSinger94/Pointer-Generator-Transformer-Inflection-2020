import os
import shutil


LANGS = {
    'Afro-Asiatic': 'mlt orm syc'.split(),
    'Algic': 'cre'.split(),
    'Australian': 'mwf'.split(),
    'Dravidian': 'kan tel'.split(),
    'Germanic': 'gml gsw nno'.split(),
    'Indo-Aryan': 'ben  hin san urd'.split(),
    'Iranian': 'fas pus tgk'.split(),
    'Niger-Congo': 'sna'.split(),
    'Nilo-Sahan': 'dje'.split(),
    'Romance': 'ast cat frm fur glg lld vec xno'.split(),
    'Sino-Tibetan': 'bod'.split(),
    'Siouan': 'dak'.split(),
    'Tungusic': 'evn'.split(),
    'Turkic': 'aze bak crh kaz kir kjh tuk uig uzb'.split(),
    'Uralic': 'kpv lud olo udm vro'.split(),
    'Uto-Aztecan': 'ood'.split(),
    'austronesian': 'mlg ceb hil tgl mao'.split(),
    'germanic': 'dan isl nob swe nld eng deu gmh frr ang'.split(),
    'niger-congo': 'nya kon lin lug sot swa zul aka gaa'.split(),
    'oto-manguean': 'cpa azg xty zpv ctp czn cly otm ote pei'.split(),
    'uralic': 'est fin izh krl liv vep vot mhr myv mdf sme'.split()
}

OUT_DIR = 'task0-data/out'
DEV_DIR = 'task0-data/DEVELOPMENT-LANGUAGES'
SURP_DIR = 'task0-data/SURPRISE-LANGUAGES'

def move_family_dataset(family, in_dir):
    for lang in sorted(LANGS[family]):
        for mode in ['trn', 'dev', 'tst']:
            shutil.copy(f'{in_dir}/{family}/{lang}.{mode}', f'{OUT_DIR}/{lang}.{mode}')

def move_entire_dataset():
    for family in LANGS.keys():
        if os.path.exists(f"{DEV_DIR}/{family}"):
            print('moving DEV family', family)
            move_family_dataset(family, DEV_DIR)
        else:
            print('moving SURP family', family)
            move_family_dataset(family, SURP_DIR)

if __name__ == "__main__":
    try:
        os.mkdir(OUT_DIR)
    except:
        pass
    move_entire_dataset()
