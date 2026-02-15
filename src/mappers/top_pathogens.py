import re

CRITICAL_PATHOGENS = r"\b(?:Enterobacter|Escherichia|Klebsiella|Citrobacter|Serratia|Proteus|Morganella|Providencia|Acinetobacter baumannii|Mycobacterium)\b" 
HIGH_PATHOGENS = r"\b(?:Salmonella Typhi|Shigella spp|Enterococcus faecium|Pseudomonas aeruginosa|Non-typhoidal Salmonella|Neisseria gonorrhoeae|Staphylococcus aureus)\b" 
MEDIUM_PATHOGENS = r"\b(?:Group A Streptococci|Streptococcus pneumoniae|Haemophilus influenzae|Group B Streptococci)\b" 

ALL_PATHOGENS = f"{CRITICAL_PATHOGENS}|{HIGH_PATHOGENS}|{MEDIUM_PATHOGENS}"


CRITICAL_PATHOGENGENUS = [
    'Acinetobacter',
    'Citrobacter',
    'Enterobacter',
    'Escherichia',
    'Klebsiella',
    'Morganella',
    'Mycobacterium',
    'Proteus',
    'Providencia',
    'Serratia'
]

HIGH_PATHOGENGENUS = [
    'Enterococcus',
    'Neisseria',
    'Pseudomonas',
    'Salmonella',
    'Shigella',
    'Staphylococcus'
]

MEDIUM_PATHOGENGENUS = [
    'Haemophilus', 
    'Streptococcus'
]