antibiotic_list = [
    "AMC - Amoxicillin/clavulanic acid", 
    "AMK - Amikacin", "AMP - Ampicillin",
    "AMS - Ampicillin/Sulbactam", 
    "AMX - Amoxicillin", "AZT - Aztreonam",
    "CAZ - Ceftazidim", "CEP - Cefepim", "CEZ - Cefazolin", "CIP - Ciprofloxacin",
    "CLI - Clindamycin","CMP - Chloramphenicol","COL - Colistin","CRO - Ceftriaxon",
    "CTM - Cefotiam","CTX - Cefotaxim","CXM - Cefuroxim","DOX - Doxycyclin",
    "DPT - Daptomycin","ERT - Ertapenem","ERY - Erythromycin","FLU - Flucloxacillin",
    "FOS - Fosfomycin","FUS - Fusidic acid","GEN - Gentamicin",
    "GHL - Gentamicin 500 (high level)","IMP - Imipenem","LEV - Levofloxacin",
    "LIZ - Linezolid","MER - Meropenem","MOX - Moxifloxacin","MUP - Mupirocin",
    "NFT - Nitrofurantoin","OXA - Oxacillin","PEN - Penicillin","PIP - Piperacillin",
    "PIT - Piperacillin/Tazobactam","RAM - Rifampicin",
    "SHL - Streptomycin 1000 (high level)","STR - Streptomycin","SXT - Co-Trimoxazol",
    "TET - Tetracyclin","TGC - Tigecyclin","TOB - Tobramycin","TPL - Teicoplanin",
    "VAN - Vancomycin","NAL - Nalidixic acid","SLD - Sulfadiazin","TRP - Trimethoprim",
    "APH - Amphotericin B","AZL - Azlocillin","AZM - Azithromycin","BTC - Bacitracin",
    "CAR - Carbenicillin","CAS - Caspofungin","CEC - Cefaclor","CEX - Cefalexin",
    "CFA - Cefuroxim-Axetil","CFI - Cefixim","CFL - Ceftiofur","CFR - Cefadroxil",
    "CIB - Ceftibuten","CLN - Clinafloxacin","CLR - Clarithromycin","COX - Cefoxitin",
    "CPO - Cefpodoxim","CPP - Cefpodoxim-Proxetil","CTN - Cefalotin","CTR - Clotrimazol",
    "CTT - Cefotetan","FCA - Fluconazol","ITR - Itraconazol","KAN - Kanamycin",
    "LIN - Lincomycin","LOM - Lomefloxacin","MNO - Minocyclin","MOL - Moxalactam",
    "MTR - Metronidazol","MZL - Mezlocillin","NEO - Neomycin","NET - Netilmicin",
    "NIT - Nitroxolin","NOR - Norfloxacin","NOV - Novobiocin","OFX - Ofloxacin",
    "PIM - Pipemidic acid","PIS - Piperacillin/Sulbactam","POL - Polymyxin",
    "PRI - Pristinamycin","QPD - Quinupristin/Dalfopristin","RIB - Rifabutin",
    "ROX - Roxithromycin","SONST - Other antibiotic","TCC - Ticarcillin/clavulanic acid",
    "TEL - Telithromycin","TIC - Ticarcillin","5FC - 5-Fluorcytosin","APL - Apalcillin",
    "BAC - Bacampicillin","CAP - Cefapirin","CDT - Cefditoren","CED - Cefradin",
    "CEQ - Cefqinom","CFD - Cefdinir","CFS - Cefsulodin","CFT - Cefetamet",
    "CIN - Cinoxacin","CLO - Cefaloridin","CLX - Cloxiquin","CMD - Cefamandol",
    "CMT - Cefmetazol","CMX - Cefmenoxim","CNI - Cefonicid","COM - Cotrimazol",
    "CPC - Cefpodoxim/clavulanic acid","CPI - Cefpirom","CPR - Capreomycin",
    "CPY - Cefprozil","CPZ - Cefoperazon","CRI - Cefatrizin","CSE - Cycloserin",
    "CTE - Chlortetracyclin","CXC - Cefotaxim/clavulanic acid",
    "CZC - Ceftazidim/clavulanic acid","CZD - Cefazedon","CZX - Ceftizoxim",
    "DAN - Danofloxacin","DIC - Dicloxacillin","DIR - Dirithromycin","ECO - Econazol",
    "EMB - Ethambutol","ENO - Enoxacin","ESBL - Extended ß-Lactamase","ETH - Ethionamid",
    "FLE - Fleroxacillin","FRA - Framycetin","GAT - Gatifloxacin","GEM - Gemifloxacin",
    "GRE - Grepafloxacin","INH - Isoniazid","ISE - Isepamicin",
    "ISO - Isooxazolylpenicillin","JOS - Josamycin","KET - Ketoconazol",
    "KHL - Kanamycin (high level)","LAM - Lampren","LMO - Latamofex","LOR - Loracarbef",
    "MAR - Marbofloxacin","MCL - Mecillinam","MCZ - Miconazol",
    "MET - Methicillin","MID - Midekamycin","MZS - Sulbactam/Mezlocillin",
    "NYS - Nystatin","OLE - Oleandomycin","PAN - Panipenem","PAR - Paromycin",
    "PAS - p-Aminosalicylic acid","PEF - Perfloxacin","PIV - Pivampicillin",
    "PTH - Protionamid","PZA - Pyrazinamid","ROC - Rocephin","ROS - Rosoxacin",
    "SAZ - Sulphamethoxazol","SFX - Sulfamethizol","SIS - Sisomycin","SLN - Sulfalen",
    "SOX - Sulfisoxazol","SPF - Sparfloxacin","SPM - Spiramycin","SUL - Sulbactam",
    "TFX - Trovafloxacin","TMO - Temocillin","TOS - Tosufloxacin","VOR - Voriconazol",
    "CTL - Ceftarolin","DOR - Doripenem","TLV - Telavancin",
    "CZA - Ceftazidim/Avibactam","CZT - Ceftolozan/Tazobactam","ORI - Oritavancin",
    "BPR - Ceftobiprol","DAL - Dalbavancin","TZD - Tedizolid","MIC - Micafungin",
    "SPC - Spectinomycin","ANI - Anidulafungin","IVC - Isavuconazol",
    "POS - Posaconazol","FID - Fidaxomicin","OMA - Omadacyclin","CCO - Cefiderocol",
    "PLA - Plazomicin","DEL - Delafloxacin","LEF - Lefamulin","PIC - Pivmecillinam"
]



FIRST_LINE = [
    "AMX - Amoxicillin",
    "AMP - Ampicillin",
    "PEN - Penicillin",
    "OXA - Oxacillin",
    "FLU - Flucloxacillin",
    "CEZ - Cefazolin",
    "CXM - Cefuroxim",
    "CEC - Cefaclor",
    "CEX - Cefalexin",
    "CFA - Cefuroxim-Axetil",
    "CFR - Cefadroxil",
    "CPY - Cefprozil",
    "CTN - Cefalotin",
    "CLO - Cefaloridin",
    "FLE - Fleroxacillin",
    "BAC - Bacampicillin",
    "PIV - Pivampicillin",
    "MCL - Mecillinam",
    "PIC - Pivmecillinam",
    "SXT - Co-Trimoxazol",
    "TRP - Trimethoprim",
    "DOX - Doxycyclin",
    "TET - Tetracyclin",
    "NFT - Nitrofurantoin",
    "FOS - Fosfomycin",
    "MTR - Metronidazol",
    "AMC - Amoxicillin/clavulanic acid",
]

SECOND_LINE = [
    "AMS - Ampicillin/Sulbactam",
    "PIP - Piperacillin",
    "PIS - Piperacillin/Sulbactam",
    "PIT - Piperacillin/Tazobactam",
    "TIC - Ticarcillin",
    "TCC - Ticarcillin/clavulanic acid",
    "AZL - Azlocillin",
    "MZL - Mezlocillin",
    "CRO - Ceftriaxon",
    "CTX - Cefotaxim",
    "CAZ - Ceftazidim",
    "CEP - Cefepim",
    "CTM - Cefotiam",
    "COX - Cefoxitin",
    "CTT - Cefotetan",
    "CPO - Cefpodoxim",
    "CPP - Cefpodoxim-Proxetil",
    "CFI - Cefixim",
    "CFD - Cefdinir",
    "CZX - Ceftizoxim",
    "CMX - Cefmenoxim",
    "CMD - Cefamandol",
    "CMT - Cefmetazol",
    "CNI - Cefonicid",
    "CPI - Cefpirom",
    "CPZ - Cefoperazon",
    "CRI - Cefatrizin",
    "CFT - Cefetamet",
    "CEQ - Cefqinom",
    "CFS - Cefsulodin",
    "CAP - Cefapirin",
    "AZT - Aztreonam",
    "ROC - Rocephin",  # brand of ceftriaxone
    "CIP - Ciprofloxacin",
    "LEV - Levofloxacin",
    "MOX - Moxifloxacin",
    "NOR - Norfloxacin",
    "OFX - Ofloxacin",
    "ENO - Enoxacin",
    "LOM - Lomefloxacin",
    "PEF - Perfloxacin",
    "SPF - Sparfloxacin",
    "GAT - Gatifloxacin",
    "GEM - Gemifloxacin",
    "GRE - Grepafloxacin",
    "ROS - Rosoxacin",
    "DAN - Danofloxacin",
    "TOS - Tosufloxacin",
    "TFX - Trovafloxacin",
    "CIN - Cinoxacin",
    "NAL - Nalidixic acid",
    "NIT - Nitroxolin",
    "GEN - Gentamicin",
    "TOB - Tobramycin",
    "NEO - Neomycin",
    "NET - Netilmicin",
    "KAN - Kanamycin",
    "SIS - Sisomycin",
    "STR - Streptomycin",
    "GHL - Gentamicin 500 (high level)",
    "SHL - Streptomycin 1000 (high level)",
    "AZM - Azithromycin",
    "CLR - Clarithromycin",
    "ROX - Roxithromycin",
    "DIR - Dirithromycin",
    "JOS - Josamycin",
    "TEL - Telithromycin",
    "ERY - Erythromycin",
    "CLI - Clindamycin",
    "LIN - Lincomycin",
    "CMP - Chloramphenicol",
    "MUP - Mupirocin",
    "FUS - Fusidic acid",
    "MNO - Minocyclin",
    "TMO - Temocillin",
    "PAR - Paromycin",
    "SAZ - Sulphamethoxazol",
    "SFX - Sulfamethizol",
    "SLN - Sulfalen",
    "SOX - Sulfisoxazol",
    "COM - Cotrimazol",  # if this is clotrimazole -> antifungal; see EXCLUDED below
    "LMO - Latamofex",   # (latamoxef)
    "MOL - Moxalactam",
    "APL - Apalcillin",
    "CAR - Carbenicillin",
    "AZL - Azlocillin",
    "SPC - Spectinomycin",
    "DEL - Delafloxacin",
    "LEF - Lefamulin",
    "OMA - Omadacyclin",
]

LAST_RESORT = [
    "POL - Polymyxin", 
    "PRI - Pristinamycin",  # streptogramin, typically reserved
    "QPD - Quinupristin/Dalfopristin",
    "COL - Colistin",
    "VAN - Vancomycin",
    "TPL - Teicoplanin",
    "LIZ - Linezolid",
    "TZD - Tedizolid",
    "DPT - Daptomycin",
    "ORI - Oritavancin",
    "DAL - Dalbavancin",
    "TLV - Telavancin",
    "TGC - Tigecyclin",
    "MER - Meropenem",
    "IMP - Imipenem",
    "ERT - Ertapenem",
    "DOR - Doripenem",
    "PAN - Panipenem",
    "CZA - Ceftazidim/Avibactam",
    "CZT - Ceftolozan/Tazobactam",
    "CCO - Cefiderocol",
    "CTL - Ceftarolin",
    "BPR - Ceftobiprol",
    "PLA - Plazomicin",
    "RAM - Rifampicin"
]



regimens = {
    "UTI": {
        "first_line": [
            "NFT - Nitrofurantoin_Tested",
            "FOS - Fosfomycin_Tested",
            "PIC - Pivmecillinam_Tested",
            "SXT - Co-Trimoxazol_Tested",
            "NIT - Nitroxolin_Tested"
        ],
        "second_line": [
            "TRP - Trimethoprim_Tested",
            "AMC - Amoxicillin/clavulanic acid_Tested",
            "CFR - Cefadroxil_Tested",
            "CEX - Cefalexin_Tested",
            "CPO - Cefpodoxim_Tested",
            "CPP - Cefpodoxim-Proxetil_Tested",
            "CFI - Cefixim_Tested",
            "CRO - Ceftriaxon_Tested",
            "CEP - Cefepim_Tested",
            "ERT - Ertapenem_Tested"
        ],
        "last_resort": [
            "CIP - Ciprofloxacin_Tested",
            "LEV - Levofloxacin_Tested",
            "OFX - Ofloxacin_Tested",
            "GEN - Gentamicin_Tested",
            "AMK - Amikacin_Tested",
            "MER - Meropenem_Tested",
            "IMP - Imipenem_Tested",
            "PIT - Piperacillin/Tazobactam_Tested"
        ]
    },

    "Respiratory infections": {
        "first_line": [
            "AMX - Amoxicillin_Tested",
            "DOX - Doxycyclin_Tested",
            "AMC - Amoxicillin/clavulanic acid_Tested"
        ],
        "second_line": [
            "AZM - Azithromycin_Tested",
            "CLR - Clarithromycin_Tested",
            "ERY - Erythromycin_Tested",
            "LEV - Levofloxacin_Tested",
            "MOX - Moxifloxacin_Tested",
            "CRO - Ceftriaxon_Tested",
            "CTX - Cefotaxim_Tested"
        ],
        "last_resort": [
            "VAN - Vancomycin_Tested",
            "LIZ - Linezolid_Tested",
            "MER - Meropenem_Tested",
            "IMP - Imipenem_Tested",
            "PIT - Piperacillin/Tazobactam_Tested",
            "CEP - Cefepim_Tested",
            "COL - Colistin_Tested"
        ]
    },

    "Skin and soft tissue infections": {
        "first_line": [
            "FLU - Flucloxacillin_Tested",
            "CEX - Cefalexin_Tested",
            "CEZ - Cefazolin_Tested",
            "PEN - Penicillin_Tested"
        ],
        "second_line": [
            "CLI - Clindamycin_Tested",
            "DOX - Doxycyclin_Tested",
            "SXT - Co-Trimoxazol_Tested"
        ],
        "last_resort": [
            "VAN - Vancomycin_Tested",
            "LIZ - Linezolid_Tested",
            "DPT - Daptomycin_Tested",
            "TPL - Teicoplanin_Tested",
            "TGC - Tigecyclin_Tested",
            "COL - Colistin_Tested"
        ]
    },

    "ENT infections (pharyngitis, sinusitis, otitis)": {
        "first_line": [
            "PEN - Penicillin_Tested",
            "AMX - Amoxicillin_Tested",
            "AMC - Amoxicillin/clavulanic acid_Tested"
        ],
        "second_line": [
            "AZM - Azithromycin_Tested",
            "CLR - Clarithromycin_Tested",
            "CEX - Cefalexin_Tested",
            "CXM - Cefuroxim_Tested",
            "CFI - Cefixim_Tested",
            "DOX - Doxycyclin_Tested"
        ],
        "last_resort": [
            "LEV - Levofloxacin_Tested",
            "MOX - Moxifloxacin_Tested"
        ]
    },

    "Intra-abdominal infections": {
        "first_line": [
            "CRO - Ceftriaxon_Tested",
            "MTR - Metronidazol_Tested",
            "PIT - Piperacillin/Tazobactam_Tested"
        ],
        "second_line": [
            "CEP - Cefepim_Tested",
            "ERT - Ertapenem_Tested"
        ],
        "last_resort": [
            "MER - Meropenem_Tested",
            "IMP - Imipenem_Tested"
        ]
    },

    "CNS infections (meningitis)": {
        "first_line": [
            "CRO - Ceftriaxon_Tested",
            "CTX - Cefotaxim_Tested",
            "VAN - Vancomycin_Tested"
        ],
        "second_line": [
            "AMP - Ampicillin_Tested"
        ],
        "last_resort": [
            "MER - Meropenem_Tested"
        ]
    },

    "Clostridioides difficile infection": {
        "first_line": [
            "FID - Fidaxomicin_Tested"
        ],
        "second_line": [
            "VAN - Vancomycin_Tested"
        ],
        "last_resort": [
            "MTR - Metronidazol_Tested"
        ]
    }
}
