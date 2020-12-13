import numpy as np
import pandas as pd

# Adapted from https://github.com/niranjangavade98/DNA-to-Protein-sequence
# and https://github.com/av1659/fbgan/blob/master/utils/bio_utils.py

DNA_protein_MAP = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': 'P', 'TAG': 'P',
    'TGC': 'C', 'TGT': 'C', 'TGA': 'P', 'TGG': 'W',
}

protein_DNA_MAP = {v: k for k, v in DNA_protein_MAP.items()}
protein_DNA_MAP['P'] = 'TAG'


def parse(sequences):
    if type(sequences) == str:
        parsed = np.array([a for a in sequences])
        return parsed

    parse = lambda seq: np.array([a for a in seq])
    parsed = pd.DataFrame(sequences).iloc[:, 0].apply(parse).to_numpy()

    return parsed

def protein_to_DNA(protein_sequences):
    global protein_DNA_MAP

    parsed = parse(protein_sequences)

    DNA_sequences = []

    if type(parsed[0]) in (str, np.str_):
        DNA_merged = ''.join([a for a in parsed])
        DNA_sequences += ['ATG' + DNA_merged + "TAG"]
        return DNA_sequences

    for seq in parsed:
        DNA = [protein_DNA_MAP[a] for a in seq]
        DNA_merged = ''.join([a for a in DNA])
        DNA_sequences += ['ATG' + DNA_merged + "TAG"]

    DNA_sequences = np.array(DNA_sequences).reshape(-1, 1)
    return DNA_sequences





def translate(seq):
    table = DNA_protein_MAP
    table['TAA'], table['TGA'], table['TAG'] = '','',''
   
    protein = ""
    seq = seq[0].split('P')[0]
    for i in range(0, len(seq), 3):
        try:
            codon = seq[i:i + 3]
            protein += table[codon]
        except:
            protein += ""
    return protein


def DNA_to_protein(sequences):
    result = []
    for seq in sequences:
        result.append(translate(seq))
    return result



