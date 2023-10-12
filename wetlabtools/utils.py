"""
This module contains simple helper functions
"""

# useful functions for data normalization
def normalize_percent_max(y: float, max_value: float, min_value: float):
    '''Function to normalize data to a scale 0 - 100% of data range'''
    return ((y - min_value) / (max_value - min_value)) * 100

def rel_scale(y: float, max_value: float):
    '''Function to scale data to a axis 0 to 1 where the maximal value is 1.'''
    return y / max_value


# useful dictonaries and lists
aa_1_letter = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_3_letter = ['Ala', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His', 'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg', 'Ser', 'Thr', 'Val', 'Trp', 'Tyr']

aa_123 = {'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe', 'G': 'Gly',
          'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu', 'M': 'Met', 'N': 'Asn',
          'P': 'Pro', 'Q': 'Gln', 'R': 'Arg', 'S': 'Ser', 'T': 'Thr', 'V': 'Val',
          'W': 'Trp', 'Y': 'Tyr'
          }

aa_321 = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Glu': 'E',
          'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
          'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
          'Tyr': 'Y', 'Val': 'V'
          }