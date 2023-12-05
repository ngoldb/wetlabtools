"""
This module contains simple helper functions
"""
import statistics



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


def find_consecutive_blocks(s, threshold: float=0.1):
    """
    s: series of numbers (e.g. list, pd.Series)
    threshold: float, sensitivity for identifying breaks
    return_indices: bool, 

    Function to identify blocks of consecutive data in a series of numbers.
    Returns a list of indices of the blocks: [start, end]
    """

    # calculate the difference between the numbers in the series
    diff = []
    for i in range(1, len(s)):
        diff.append(s[i] - s[i-1])

    # second quantile of distribution of the differences
    q2 = statistics.quantiles(diff)[1]

    # finding blocks
    lower = 0
    blocks = []

    for i, x in enumerate(diff):

        # append i if difference is bigger than cutoff
        if x > q2 + q2 * threshold:
            blocks.append([lower, i])
            lower = i + 1

    # appending last block
    blocks.append([lower, len(s) - 1]) # accounting for 0-indexing

    return blocks