#!/usr/bin/env python3

'''Script for computing sequence alignments using pair HMM.
Arguments:
    f - FASTA file with sequences in FASTA format.
    s - JSON with the score matrix for transition probabilities.
    d - delta for the alignment.

Outputs:
    Prints alignment to console.

Example Usage:
    python pHMM.py -f sequences.fasta -s amino.json -d 1
'''

import argparse
import json
import numpy as np

'''Computes the transition values according to the methods in the paper
Arguments:
    r: Indel rate (float)
    t: Time (float)
    a: indel length parameter (float)
    d: delta (int)
Returns:
    transition_matrix: dictionary of transition states'''
def calculate_transitions(r, t, a, d):
    transition_matrix = {'s':{"M": 0.0, "I": 0.0, "D": 0.0, "e": 0.0},
        'M':{"M": 0.0, "I": 0.0, "D": 0.0, "e": 0.0},
        'I':{"M": 0.0, "I": 0.0, "D": 0.0, "e": 0.0},
        'D':{"M": 0.0, "I": 0.0, "D": 0.0, "e": 0.0}}
    P_id = 1- np.e**(-2*r*t)
    P_prime_id = 1-(1/(2*r*t))*(1-np.e**(-2*r*t))
    E_0 = 1
    E_00 = 1 - P_id*(1 - (1-a)/(4+4*a))*P_prime_id
    E_10 = (1-a) + ((a*(1-a))/(2+2*a))*P_prime_id - ((7-7*a)/8)*P_id
    E_11 = a + ((a**2)/(1-a**2))*P_prime_id + ((1-a)/2)*P_id
    E_12 = (a**2/(2+2*a))*P_prime_id + ((3-3*a)/8)*P_id
    E_1 = 1 + (a/(2-2*a))*P_prime_id
    E_01 = 1/2*P_id*((1-P_prime_id) + 1/2*(a/(a+1))*P_prime_id + 1/2*P_prime_id)
    E_02 = 1/2*P_id*(1 + (a/(a+1)*1/2*P_prime_id))
    
    transition_matrix['s']['M'] = np.log(E_00)
    transition_matrix['s']['I'] = np.log((E_01 + E_02)/(2*E_0))
    transition_matrix['s']['D'] = np.log((E_01 + E_02)/(2*E_0))
    transition_matrix['s']['e'] = np.log(d*E_00)
    transition_matrix['M']['M'] = np.log(E_00)
    transition_matrix['M']['I'] = np.log((E_01 + E_02)/(2*E_0))
    transition_matrix['M']['D'] = np.log((E_01 + E_02)/(2*E_0))
    transition_matrix['M']['e'] = np.log(d*E_00)
    transition_matrix['I']['M'] = np.log(E_10/E_1)
    transition_matrix['I']['I'] = np.log(E_11/E_1)
    transition_matrix['I']['D'] = np.log(E_12/E_1)
    transition_matrix['I']['e'] = np.log(d*(E_10/E_1))
    transition_matrix['D']['M'] = np.log(E_10/E_1)
    transition_matrix['D']['I'] = np.log(E_12/E_1)
    transition_matrix['D']['D'] = np.log(E_11/E_1)
    transition_matrix['D']['e'] = np.log(d*(E_10/E_1))
    return transition_matrix

'''Aligns sequences given three traceback matrices
Arguments:
    x: the first string we're aligning
    y: the second string we're aligning
    t_M: traceback matrix for match state
    t_X: traceback matrix for x matched with gap
    t_Y: traceback matrix for y matched with gap
    max: int representing which matrix terminates with the highest value
Returns:
    a_x: the string for the alignment of x's sequence
    a_y: the string for the alignment of y's sequence
'''
def traceback(x, y, t_m, t_x, t_y, max):
    ''' Complete this function. '''
    dict = {0: t_m, 1: t_x, 2: t_y}
    final_x = x[-1]
    final_y = y[-1]
    if max == 1:
        final_y = '-'
    elif max == 2:
        final_x = '-'
    size_x = len(x)
    size_y = len(y)
    curr_mat = dict[max]
    curr_ind_x = size_x-1
    curr_ind_y = size_y-1
    i = curr_mat[curr_ind_x][curr_ind_y]
    while (curr_ind_x > 0 and curr_ind_y > 0): 
        if i != -1:
            if (i == 0):
                final_x = x[curr_ind_x-1] + final_x
                final_y = y[curr_ind_y-1] + final_y
                curr_ind_x = curr_ind_x-1
                curr_ind_y = curr_ind_y-1
                curr_mat = dict[t_m[curr_ind_x][curr_ind_y]]
                i = t_m[curr_ind_x][curr_ind_y]
            elif (i == 1):
                final_x = x[curr_ind_x-1] + final_x
                final_y = '-' + final_y
                curr_ind_x = curr_ind_x-1
                i = t_x[curr_ind_x-1][curr_ind_y]
                curr_mat = dict[i]
            elif (i ==2):
                final_x = '-' + final_x
                final_y = y[curr_ind_y - 1] + final_y
                curr_ind_y = curr_ind_y-1
                i = t_y[curr_ind_x][curr_ind_y-1]
                curr_mat = dict[i]
    if curr_ind_x != 0:
        final_x = x[curr_ind_x-1] + final_x
        final_y = '-' + final_y
        curr_ind_x -= 1
    if curr_ind_y != 0:
        final_x = '-' + final_x
        final_y = y[curr_ind_y-1] + final_y
        curr_ind_y-= 1
    return final_x, final_y


'''Computes viterbi on string x and y.
Arguments:
    x: the first string we're aligning
    y: the second string we're aligning
    transition: the transition matrix
    emission: the emission matrix for MATCH state
    emission_one: emission values out of insert and delete states
Returns:
    score: the score of the optimal sequence alignment
    a_x: the aligned first string
    a_y: the aligned second string
The latter two are computed using the above traceback method.
'''
def viterbi(x, y, transition, emission, emission_one):
    ''' Recurrence matrix, redefine/use as necessary. '''
    size_x = len(x)+1
    size_y = len(y)+1
    v_m = np.zeros((size_x,size_y))
    v_x = np.zeros((size_x,size_y))
    v_y = np.zeros((size_x,size_y))
    
    ''' Traceback matrix, redefine/use as necessary. '''
    t_m = np.zeros((size_x, size_y))
    t_x = np.zeros((size_x, size_y))
    t_y = np.zeros((size_x, size_y))

    """ Base case: v_m(0,0) is 0 by defination and v_x, v_y are supposed to be log(0)"""
    v_m[0,0] = 0
    v_x[0,0] = float('-inf')
    v_y[0,0] = float('-inf')
    t_m[0,0] = -1
    t_x[0,0] = -1
    t_y[0,0] = -1
    """ Base case: populate 0th row and column with 's' to state transitions."""
    for i in range(1,size_x):
        v_m[i][0] = transition["s"]["M"]
        v_x[i][0] = transition["s"]["I"]
        v_y[i][0] = transition["s"]["D"]
    for i in range(1,size_y):
        v_m[0][i] = transition["s"]["M"]
        v_x[0][i] = transition["s"]["I"]
        v_y[0][i] = transition["s"]["D"]
    """ Begin calculating values and populating the matrix"""
    """ 0 represents match, 1 represents x and 2 represents y"""
    for i in range(1,size_x):
        for j in range (1, size_y):
            x_amino = x[i-1]
            y_amino = y[j-1]
            ind_x = i-1
            ind_y = j-1
            p_ij = np.log(emission[x_amino][y_amino])
            q_i = np.log(emission_one[x_amino])
            q_j = np.log(emission_one[y_amino])
            max_m = p_ij + transition["M"]["M"]+v_m[i-1][j-1]
            max_ind_m = 0
            m_x = p_ij + transition["I"]["M"] + v_x[i-1][j-1]
            m_y = p_ij + transition["D"]["M"] + v_y[i-1][j-1]
            if m_x > max_m:
                max_m = m_x
                max_ind_m = 1
            if m_y > max_m:
                max_m = m_y
                max_ind_m = 2
            v_m[i][j] = max_m
            t_m[i][j] = max_ind_m

            max_x = q_i + transition["M"]["I"] + v_m[i-1][j]
            max_ind_x = 0
            m_x = q_i + transition["I"]["I"] + v_x[i-1][j]
            if m_x > max_x:
                max_x = m_x
                max_ind_x = 1
            v_x[i][j] = max_x
            t_x[i][j] = max_ind_x

            max_y = q_j + transition["M"]["D"] + v_m[i][j-1]
            max_ind_y = 0
            m_y = q_j + transition["D"]["D"] + v_y[i][j-1]
            if m_y > max_y:
                max_y = m_y
                max_ind_y = 2
            v_y[i][j] = max_y
            t_y[i][j] = max_ind_y
    max_mat = 0
    max_ind = 0
    term_m = transition["M"]["e"] + v_m[size_x-1][size_y-1]
    term_x = transition["I"]["e"] + v_x[size_x-1][size_y-1]
    term_y = transition["D"]["e"] + v_y[size_x-1][size_y-1]
    if max([term_m, term_x, term_y]) == term_m:
        max_mat = term_m 
        max_ind = 0
    elif max([term_m, term_x, term_y]) == term_x:
        max_mat = term_x
        max_ind = 1
    elif max([term_m, term_x, term_y]) == term_y:
        max_mat = term_y 
        max_ind = 2
            

    a_x, a_y = traceback(x, y, t_m, t_x, t_y, max_ind)
    return max_mat, (a_x, a_y)

'''Prints two aligned sequences formatted for convenient inspection.
Arguments:
    a_x: the first sequence aligned
    a_y: the second sequence aligned
Outputs:
    Prints aligned sequences (80 characters per line) to console
'''
def print_alignment(a_x, a_y):
    assert len(a_x) == len(a_y), "Sequence alignment lengths must be the same."
    for i in range(1 + (len(a_x) // 80)):
        start = i * 80
        end = (i + 1) * 80
        print(a_x[start:end])
        print(a_y[start:end])
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate sequence alignments for two sequences with a linear gap penalty.')
    parser.add_argument('-f', action="store", dest="f", type=str, required=True)
    parser.add_argument('-s', action="store", dest="s", type=str, required=True)
    parser.add_argument('-d', action="store", dest="d", type=float, required=True)

    args = parser.parse_args()
    fasta_file = args.f
    score_matrix_file = args.s
    d = args.d
    
    with open(fasta_file) as f:
        _, x, _, y = [line.strip() for line in f.readlines()]
    with open(score_matrix_file) as f:
        s = json.loads(f.readlines()[0])

    emission_one = {
        "A": 0.074,
        "R": 0.042,
        "N": 0.044,
        "D": 0.059,
        "C": 0.033,
        "Q": 0.058,
        "E": 0.037,
        "G": 0.074,
        "H": 0.029,
        "I": 0.038,
        "L": 0.076,
        "K": 0.072,
        "M": 0.018,
        "F": 0.040,
        "P": 0.050,
        "S": 0.081,
        "T": 0.062,
        "W": 0.013,
        "Y": 0.033,
        "V": 0.068}

    transitions = calculate_transitions(0.0165, 0.876, 0.554, d)

    score, (a_x, a_y) = viterbi(x, y, transitions, s, emission_one)
    print("Alignment:")
    print_alignment(a_x, a_y)
    print("Score: " + str(score))


if __name__ == "__main__":
    main()
