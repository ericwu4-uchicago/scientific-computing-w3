import numpy as np
import argparse
import matplotlib.pyplot as plt
import scipy.stats
import time

PAM250 = {
'A': {'A': 2,  'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K': -1, 'L': -2, 'M': -1, 'N':  0, 'P':  1, 'Q':  0, 'R': -2, 'S':  1, 'T':  1, 'V':  0, 'W': -6, 'Y': -3},
'C': {'A': -2, 'C': 12, 'D': -5, 'E':-5, 'F': -4, 'G': -3, 'H': -3, 'I': -2, 'K': -5, 'L': -6, 'M': -5, 'N': -4, 'P': -3, 'Q': -5, 'R': -4, 'S':  0, 'T': -2, 'V': -2, 'W': -8, 'Y':  0},
'D': {'A': 0,  'C': -5, 'D':  4, 'E': 3, 'F': -6, 'G':  1, 'H':  1, 'I': -2, 'K':  0, 'L': -4, 'M': -3, 'N':  2, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
'E': {'A': 0,  'C': -5, 'D':  3, 'E': 4, 'F': -5, 'G':  0, 'H':  1, 'I': -2, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
'F': {'A': -3, 'C': -4, 'D': -6, 'E':-5, 'F':  9, 'G': -5, 'H': -2, 'I':  1, 'K': -5, 'L':  2, 'M':  0, 'N': -3, 'P': -5, 'Q': -5, 'R': -4, 'S': -3, 'T': -3, 'V': -1, 'W':  0, 'Y':  7},
'G': {'A': 1,  'C': -3, 'D':  1, 'E': 0, 'F': -5, 'G':  5, 'H': -2, 'I': -3, 'K': -2, 'L': -4, 'M': -3, 'N':  0, 'P':  0, 'Q': -1, 'R': -3, 'S':  1, 'T':  0, 'V': -1, 'W': -7, 'Y': -5},
'H': {'A': -1, 'C': -3, 'D':  1, 'E': 1, 'F': -2, 'G': -2, 'H':  6, 'I': -2, 'K':  0, 'L': -2, 'M': -2, 'N':  2, 'P':  0, 'Q':  3, 'R':  2, 'S': -1, 'T': -1, 'V': -2, 'W': -3, 'Y':  0},
'I': {'A': -1, 'C': -2, 'D': -2, 'E':-2, 'F':  1, 'G': -3, 'H': -2, 'I':  5, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -2, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -5, 'Y': -1},
'K': {'A': -1, 'C': -5, 'D':  0, 'E': 0, 'F': -5, 'G': -2, 'H':  0, 'I': -2, 'K':  5, 'L': -3, 'M':  0, 'N':  1, 'P': -1, 'Q':  1, 'R':  3, 'S':  0, 'T':  0, 'V': -2, 'W': -3, 'Y': -4},
'L': {'A': -2, 'C': -6, 'D': -4, 'E':-3, 'F':  2, 'G': -4, 'H': -2, 'I':  2, 'K': -3, 'L':  6, 'M':  4, 'N': -3, 'P': -3, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V':  2, 'W': -2, 'Y': -1},
'M': {'A': -1, 'C': -5, 'D': -3, 'E':-2, 'F':  0, 'G': -3, 'H': -2, 'I':  2, 'K':  0, 'L':  4, 'M':  6, 'N': -2, 'P': -2, 'Q': -1, 'R':  0, 'S': -2, 'T': -1, 'V':  2, 'W': -4, 'Y': -2},
'N': {'A': 0,  'C': -4, 'D':  2, 'E': 1, 'F': -3, 'G':  0, 'H':  2, 'I': -2, 'K':  1, 'L': -3, 'M': -2, 'N':  2, 'P':  0, 'Q':  1, 'R':  0, 'S':  1, 'T':  0, 'V': -2, 'W': -4, 'Y': -2},
'P': {'A': 1,  'C': -3, 'D': -1, 'E':-1, 'F': -5, 'G':  0, 'H':  0, 'I': -2, 'K': -1, 'L': -3, 'M': -2, 'N':  0, 'P':  6, 'Q':  0, 'R':  0, 'S':  1, 'T':  0, 'V': -1, 'W': -6, 'Y': -5},
'Q': {'A': 0,  'C': -5, 'D':  2, 'E': 2, 'F': -5, 'G': -1, 'H':  3, 'I': -2, 'K':  1, 'L': -2, 'M': -1, 'N':  1, 'P':  0, 'Q':  4, 'R':  1, 'S': -1, 'T': -1, 'V': -2, 'W': -5, 'Y': -4},
'R': {'A': -2, 'C': -4, 'D': -1, 'E':-1, 'F': -4, 'G': -3, 'H':  2, 'I': -2, 'K':  3, 'L': -3, 'M':  0, 'N':  0, 'P':  0, 'Q':  1, 'R':  6, 'S':  0, 'T': -1, 'V': -2, 'W':  2, 'Y': -4},
'S': {'A': 1,  'C':  0, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P':  1, 'Q': -1, 'R':  0, 'S':  2, 'T':  1, 'V': -1, 'W': -2, 'Y': -3},
'T': {'A': 1,  'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  0, 'H': -1, 'I':  0, 'K':  0, 'L': -2, 'M': -1, 'N':  0, 'P':  0, 'Q': -1, 'R': -1, 'S':  1, 'T':  3, 'V':  0, 'W': -5, 'Y': -3},
'V': {'A': 0,  'C': -2, 'D': -2, 'E':-2, 'F': -1, 'G': -1, 'H': -2, 'I':  4, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -1, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -6, 'Y': -2},
'W': {'A': -6, 'C': -8, 'D': -7, 'E':-7, 'F':  0, 'G': -7, 'H': -3, 'I': -5, 'K': -3, 'L': -2, 'M': -4, 'N': -4, 'P': -6, 'Q': -5, 'R':  2, 'S': -2, 'T': -5, 'V': -6, 'W': 17, 'Y':  0},
'Y': {'A': -3, 'C':  0, 'D': -4, 'E':-4, 'F':  7, 'G': -5, 'H':  0, 'I': -1, 'K': -4, 'L': -1, 'M': -2, 'N': -2, 'P': -5, 'Q': -4, 'R': -4, 'S': -3, 'T': -3, 'V': -2, 'W':  0, 'Y': 10}}

BLOSUM62={'C':{'C':9,'S':-1,'T':-1,'P':-3,'A':0,'G':-3,'N':-3,'D':-3,'E':-4,'Q':-3,'H':-3,'R':-3,'K':-3,'M':-1,'I':-1,'L':-1,'V':-1,'F':-2,'Y':-2,'W':-2},
          'S':{'C':-1,'S':4,'T':1,'P':-1,'A':1,'G':0,'N':1,'D':0,'E':0,'Q':0,'H':-1,'R':-1,'K':0,'M':-1,'I':-2,'L':-2,'V':-2,'F':-2,'Y':-2,'W':-3},
          'T':{'C':-1,'S':1,'T':4,'P':1,'A':-1,'G':1,'N':0,'D':1,'E':0,'Q':0,'H':0,'R':-1,'K':0,'M':-1,'I':-2,'L':-2,'V':-2,'F':-2,'Y':-2,'W':-3},
          'P':{'C':-3,'S':-1,'T':1,'P':7,'A':-1,'G':-2,'N':-1,'D':-1,'E':-1,'Q':-1,'H':-2,'R':-2,'K':-1,'M':-2,'I':-3,'L':-3,'V':-2,'F':-4,'Y':-3,'W':-4},
          'A':{'C':0,'S':1,'T':-1,'P':-1,'A':4,'G':0,'N':-1,'D':-2,'E':-1,'Q':-1,'H':-2,'R':-1,'K':-1,'M':-1,'I':-1,'L':-1,'V':-2,'F':-2,'Y':-2,'W':-3},
          'G':{'C':-3,'S':0,'T':1,'P':-2,'A':0,'G':6,'N':-2,'D':-1,'E':-2,'Q':-2,'H':-2,'R':-2,'K':-2,'M':-3,'I':-4,'L':-4,'V':0,'F':-3,'Y':-3,'W':-2},
          'N':{'C':-3,'S':1,'T':0,'P':-2,'A':-2,'G':0,'N':6,'D':1,'E':0,'Q':0,'H':-1,'R':0,'K':0,'M':-2,'I':-3,'L':-3,'V':-3,'F':-3,'Y':-2,'W':-4},
          'D':{'C':-3,'S':0,'T':1,'P':-1,'A':-2,'G':-1,'N':1,'D':6,'E':2,'Q':0,'H':-1,'R':-2,'K':-1,'M':-3,'I':-3,'L':-4,'V':-3,'F':-3,'Y':-3,'W':-4},
          'E':{'C':-4,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':2,'E':5,'Q':2,'H':0,'R':0,'K':1,'M':-2,'I':-3,'L':-3,'V':-3,'F':-3,'Y':-2,'W':-3},
          'Q':{'C':-3,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':0,'E':2,'Q':5,'H':0,'R':1,'K':1,'M':0,'I':-3,'L':-2,'V':-2,'F':-3,'Y':-1,'W':-2},
          'H':{'C':-3,'S':-1,'T':0,'P':-2,'A':-2,'G':-2,'N':1,'D':1,'E':0,'Q':0,'H':8,'R':0,'K':-1,'M':-2,'I':-3,'L':-3,'V':-2,'F':-1,'Y':2,'W':-2},
          'R':{'C':-3,'S':-1,'T':-1,'P':-2,'A':-1,'G':-2,'N':0,'D':-2,'E':0,'Q':1,'H':0,'R':5,'K':2,'M':-1,'I':-3,'L':-2,'V':-3,'F':-3,'Y':-2,'W':-3},
          'K':{'C':-3,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':-1,'E':1,'Q':1,'H':-1,'R':2,'K':5,'M':-1,'I':-3,'L':-2,'V':-3,'F':-3,'Y':-2,'W':-3},
          'M':{'C':-1,'S':-1,'T':-1,'P':-2,'A':-1,'G':-3,'N':-2,'D':-3,'E':-2,'Q':0,'H':-2,'R':-1,'K':-1,'M':5,'I':1,'L':2,'V':-2,'F':0,'Y':-1,'W':-1},
          'I':{'C':-1,'S':-2,'T':-2,'P':-3,'A':-1,'G':-4,'N':-3,'D':-3,'E':-3,'Q':-3,'H':-3,'R':-3,'K':-3,'M':1,'I':4,'L':2,'V':1,'F':0,'Y':-1,'W':-3},
          'L':{'C':-1,'S':-2,'T':-2,'P':-3,'A':-1,'G':-4,'N':-3,'D':-4,'E':-3,'Q':-2,'H':-3,'R':-2,'K':-2,'M':2,'I':2,'L':4,'V':3,'F':0,'Y':-1,'W':-2},
          'V':{'C':-1,'S':-2,'T':-2,'P':-2,'A':0,'G':-3,'N':-3,'D':-3,'E':-2,'Q':-2,'H':-3,'R':-3,'K':-2,'M':1,'I':3,'L':1,'V':4,'F':-1,'Y':-1,'W':-3},
          'F':{'C':-2,'S':-2,'T':-2,'P':-4,'A':-2,'G':-3,'N':-3,'D':-3,'E':-3,'Q':-3,'H':-1,'R':-3,'K':-3,'M':0,'I':0,'L':0,'V':-1,'F':6,'Y':3,'W':1},
          'Y':{'C':-2,'S':-2,'T':-2,'P':-3,'A':-2,'G':-3,'N':-2,'D':-3,'E':-2,'Q':-1,'H':2,'R':-2,'K':-2,'M':-1,'I':-1,'L':-1,'V':-1,'F':3,'Y':7,'W':2},
          'W':{'C':-2,'S':-3,'T':-3,'P':-4,'A':-3,'G':-2,'N':-4,'D':-4,'E':-3,'Q':-2,'H':-2,'R':-3,'K':-3,'M':-1,'I':-3,'L':-2,'V':-3,'F':1,'Y':2,'W':11}
          }

def calcbest(above, diag, left, scope, matrixcost, gap):
    '''computes a single entry of DP matrix'''
    values = np.zeros(3)
    values[0] = above + gap
    values[1] = left + gap
    values[2] = diag + matrixcost
    highest = max(values)
    if (scope == 'local' and highest < 0):
        return 0
    return highest

def find_align(first, second, scope, matrix, gap):
    '''computes cost matrix'''
    costs = np.zeros((1 + len(first), 1 + len(second)))
    #initial setup for global cost matrix
    if (scope == 'global'):
        for i in range(1, 1+len(first)):
            costs[i][0] = i * gap
        for i in range(1, 1+len(second)):
            costs[0][i] = i * gap
    #compute scores for all other entries
    for i in range(1, 1+len(first)):
        for j in range(1, 1 + len(second)):
            matrixcost = matrix[first[i-1].upper()][second[j-1].upper()]
            costs[i][j] = calcbest(costs[i-1][j], costs[i-1][j-1], costs[i][j-1], scope, matrixcost, gap)
    return costs

def backstep(costs, firstloc, secondloc, scope, gap): 
    '''finds a step on the path of a given cost matrix'''
    if (costs[firstloc][secondloc] == gap + costs[firstloc - 1][secondloc]):
        return 1
    if (costs[firstloc][secondloc] == gap + costs[firstloc][secondloc - 1]):
        return 2
    return 3

def find_traversal(costs, first, second, scope, gap):
    '''finds the traversal along the cost matrix and print wanted output'''
    score = 0
    firstform = ""
    secondform = ""
    firstloc = -1
    secondloc = -1
    matches = 0
    pathlen = 0
    if (scope == 'global'):
        firstloc = len(first)
        secondloc = len(second)
        score = costs[firstloc][secondloc]
        while (firstloc > 0 or secondloc > 0):
            inc = backstep(costs, firstloc, secondloc, scope, gap)
            if (inc == 1):
                firstform = first[firstloc] + firstform
                secondform = '-' + secondform
                firstloc += -1
            elif (inc == 2):
                secondform = second[secondloc] + secondform
                firstform = '-' + firstform
                secondloc += -1
            elif (inc == 3):
                firstloc += -1
                secondloc += -1
                firstform = first[firstloc] + firstform
                secondform = second[secondloc] + secondform
                matches += 1
            pathlen += 1
                
    elif (scope == "local"):
        locs = np.unravel_index(costs.argmax(), costs.shape)
        #https://stackoverflow.com/questions/3584243/get-the-position-of-the-largest-value-in-a-multi-dimensional-numpy-array
        firstloc = locs[0]
        secondloc = locs[1]
        score = costs[firstloc][secondloc]
        while (costs[firstloc][secondloc] != 0):
            inc = backstep(costs, firstloc, secondloc, scope, gap)
            if (inc == 1):
                firstform = first[firstloc] + firstform
                secondform = '-' + secondform
                firstloc += -1
            elif (inc == 2):
                secondform = second[secondloc] + secondform
                firstform = '-' + firstform
                secondloc += -1
            elif (inc == 3):
                firstloc += -1
                secondloc += -1
                firstform = first[firstloc] + firstform
                secondform = second[secondloc] + secondform
                matches += 1
            pathlen += 1
    print(f"seq1: {firstform}\nseq2: {secondform}\nscore: {score}\nsequence identity: {matches/pathlen}")

def parse_fasta(filename):
    '''returns sequence in given file'''
    #we are lazily going to assume this thing works and only has 1 thing 
    f = open(filename, 'r')
    line = f.readline().strip() 
    desc = ""
    seq = ""
    while line:
        if line[0] == '>':
            if desc != "":
                seq = ""
            desc = line[1:]
        else:
            seq += line
        line = f.readline().strip()
    f.close()
    return seq
    

argParser = argparse.ArgumentParser() #https://www.tutorialspoint.com/python/python_command_line_arguments.htm
argParser.add_argument("-seq1 ", "--seq1", help="first sequence")
argParser.add_argument("-seq2 ", "--seq2", help="second sequence")
argParser.add_argument("-type ", "--type", help="global or local")
argParser.add_argument("-matrix ", "--matrix", help="weight matrix")
argParser.add_argument("-gap_penalty ", "--gap", help="gap penalty should be negative")
#define all arguments

args = argParser.parse_args()

#parse arguments from user input
if (args.matrix == "pam250"):
    matrix = PAM250
else:
    matrix = BLOSUM62
first = parse_fasta(args.seq1)
second = parse_fasta(args.seq2)
scope = args.type
gap = int(args.gap)
#compute cost matrix
costs = find_align(first, second, scope, matrix, gap) 
#print(costs)
#get sequence and related
find_traversal(costs, first, second, scope, gap)                