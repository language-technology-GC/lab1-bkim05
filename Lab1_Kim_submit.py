#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:37:33 2020

@author: Boram

path similarity,
Leacock-Chodorow similarity: it return a score denoting the similarity of the two `Synset` objects,
            normally greater than 0. None is returned if no connecting path
            could be found. If a `Synset` is compared with itself, the
            maximum score is returned, which varies depending on the taxonomy
            depth.

Wu-Palmer similarity,
Resnik similarity,
Jiang-Conrath similarity, and
Lin similarity.

"""


def computeCorr( compfactor, datfile ):
    ''' '''
    import pandas as pd
    from scipy.stats import spearmanr
    
    dat = pd.read_csv(datfile)
            
    collist = ['PathSim', 'WUP', 'LCH', 'RES', 'JNC', 'Lin']

    rdf = pd.DataFrame(columns= ['Methods', f'Coeff_{compfactor}'] )

    for c in range(len(collist)):
        # choose the column
        cn = collist[c]     
        
        # For the columne to be anlyzed, exclude the row that contains NA
        df = dat[ dat[cn].notna() ]
                
        coef, _ = spearmanr(df[compfactor], df[cn])
        # Coefficient to 4 decimals
        rdf.loc[c, ['Methods', f'Coeff_{compfactor}']] = [cn, round(coef, 4)]        
    rdf.to_csv(f'CorrelationTable_with_{compfactor}.csv', index=False)
    print(rdf)
    print(f'Out file name: CorrelationTable_with_{compfactor}.csv')



'''
Issue & solution

ppmi script provides unsorted word pair result different from ws353.tsv file. 
To solve this issue a word pair was saved as a frozen set in a dictionary together with corresponding value.
Then dictionaries were merged using the common keys. 
'''


def df2Dict( df_path:str ):
    import pandas as pd    
    ''' Convert data frame to dictionary with a word pair as frozen set. 
        It makes easier to merge the results with unsorted word pairs'''
    
    if 'tsv' in df_path:
        df = pd.read_csv( df_path, sep='\t', header=None)
    
    else: 
        df = pd.read_csv( df_path )
    
    if not 'word1' in df.columns.tolist():
        df.rename(columns={0:'word1', 1:'word2'}, inplace= True)
    
    # strip the key values 
    df.word1 = df['word1'].astype(str).map(lambda x: x.strip().lower() )
    df.word2 = df['word2'].astype(str).map(lambda x: x.strip().lower() )
    df.set_index(['word1', 'word2'], inplace=True)

    dfdict = {frozenset(k[0]): list(k[1:])
              for k in df.itertuples() }
    return dfdict



def mergeDict(dict1, dict2):
   ''' Merge two dictionaries keeping only common keys with values as list
       dict1 should be the smaller dictionary wich has the subset of the keys in dict2'''
   dict3 = dict1
   for key, value in dict3.items():
       if key in dict2:
          dict3[key] =  value + dict2[key]
   return dict3



def dict2df(compfactor, merged_dict ):
    allCol = [compfactor] + ['PathSim', 'WUP', 'LCH', 'RES', 'JNC', 'Lin']
    outdf = pd.DataFrame.from_dict(merged_dict, orient='index')\
                        .rename(columns={n: name for n, name in enumerate(allCol)})                   
    outdf.to_csv(f'{compfactor}_allMethods_ordered.csv')
    print(f'Out file name: {compfactor}_allMethods_ordered.csv')
    return outdf


#%% Part 1 Compare the human judgment with all 6 methods of WordNet similarity result

from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
import pandas as pd

# Load the data
dat = pd.read_csv('ws353.tsv', sep='\t', header=None).rename( columns = {0:'word1', 1:'word2', 2:'Human'})

# save words pairs for Part 2 & 3 analysis
pairdoc = dat[ ['word1', 'word2'] ]
pairdoc.to_csv('pair_doc.tsv', sep="\t", index=False, header=False )

''' Choose synset
Computing the lch similarity requires the words to have the same part of speech.
Although the other methods do not require two synsents to have the same POS, 
to accurately compare the performance of the 6 methods, the same synset from each word was used for analysis   
'''

for i in dat.index:
    #print(i)
    w1, w2 = dat.iloc[i, [0,1] ]

    # Process with the most frequent sense if two word has the same POS      
    # Matching POS of two synsets if they differ each other
    # It checks how many unique POS each synset contains
    # It chose the most frequent sense from each word with the same POS
    
    w1s = wordnet.synsets(w1)[0]
    w2s = wordnet.synsets(w2)[0]
    
    if not w1s.name().split('.')[0] == w2s.name().split('.')[0]:
         posw1 = pd.Series( [x.name().split('.')[1] for x in wordnet.synsets(w1)] )
         posw2 = pd.Series( [x.name().split('.')[1] for x in wordnet.synsets(w2)] )
                
         # Find the common POS         
         c_pos = list(set(posw1)&set(posw2))
         
         # Find the index of the most frequent sense with the common POS
         w1idx = posw1[posw1.isin(c_pos)].index[0]
         w2idx = posw2[posw2.isin(c_pos)].index[0]

         w1s = wordnet.synsets(w1)[w1idx]
         w2s = wordnet.synsets(w2)[w2idx]
    
    # Compute the 6 methods
    dat.loc[i, 'PathSim'] = wordnet.path_similarity(w1s, w2s, simulate_root=False)
    dat.loc[i, 'WUP'] = wordnet.wup_similarity(w1s, w2s, simulate_root=False)    
    dat.loc[i, 'LCH'] = wordnet.lch_similarity(w1s, w2s, simulate_root=False)
    
    ic = wordnet_ic.ic('ic-bnc-resnik-add1.dat')
    dat.loc[i, 'RES'] = wordnet.res_similarity(w1s, w2s, ic)
    dat.loc[i, 'JNC'] = wordnet.jcn_similarity(w1s, w2s, ic)
    dat.loc[i, 'Lin'] = wordnet.lin_similarity(w1s, w2s, ic)

# Save the result of Human judgment and 6 methods
dat.to_csv('Human_allMethods.csv', index=False)

# count the number of covered pairs
coverdW = dat.count().rename_axis('Methods').reset_index(name='# coverved Pair')
coverdW.to_csv('WordNet_covered_pair.csv', index=False )
'''
WordNet analysis covered the most of the word pairs, PathSim, WUP and LCH missed one word pair.
RES, JNC and Lin covered all the pairs.
'''

# Save the result of 6 methods
dat.drop('Human', axis=1, inplace=True)
dat.to_csv('allMethods_Similarity.csv', index=False)

# Compute correlation between Human judgment and WordNet similarity result
datfile = 'Human_allMethods.csv'
compfactor = 'Human'

computeCorr( compfactor, datfile )
'''
Correlation between human judgment and WordNet Similarity ranged from 0.45 (Path, LCH) to 0.58 (RES, LIN).
'''

#%% Part 2 Compare PPMI with all 6 methods

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

pairs = pd.read_csv('pair_doc.tsv', sep="\t", header=None )

# Save the list of stopwords
stopwordsList = list(stopwords.words('english'))

# tokenize the data and save it
with open("news.2018.en.shuffled.deduped") as in_file, open("tokenslines.txt", "w") as out_file:
    for line in in_file:
        words = word_tokenize(line)      
        for word in words:
            # remove punctuation and stopwords
            if word.isalpha() and word not in stopwordsList:
               # save each token separated with a space
                out_file.write(f'{word} ')
        # Put newline for each sentence
        out_file.write("\n")
        


"""
run with command line

python ppmi.py  --results_path ./ppmiResultTable.tsv\
                --pairs_path ./pair_doc.tsv\
                --tok_path ./tokenslines.txt

result

INFO: 277 words tracked
INFO: 203 pairs tracked
INFO: 208457753 tokens counted
INFO: 185 pairs covered (out of 203 words) 

PPMI analysis covered around 91% of the word pairs (185 pairs out of 203 words) 

"""
# conver to dictionary
PPMIdict = df2Dict( 'ppmiResultTable.tsv' )
allmethodsDict = df2Dict( 'allMethods_Similarity.csv' )

# merge two dictionary based on the common keys
ppmi_methods_dict = mergeDict(PPMIdict, allmethodsDict)

# convert a merged dict to dataframe 
compfactor = 'PPMI'
ppmi_methods_df = dict2df(compfactor, ppmi_methods_dict )

# Compute correlation
datfile = 'PPMI_allMethods_ordered.csv'
computeCorr( compfactor, datfile )

'''
Correlation result between PPMI and all 6 methods ranged from -0.02 (Lin) to 0.35 (LCH).
Overall the coefficient was lower than that of human judgment compared in Part 1.
'''


#%% Part 3

"""
run with command line

python word2vec.py --results_path ./cosSimTable.tsv --pairs_path ./pair_doc.tsv --tok_path ./tokenslines.txt

"""

#Convert to dictionary
cosSimDict = df2Dict( 'cosSimTable.tsv' )
allmethodsDict = df2Dict( 'allMethods_Similarity.csv' )

# merge two dictionary based on the common keys
cosSim_methods_dict = mergeDict(cosSimDict, allmethodsDict)

# convert a merged dict to dataframe 
compfactor= 'cosSim'
cosSim_method_df = dict2df(compfactor, cosSim_methods_dict )

# Compute correlation
datfile = 'cosSim_allMethods_ordered.csv'
computeCorr( compfactor, datfile )


'''
CosSimilarity presents relatively higher coefficient comparable to 
Part1 correlation between human judment and all 6 methods.
The result rages from 0.44 (Path similarity) to 0.55 (RES).
'''

'''Overall, it was great lab to explore word similarity and try various methods.
It wondered how other people choose appropriate synsets and 
also what is crucial factors for human to judge similarity and how they differ from machine.

I prefer cosine similarity as it is intuitive and it provides flexibility to apply multi-dimensional data.
I'd like to apply this method to acoustic data measuring vowel formant dispersion 
and multi-dimensional articulatory data.'
'''