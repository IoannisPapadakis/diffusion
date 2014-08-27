"""
Run Text Similarity Algorithm on Biotech and AI Papers and Patents
Nathan Goldschlag
August 25, 2014
Version 1.0
Written in Python 2.7

This python program calculates the syntactic similarity between biotech and AI abstracts and biotech and AI patents. 

"""
myModulesPath = 'C:\\Users\\ngold\\Documents\\Python Library\\GitHub\\generic_methods'
## IMPORT LIBRARIES
import sys
sys.path.append(myModulesPath)
import urllib
import urllib2
import re
from bs4 import BeautifulSoup
import csv
import string
import operator
import nltk
import numpy as np
import time
import pandas as pd
from unicodedata import normalize
import matplotlib.pyplot as plt
import math
from collections import Counter
import random
import multiprocessing
from generic_python_functions import *

## Parameters
difPath = 'd:/diffusion_data'
# longest common subsequence weights
wlcs = 0.34
wmclcs1 = 0.33
wmclcsn = 0.33


stopWords = []
with open(myModulesPath+'/english_stopwords.txt', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        for i in row:
            stopWords.append(i)

puncToStrip = [".",",","?","\"",":",";","'s"]
def cleanWebInput(text):
    text = text.strip()
    text = text.replace("<summary>","")
    text = text.replace("</summary>","")
    text = text.replace("<abstract>","")
    text = text.replace("<bibno>","")
    text = text.replace("<page_from>","")
    text = text.replace("<page_to>","")
    text = text.replace("<jrnl_code>","")
    text = text.replace("<volume>","")
    text = text.replace("<issue>","")
    text = text.replace("<italic>","")
    text = text.replace("</italic>","")
    text = text.replace("<a>","")
    text = text.replace("</a>","")
    text = text.replace("<ul>","")
    text = text.replace("</ul>","")
    text = text.replace("<i>","")
    text = text.replace("</i>","")
    text = text.replace("<b>","")
    text = text.replace("</b>","")
    text = text.replace("<br>","")
    text = text.replace("</br>","")
    text = text.replace("<br />","")
    text = re.sub('\<h\w+\>','',text)
    text = re.sub('\</h\w+\>','',text)
    text = text.replace("<inline-equation>","")
    text = text.replace("</inline-equation>","")
    text = text.replace("<sup>","supscrpt")
    text = text.replace("</sup>","supscrpt")
    text = text.replace("<sub>","subscrpt")
    text = text.replace("</sub>","subscrpt")
    text = text.replace("<item>","")
    text = text.replace("</item>","")
    text = text.replace("<list>","")
    text = text.replace("</list>","")
    text = text.replace("<li>","")
    text = text.replace("</li>","")
    text = text.replace("<p>","")
    text = text.replace("</p>","")
    text = text.replace("\n"," ")
    text = text.replace("<div class=\"pubabstract\">","")
    text = text.replace("</div>","")
    text = text.replace("<author> <name>","")
    text = text.replace("</name> </author>","")
    text = text.replace("<subscrpt>", "subscrpt")
    text = text.replace("</subscrpt>", "subscrpt")
    text = text.replace("<supscrpt>", "supscrpt")
    text = text.replace("</supscrpt>", "supscrpt")
    text = re.sub('\<\w+\>','',text)
    text = re.sub('\</\w+\>','',text)
    text = re.sub('\&\#\d+\;','',text)
    text = re.sub('\&\w+\;','',text)
    return text

def scrub(text):
    text = text.strip()
    text = text.lower()
    text = text.translate(string.maketrans("",""), string.punctuation)
    text = text.replace("  ", " ")
    return text

def stopWordScrub(splitText):
    newText = []
    for i in splitText:
        if i not in stopWords:
            newText.append(i)
    return newText

def lemma(splitText):
    lmtzr = nltk.stem.WordNetLemmatizer()
    for n,i in enumerate(splitText):
        splitText[n] = lmtzr.lemmatize(i)
    return splitText

def countWords(splitText):
    wordCount = {}
    for word in splitText:
        if word in wordCount:
            wordCount[word] += 1
        else:
            wordCount[word] = 1
    return sorted(wordCount.iteritems(), key=operator.itemgetter(1), reverse=True)


def lcs(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = \
                    max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    return round( (len(result))**2 / (float(len(a))*float(len(b))) ,3)

def mclcs1(s1,s2):
    clcs = ''
    i = 0
    while s1[i]==s2[i]:
        clcs = clcs + s1[i]
        i += 1
        if i > min(len(s1)-1,len(s2)-1):
            return round( (len(clcs))**2 / (float(len(s1))*float(len(s2))) ,3)
    return round( (len(clcs))**2 / (float(len(s1))*float(len(s2))) ,3)

def mclcsn(s1, s2):
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    result = s1[x_longest - longest: x_longest]
    return round( (len(result))**2 / (float(len(s1))*float(len(s2))) ,3)

def simLCS(x,y):
    # Clean input strings
    x = cleanWebInput(x)
    x = scrub(x)
    x = x.split()
    x = stopWordScrub(x)
    x = lemma(x)
    y = cleanWebInput(y)
    y = scrub(y)
    y = y.split()
    y = stopWordScrub(y)
    y = lemma(y)
    
    # Assign longer string to r
    if len(x)>len(y):
        r = x
        p = y
    else:
        r = y
        p = x
    
    # Find exact matches
    deltaList = list(set(r) & set(p))
    delta = 0
    for word in deltaList:
        delta += min(len(filter(lambda x: x == word, r)), len(filter(lambda x: x == word, p)))
    m = len(p)
    n = len(r)
    
    # Create new lists without exact matches
    newP = list(p)
    newR = list(r)
    for i in deltaList: 
        for j in p:
            if j == i:
                newP.remove(j)
        for k in r:
            if k == i:
                newR.remove(k)
    # If len of the shorter list after removing exact matches == 1 then return exact match, else continue processing
    if len(newP) == 0:
        similarity = 1.0
    else:
        # Create NLCS matrix
        for i in range(len(newP)):
            row = []
            for j in range(len(newR)):
                row.append( wlcs*lcs(newP[i],newR[j]) + wmclcs1*mclcs1(newP[i],newR[j]) + wmclcsn*mclcsn(newP[i],newR[j]) )
            if i==0:
                S = np.array([row])
            else:
                S = np.append(S, [row], axis=0)
        # Cycle through, finding largest value and removing row and column of max
        rho = []
        while len(S)>1 and len(S[0])>1:
            sMax = S.max()
            sMaxPos = np.argwhere(S.max() == S)
            rho.append(sMax)
            # Remove Column
            S = np.delete(S,sMaxPos[0][1],1)
            # Remove Row
            S = np.delete(S,sMaxPos[0][0],0)
            if len(S)==1 or len(S[0])==1:
                sMax = S.max()
                sMaxPos = np.argwhere(S.max() == S)
                rho.append(sMax)
        similarity = ((delta + sum(rho))*(m+n))/(2*m*n)
    return similarity

def simCosine(str1,str2):
    str1 = cleanWebInput(str1)
    str1 = scrub(str1)
    str1 = str1.split()
    str1 = stopWordScrub(str1)
    vec1 = Counter(str1)

    str2 = cleanWebInput(str2)
    str2 = scrub(str2)
    str2 = str2.split()
    str2 = stopWordScrub(str2)
    vec2 = Counter(str2)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def runSim(objDF,smpDF,simDFCols,objVars,smpVars,method):
    """
    Runs similarity regressions and returns a dataframe with the similarity, calculated using either the 
    longest common subsequence methods or the cosine similarity, of each abstract in the objDF dataframe 
    to each abstract in the smpDF dataframe
    Parameters:
    objDF = dataframe of focus abstracts
    smpDF = dataframe of abstracts against which similarity will be calculated
    simDFCols = list of column names to be used in the similarity dataframe returned
    objVars = list of column names on the objDF dataframe to be returned on the similarity dataframe
    smpVars = list of column names on the smpDF dataframe to be returend on the similarity dataframe
    method = method for calculating similarity, lcs or cosine
    """
    simDF = pd.DataFrame(columns=simDFCols)
    
    for obj_ind,obj_row in objDF.iterrows():
        if obj_ind%10==0:
            print obj_ind
        objAbs = obj_row['patent_abstract']
        # Build dictionary of objVars with values from the objDF
        objVarDict = {}
        for i in range(len(objVars)):
            objVarDict[i] = obj_row[objVars[i]]
        
        for smp_ind,smp_row in smpDF.iterrows():
            if method == 'lcs':
                if smp_ind%1000 == 0:
                    print smp_ind
            smpAbs = smp_row['Abstract']
            # Build dictionary of smpVars with values from the smpDF
            smpVarDict = {}
            for i in range(len(smpVars)):
                smpVarDict[i] = smp_row[smpVars[i]]
            
            if method=='lcs':
                sim = simLCS(objAbs,smpAbs)
            elif method =='cosine':
                sim = simCosine(objAbs,smpAbs)
            else:
                print "No method"
                break
            # Build a list of variable values to be appended to the simDF, pulled from both dictionaries
            rowVars = []
            for varKey in range(len(objVars)):
                rowVars.append(objVarDict[varKey])            
            for varKey in range(len(smpVars)):
                rowVars.append(smpVarDict[varKey])
            rowVars.append(sim)
            # Write the new record to the simDF
            row = pd.Series(rowVars,index=simDFCols)
            simDF = simDF.append(row,ignore_index=True)
    
    return simDF

def multi_runSim(objDF,smpDF,simDFCols,objVars,smpVars,method,out_q):
    """
    Runs similarity regressions and returns a dataframe with the similarity, calculated using either the 
    longest common subsequence methods or the cosine similarity, of each abstract in the objDF dataframe 
    to each abstract in the smpDF dataframe
    Parameters:
    objDF = dataframe of focus abstracts
    smpDF = dataframe of abstracts against which similarity will be calculated
    simDFCols = list of column names to be used in the similarity dataframe returned
    objVars = list of column names on the objDF dataframe to be returned on the similarity dataframe
    smpVars = list of column names on the smpDF dataframe to be returend on the similarity dataframe
    method = method for calculating similarity, lcs or cosine
    """
    t0 = time.clock()
    print 'Starting', multiprocessing.current_process().name, time.clock() - t0, '\n'
    # reindex so that percentage printouts are correct
    objDF.index = np.arange(1,len(objDF)+1)
    simDF = pd.DataFrame(columns=simDFCols)
    progSplit = splitter(list(objDF.index),10)
    for obj_ind,obj_row in objDF.iterrows():
        if obj_ind in [i[-1] for i in progSplit]:
            prct = round(obj_ind/float(len(objDF)),3)
            prct = str(prct*100)[:4 + 2]
            print '{0} is {1}% complete'.format(multiprocessing.current_process().name, prct)
        objAbs = obj_row['patent_abstract']
        # Build dictionary of objVars with values from the objDF
        objVarDict = {}
        for i in range(len(objVars)):
            objVarDict[i] = obj_row[objVars[i]]
        
        for smp_ind,smp_row in smpDF.iterrows():
            smpAbs = smp_row['Abstract']
            # Build dictionary of smpVars with values from the smpDF
            smpVarDict = {}
            for i in range(len(smpVars)):
                smpVarDict[i] = smp_row[smpVars[i]]
            
            if method=='lcs':
                sim = simLCS(objAbs,smpAbs)
            elif method =='cosine':
                sim = simCosine(objAbs,smpAbs)
            else:
                print "No method"
                break
            # Build a list of variable values to be appended to the simDF, pulled from both dictionaries
            rowVars = []
            for varKey in range(len(objVars)):
                rowVars.append(objVarDict[varKey])            
            for varKey in range(len(smpVars)):
                rowVars.append(smpVarDict[varKey])
            rowVars.append(sim)
            # Write the new record to the simDF
            row = pd.Series(rowVars,index=simDFCols)
            simDF = simDF.append(row,ignore_index=True)
    
    print '\n', 'Exiting', multiprocessing.current_process().name, 'minutes:{0}'.format(round(float(time.clock()-t0)/60.0,2)), '\n'
    out_q.put(simDF)
        
def main():
    # User input of processes to run
    params = {'doBioSim':'null', 'doAISim':'null'}
    for p in params.keys():
        print 'would you like to {0} (y/n)? '.format(p),
        answer = 'null'
        while answer not in ['y','n']:
            answer = raw_input()
            if answer in ['y','n']:
                params[p] = answer
            else:
                print 'invalid input'
    
    if params['doBioSim']=='y':
        ## Process Biotech
        # load biotech data
        bioPats = pd.DataFrame.from_csv(difPath+'/bio_patents.csv')
        bioAbs = pd.DataFrame.from_csv(difPath+'/bio_abstracts.csv')
        # reindex loaded data
        bioPats.index = np.arange(1,len(bioPats)+1)
        bioAbs.index = np.arange(1,len(bioAbs)+1)
        
        simDFCols = ['PatNum','PatYear','PapTitle','PapAuthors','PapJRef','PapYear','Similarity']
        objVars = ['patent','gyear']
        smpVars = ['Title','Authors','JRef','Year']
        
        ## calculate biotech similarity
        print '\n', 'Start Processing Biotech' 
        # split patents and kick off multiprocesses
        lenBioPats = range(0,len(bioPats)+1)
        splitSlices = splitterSlices(lenBioPats,5)
        out_q = multiprocessing.Queue()
        
        # start processes to calculate similarity
        processes = []
        for i in range(5):
            w = multiprocessing.Process(name='wkr_bio_{0}'.format(i), target=multi_runSim, args=(bioPats[splitSlices[i][0]:splitSlices[i][1]],bioAbs,simDFCols,objVars,smpVars,'lcs',out_q))
            processes.append(w)
            w.start()
            time.sleep(2)
        
        # gather and join output of processes
        finalSimDF = pd.DataFrame(columns=simDFCols)
        for i in range(len(processes)):
            finalSimDF = pd.concat([finalSimDF,out_q.get()])
        
        # reindex and store joined output 
        finalSimDF.index = np.arange(1,len(finalSimDF)+1)
        finalSimDF.to_csv(difPath+'/bio_similarity.csv')
        
        # join processes
        for p in processes:
            p.join()
        print 'Workers:',processes,'finished and joined'
        # clear output queue
        out_q = ''

    if params['doAISim']=='y':
        ## Process AI
        # load ai data
        aiPats = pd.DataFrame.from_csv(difPath+'/ai_patents.csv')
        aiAbs = pd.DataFrame.from_csv(difPath+'/ai_abstracts.csv')
        # reindex loaded data
        aiPats.index = np.arange(1,len(aiPats)+1)
        aiAbs.index = np.arange(1,len(aiAbs)+1)
        # create year variable for abstract data
        aiAbs['SubmitDate'] = pd.to_datetime(aiAbs['SubmitDate'])
        aiAbs['Year'] = pd.DatetimeIndex(aiAbs['SubmitDate']).year
        
        simDFCols = ['PatNum','PatYear','PapTitle','PapAuthors','PapJRef','PapYear','Similarity']
        objVars = ['patent','gyear']
        smpVars = ['Title','Authors','JRef','Year']
        
        ## calculate ai similarity
        print '\n', 'Start Processing AI' 
        # split patents and kick off multiprocesses
        lenAIPats = range(0,len(aiPats)+1)
        splitSlices = splitterSlices(lenAIPats,5)
        out_q = multiprocessing.Queue()
        
        # start processes to calculate similarity
        processes = []
        for i in range(5):
            w = multiprocessing.Process(name='wkr_ai_{0}'.format(i), target=multi_runSim, args=(aiPats[splitSlices[i][0]:splitSlices[i][1]],aiAbs,simDFCols,objVars,smpVars,'lcs',out_q))
            processes.append(w)
            w.start()
            time.sleep(2)
        
        # gather and join output of processes
        finalSimDF = pd.DataFrame(columns=simDFCols)
        for i in range(len(processes)):
            finalSimDF = pd.concat([finalSimDF,out_q.get()])
        
        # reindex and store joined output
        finalSimDF.index = np.arange(1,len(finalSimDF)+1)    
        finalSimDF.to_csv(difPath+'/ai_similarity.csv')
        
        # join processes
        for p in processes:
            p.join()
        print 'Workers:',processes,'finished and joined'
    
if __name__=="__main__":
    main()
