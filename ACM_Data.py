# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import urllib
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
from scipy import stats
import statsmodels.api as sm
import os
import sys
import ast

# <codecell>

stopWords = []
with open('english_stopwords.txt', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        for i in row:
            stopWords.append(i)
print len(stopWords)

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
    #for punc in puncToStrip:
    #    text = text.replace(punc,"")
    text = text.replace("  ", " ")
    return text

def preSuf(splitText):
    for word in splitText:
        if len(word) > 3:
            if word[-2:] == 'ed':
                if word[0:-2] in splitText:
                    for n,i in enumerate(splitText):
                        if i == word:
                            splitText[n] = word[0:-2]
            elif word[-1:] == 's':
                if word[0:-1] in splitText:
                    for n,i in enumerate(splitText):
                        if i == word:
                            splitText[n] = word[0:-1]
            elif word[-3:] == 'ing':
                if word[0:-3] in splitText:
                    for n,i in enumerate(splitText):
                        if i == word:
                            splitText[n] = word[0:-3]
    return splitText

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


wlcs = 0.34
wmclcs1 = 0.33
wmclcsn = 0.33

def lcs(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    #print lengths
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = \
                    max(lengths[i+1][j], lengths[i][j+1])
    #print lengths
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

def similarity(x,y):
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

# <codecell>

# Import JACM xml files methods
def checkAbs(article):
    if article.abstract:
        return article.abstract.get_text()
    else:
        return "None"

def checkFt(article):
    if article.ft_body:
        return article.ft_body.get_text()
    else:
        return "None"
def checkPageTo(article):
    if article.page_to:
        return article.page_to.get_text()
    else:
        return "None"
def procPrimaryCat(article):
    prCat = []
    if article.categories.primary_category:
        if article.categories.primary_category.cat_node:
            prCat.append(article.categories.primary_category.cat_node.get_text())
        else:
            prCat.append("None")
        if article.categories.primary_category.descriptor:
            prCat.append(article.categories.primary_category.descriptor.get_text())
        else:
            prCat.append("None")
        if article.categories.primary_category.type:
            prCat.append(article.categories.primary_category.type.get_text())
        else:
            prCat.append("None")
        return prCat
    else:
        return "None"
def procOtherCat(article):
    oCat = []
    if article.categories.findAll('other_category'):
        for oc in article.categories.findAll('other_category'):
            indCat = []
            if oc.cat_node:
                indCat.append(oc.cat_node.get_text())
            else:
                indCat.append("None")
            if oc.descriptor:
                indCat.append(oc.descriptor.get_text())
            else:
                indCat.append("None")
            if oc.type:
                indCat.append(oc.type.get_text())
            else:
                indCat.append("None")
            oCat.append(indCat)
        return oCat
    else:
        return "None"
def procGeneralTerms(article):
    gt = []
    if article.general_terms:
        for i in article.general_terms.findAll('gt'):
            gt.append(i.get_text())
        return gt
    else:
        return "None"
def procKeywords(article):
    kw = []
    if article.keywords:
        for i in article.keywords.findAll('kw'):
            kw.append(i.get_text())
        return kw
    else:
        return "None"
def procAuthors(article):
    authors = []
    if article.authors:
        for i in article.findAll('au'):
            indAuth = []
            if i.person_id:
                indAuth.append(i.person_id.get_text())
            else:
                indAuth.append("None")
            indAuth.append((str(i.first_name.get_text()),
                            str(i.middle_name.get_text()),
                            str(i.last_name.get_text())))
            if i.affiliation:
                indAuth.append(i.affiliation.get_text())
            else:
                indAuth.append("None")
            if i.role:
                indAuth.append(i.role.get_text())
            else:
                indAuth.append("None")
            
            if indAuth:
                authors.append(indAuth)
            else:
                authors.append("None")
        return authors
    else:
        return "None"
def procRefs(article):
    refs = []
    if article.references:
        rf = article.references.findAll('ref')
        for i in rf:
            indRef = []
            if i.ref_id.get_text() != "":
                indRef.append(i.ref_id.get_text())
            else:
                indRef.append("None")
            if i.ref_seq_no.get_text() != "":
                indRef.append(i.ref_seq_no.get_text())
            else:
                indRef.append("None")
            if i.ref_text:
                indRef.append(i.ref_text.get_text())
            else:
                indRef.append("None")
        return refs
    else:
        return "None"

# <headingcell level=1>

# Import ACM Data

# <codecell>

# Import JACM data from xml files
fileNames = sorted((fn for fn in os.listdir('./acmdl/periodical') if fn.startswith('JOUR-JACM')))

artHeaders = ('journal_id',
                'journal_code',
                'journal_name',
                'issue_id',
                'volume',
                'issue',
                'issue_date',
                'publication_date',
                'article_id',
                'sort_key',
                'article_publication_date',
                'seq_no',
                'title',
                'authors',
                'page_from',
                'page_to',
                'doi_number',
                'url',
                'primary_category',
                'other_categories',
                'general_terms',
                'keywords',
                'references',
                'abstract')

jacmDF = pd.DataFrame(columns=artHeaders)

for f in fileNames:
    print f
    filePath = "./acmdl/periodical/"+f
    f = open(filePath,"r")
    content = f.read()
    soup = BeautifulSoup(content)
    
    artRecs = [(soup.journal_rec.journal_id.get_text(),
                soup.journal_rec.journal_code.get_text(),
                soup.journal_rec.journal_name.get_text(),
                soup.issue_rec.issue_id.get_text(),
                soup.issue_rec.volume.get_text(),
                soup.issue_rec.issue.get_text(),
                soup.issue_rec.issue_date.get_text(),
                soup.issue_rec.publication_date.get_text(),
                art.article_id.get_text(),
                art.sort_key.get_text(),
                art.article_publication_date.get_text(),
                art.seq_no.get_text(),
                art.title.get_text(),
                procAuthors(art),
                art.page_from.get_text(),
                checkPageTo(art),
                art.doi_number.get_text(),
                art.url.get_text(),
                procPrimaryCat(art),
                procOtherCat(art),
                procGeneralTerms(art),
                procKeywords(art),
                procRefs(art),
                checkAbs(art)) 
                 for art in soup.findAll('article_rec')]
    
    articles = pd.DataFrame.from_records(artRecs, columns=artHeaders)
    
    jacmDF = jacmDF.append(articles, ignore_index=True)
    f.close()

jacmDF.to_csv('jacm_abstracts.csv')

# <codecell>

# Import CACM data from xml files
fileNames = sorted((fn for fn in os.listdir('./acmdl/periodical') if fn.startswith('MAG-CACM')))

artHeaders = ('journal_id',
                'journal_code',
                'journal_name',
                'issue_id',
                'volume',
                'issue',
                'issue_date',
                'publication_date',
                'article_id',
                'sort_key',
                'article_publication_date',
                'seq_no',
                'title',
                'authors',
                'page_from',
                'page_to',
                'doi_number',
                'url',
                'primary_category',
                'other_categories',
                'general_terms',
                'keywords',
                'references',
                'abstract')

cacmDF = pd.DataFrame(columns=artHeaders)

for f in fileNames:
    print f
    filePath = "./acmdl/periodical/"+f
    f = open(filePath,"r")
    content = f.read()
    soup = BeautifulSoup(content)
    
    artRecs = [(soup.journal_rec.journal_id.get_text(),
                soup.journal_rec.journal_code.get_text(),
                soup.journal_rec.journal_name.get_text(),
                soup.issue_rec.issue_id.get_text(),
                soup.issue_rec.volume.get_text(),
                soup.issue_rec.issue.get_text(),
                soup.issue_rec.issue_date.get_text(),
                soup.issue_rec.publication_date.get_text(),
                art.article_id.get_text(),
                art.sort_key.get_text(),
                art.article_publication_date.get_text(),
                art.seq_no.get_text(),
                art.title.get_text(),
                procAuthors(art),
                art.page_from.get_text(),
                checkPageTo(art),
                art.doi_number.get_text(),
                art.url.get_text(),
                procPrimaryCat(art),
                procOtherCat(art),
                procGeneralTerms(art),
                procKeywords(art),
                procRefs(art),
                checkAbs(art)) 
                 for art in soup.findAll('article_rec')]
    
    articles = pd.DataFrame.from_records(artRecs, columns=artHeaders)
    
    cacmDF = cacmDF.append(articles, ignore_index=True)
    f.close()

cacmDF['abstract'] = cacmDF['abstract'].apply(lambda x: normalize('NFKD',unicode(x)).encode('ASCII', 'ignore'))
cacmDF.to_csv('cacm_abstracts.csv')

# <headingcell level=1>

# Calculate Similarity

# <codecell>

jacmDF = pd.DataFrame.from_csv('jacm_abstracts.csv')
jacmDF = jacmDF[jacmDF.abstract != 'None']
jacmDF = jacmDF.drop_duplicates('title')
index = pd.Series(range(0,len(jacmDF)))
jacmDF.index=index
jacmSim = pd.DataFrame(columns=['rsa_title','rsa_authors','rsa_date','rsa_primary_category','rsa_other_categories','rsa_general_terms','rsa_keywords','pap_title','pap_authors','pap_issue','pap_volume','pap_date','pap_doi','pap_general_terms','pap_keywords','pap_primary_category','pap_other_categories','sim'])

cacmDF = pd.DataFrame.from_csv('cacm_abstracts.csv')
cacmDF = cacmDF[cacmDF.abstract != 'None']
cacmDF = cacmDF.drop_duplicates('title')
index = pd.Series(range(0,len(cacmDF)))
cacmDF.index=index
cacmSim = pd.DataFrame(columns=['rsa_title','rsa_authors','rsa_date','rsa_primary_category','rsa_other_categories','rsa_general_terms','rsa_keywords','pap_journal_code','pap_title','pap_authors','pap_issue','pap_volume','pap_date','pap_doi','pap_general_terms','pap_keywords','pap_primary_category','pap_other_categories','sim'])

# <codecell>

# Analysis and similarity matrix for RSA
t0 = time.clock()
rsaAbs = 'An encryption method is presented with the novel property that publicly revealing an encryption key does not thereby reveal the corresponding decryption key. This has two important consequences: (1) Couriers or other secure means are not needed to transmit keys, since a message can be enciphered using an encryption key publicly revealed by the intented recipient. Only he can decipher the message, since only he knows the corresponding decryption key. (2) A message can be signed using a privately held decryption key. Anyone can verify this signature using the corresponding publicly revealed encryption key. Signatures cannot be forged, and a signer cannot later deny the validity of his signature. This has obvious applications in electronic mail and electronic funds transfer systems. A message is encrypted by representing it as a number M, raising M to a publicly specified power e, and then taking the remainder when the result is divided by the publicly specified product, n, of two large secret primer numbers p and q. Decryption is similar; only a different, secret, power d is used, where e * d ≡ 1(mod (p - 1) * (q - 1)). The security of the system rests in part on the difficulty of factoring the published divisor, n.'
rsaTitle = 'A method for obtaining digital signatures and public-key cryptosystems'
rsaAuthors = [[u'PP43124782', ('R.', 'L.', 'Rivest'), u'MIT Lab. for Computer Science and Department of Mathematics, Cambridge, MA', u'Author'], [u'PP43133515', ('A.', ' ', 'Shamir'), u'MIT Lab. for Computer Science and Department of Mathematics, Cambridge, MA', u'Author'], [u'PP39028850', ('L.', ' ', 'Adleman'), u'MIT Lab. for Computer Science and Department of Mathematics, Cambridge, MA', u'Author']]
rsaDate = '02-01-1978'
rsaPC = [u'E.3', '', '']
rsaOC = [[u'K.4.1', u'Privacy', u'S'], [u'K.6.5', '', '']]
rsaGT = [u'Design', u'Human Factors', u'Performance', u'Security', u'Theory']
rsaKW = [u'authentication', u'cryptography', u'digital signatures', u'electronic funds transfer', u'electronic mail', u'factorization', u'message-passing', u'prime number', u'privacy', u'public-key cryptosystems', u'security']

for row_index,row in jacmDF.iterrows():
    if row_index%100 == 0:
        print row_index
    papAbs = row['abstract']
    papJC = row['journal_code']
    papTitle = row['title']
    papJC = row['journal_code']
    papAuthors = row['authors']
    papIssue = row['issue']
    papVolume = row['volume']
    papDate = row['publication_date']
    papDOI = row['doi_number']
    papGT = row['general_terms']
    papKW = row['keywords']
    papPCat = row['primary_category']
    papOCat = row['other_categories']
    
    sim = similarity(rsaAbs,papAbs)
    
    row = pd.Series([rsaTitle,rsaAuthors,rsaDate,rsaPC,rsaOC,rsaGT,rsaKW,papJC,papTitle,papAuthors,papIssue,papVolume,papDate,papDOI,papGT,papKW,papPCat,papOCat,sim],index=['rsa_title','rsa_authors','rsa_date','rsa_primary_category','rsa_other_categories','rsa_general_terms','rsa_keywords','pap_journal_code','pap_title','pap_authors','pap_issue','pap_volume','pap_date','pap_doi','pap_general_terms','pap_keywords','pap_primary_category','pap_other_categories','sim'])
    jacmSim = jacmSim.append(row,ignore_index=True)
        
t = time.clock() - t0
print "Minutes Lapsed:", (t/60)
jacmSim.to_csv('jacm_similarity.csv')

for row_index,row in cacmDF.iterrows():
    if row_index%100 == 0:
        print row_index
    papAbs = row['abstract']
    papJC = row['journal_code']
    papTitle = row['title']
    papAuthors = row['authors']
    papIssue = row['issue']
    papVolume = row['volume']
    papDate = row['publication_date']
    papDOI = row['doi_number']
    papGT = row['general_terms']
    papKW = row['keywords']
    papPCat = row['primary_category']
    papOCat = row['other_categories']
    
    sim = similarity(rsaAbs,papAbs)
    
    row = pd.Series([rsaTitle,rsaAuthors,rsaDate,rsaPC,rsaOC,rsaGT,rsaKW,papJC,papTitle,papAuthors,papIssue,papVolume,papDate,papDOI,papGT,papKW,papPCat,papOCat,sim],index=['rsa_title','rsa_authors','rsa_date','rsa_primary_category','rsa_other_categories','rsa_general_terms','rsa_keywords','pap_journal_code','pap_title','pap_authors','pap_issue','pap_volume','pap_date','pap_doi','pap_general_terms','pap_keywords','pap_primary_category','pap_other_categories','sim'])
    cacmSim = cacmSim.append(row,ignore_index=True)
        
t = time.clock() - t0
print "Minutes Lapsed:", (t/60)
cacmSim.to_csv('cacm_similarity.csv')

# <headingcell level=1>

# Analyze Similarity

# <codecell>

"""
to do: what happens when filter on jacm vs cacm
"""

cacmSim = pd.DataFrame.from_csv('cacm_similarity.csv')
jacmSim = pd.DataFrame.from_csv('jacm_similarity.csv')

papSim = pd.concat([cacmSim, jacmSim])
papSim = papSim.drop_duplicates('pap_title')

papSim['rsa_date'] = pd.to_datetime(papSim['rsa_date'])
papSim['rsa_year'] = papSim['rsa_date'].apply(lambda x: x.year)
papSim['pap_date'] = pd.to_datetime(papSim['pap_date'])
papSim['pap_year'] = papSim['pap_date'].apply(lambda x: x.year)

papSim = papSim.sort(['pap_year'])

# Re-index to ensure that modifications based on index are applied to the correct rows
index = pd.Series(range(0,len(papSim)))
papSim.index=index

papSim['pap_rel_year'] = papSim['pap_year'].apply(lambda x: x - 1978)
papSim['sim'] = papSim['sim'].apply(lambda x: float(x))

# Remove years
#papSim = papSim[(papSim.pap_year>1970)&(papSim.pap_year<2000)]

# <codecell>

def convertBackToList(string):
    if string != "None":
        return ast.literal_eval(string)
    else:
        return ["None"]
papSim['rsa_authors'] = papSim.rsa_authors.apply(lambda x: convertBackToList(x))
papSim['rsa_primary_category'] = papSim.rsa_primary_category.apply(lambda x: convertBackToList(x))
papSim['rsa_other_categories'] = papSim.rsa_other_categories.apply(lambda x: convertBackToList(x))
papSim['rsa_general_terms'] = papSim.rsa_general_terms.apply(lambda x: convertBackToList(x))
papSim['rsa_keywords'] = papSim.rsa_keywords.apply(lambda x: convertBackToList(x))
papSim['pap_authors'] = papSim.pap_authors.apply(lambda x: convertBackToList(x))
papSim['pap_general_terms'] = papSim.pap_general_terms.apply(lambda x: convertBackToList(x))
papSim['pap_keywords'] = papSim.pap_keywords.apply(lambda x: convertBackToList(x))
papSim['pap_primary_category'] = papSim.pap_primary_category.apply(lambda x: convertBackToList(x))
papSim['pap_other_categories'] = papSim.pap_other_categories.apply(lambda x: convertBackToList(x))

# <codecell>

# Filter on categories, matching the paper's primary and other categories to the RSA's primary and other categories
crySim = pd.DataFrame(columns=['rsa_title','rsa_authors','rsa_date','rsa_primary_category','rsa_other_categories','rsa_general_terms','rsa_keywords','pap_title','pap_authors','pap_issue','pap_volume','pap_date','pap_doi','pap_general_terms','pap_keywords','pap_primary_category','pap_other_categories','sim','rsa_year','pap_year','pap_rel_year'])

for row_index, row in papSim.iterrows():
    isIn = False
    if row['pap_primary_category'] != 'None':
        if (row['pap_primary_category']=='E.3' or row['pap_primary_category']=='K.4.1' or row['pap_primary_category']=='K.6.5') and isIn==False:
                crySim = crySim.append(row,ignore_index=True)
                isIn=True
    if row['pap_other_categories'] != 'None':
        for oc in row['pap_other_categories']:
            if (oc[0]=='E.3' or oc[0]=='K.4.1' or oc[0]=='K.6.5') and isIn==False:
                crySim = crySim.append(row,ignore_index=True)
                isIn=True
papSim = crySim

# <codecell>

# Create dummy variables on paper similarity dataframe 
papSim['D1968'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1968:
        papSim.D1968[row_index] = 1
    else:
        papSim.D1968[row_index] = 0

papSim['D1969'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1969:
        papSim.D1969[row_index] = 1
    else:
        papSim.D1969[row_index] = 0

papSim['D1970'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1970:
        papSim.D1970[row_index] = 1
    else:
        papSim.D1970[row_index] = 0
        
papSim['D1971'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1971:
        papSim.D1971[row_index] = 1
    else:
        papSim.D1971[row_index] = 0

papSim['D1972'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1972:
        papSim.D1972[row_index] = 1
    else:
        papSim.D1972[row_index] = 0
        
papSim['D1973'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1973:
        papSim.D1973[row_index] = 1
    else:
        papSim.D1973[row_index] = 0

papSim['D1974'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1974:
        papSim.D1974[row_index] = 1
    else:
        papSim.D1974[row_index] = 0

papSim['D1975'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1975:
        papSim.D1975[row_index] = 1
    else:
        papSim.D1975[row_index] = 0

papSim['D1976'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1976:
        papSim.D1976[row_index] = 1
    else:
        papSim.D1976[row_index] = 0

papSim['D1977'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1977:
        papSim.D1977[row_index] = 1
    else:
        papSim.D1977[row_index] = 0

papSim['D1978'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1978:
        papSim.D1978[row_index] = 1
    else:
        papSim.D1978[row_index] = 0

papSim['D1979'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1979:
        papSim.D1979[row_index] = 1
    else:
        papSim.D1979[row_index] = 0

papSim['D1980'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1980:
        papSim.D1980[row_index] = 1
    else:
        papSim.D1980[row_index] = 0

        
papSim['D1981'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1981:
        papSim.D1981[row_index] = 1
    else:
        papSim.D1981[row_index] = 0

papSim['D1982'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1982:
        papSim.D1982[row_index] = 1
    else:
        papSim.D1982[row_index] = 0

papSim['D1983'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1983:
        papSim.D1983[row_index] = 1
    else:
        papSim.D1983[row_index] = 0

papSim['D1984'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1984:
        papSim.D1984[row_index] = 1
    else:
        papSim.D1984[row_index] = 0

papSim['D1985'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1985:
        papSim.D1985[row_index] = 1
    else:
        papSim.D1985[row_index] = 0

papSim['D1986'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1986:
        papSim.D1986[row_index] = 1
    else:
        papSim.D1986[row_index] = 0

papSim['D1987'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1987:
        papSim.D1987[row_index] = 1
    else:
        papSim.D1987[row_index] = 0

papSim['D1988'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1988:
        papSim.D1988[row_index] = 1
    else:
        papSim.D1988[row_index] = 0

papSim['D1989'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1989:
        papSim.D1989[row_index] = 1
    else:
        papSim.D1989[row_index] = 0
        
papSim['D1990'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1990:
        papSim.D1990[row_index] = 1
    else:
        papSim.D1990[row_index] = 0
        
papSim['D1991'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1991:
        papSim.D1991[row_index] = 1
    else:
        papSim.D1991[row_index] = 0
        
papSim['D1992'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1992:
        papSim.D1992[row_index] = 1
    else:
        papSim.D1992[row_index] = 0

papSim['D1993'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1993:
        papSim.D1993[row_index] = 1
    else:
        papSim.D1993[row_index] = 0

papSim['D1994'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1994:
        papSim.D1994[row_index] = 1
    else:
        papSim.D1994[row_index] = 0
        
papSim['D1995'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1995:
        papSim.D1995[row_index] = 1
    else:
        papSim.D1995[row_index] = 0

papSim['D1996'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1996:
        papSim.D1996[row_index] = 1
    else:
        papSim.D1996[row_index] = 0

papSim['D1997'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1997:
        papSim.D1997[row_index] = 1
    else:
        papSim.D1997[row_index] = 0

papSim['D1998'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1998:
        papSim.D1998[row_index] = 1
    else:
        papSim.D1998[row_index] = 0

papSim['D1999'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 1999:
        papSim.D1999[row_index] = 1
    else:
        papSim.D1999[row_index] = 0

papSim['D2000'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 2000:
        papSim.D2000[row_index] = 1
    else:
        papSim.D2000[row_index] = 0

        
papSim['D2001'] = int()
for row_index, row in papSim.iterrows():
    if row['pap_year'] >= 2001:
        papSim.D2001[row_index] = 1
    else:
        papSim.D2001[row_index] = 0

# <codecell>

papSim['pap_year'] = papSim['pap_year'].apply(lambda x: float(x)) 
papSim['sim'] = papSim['sim'].apply(lambda x: float(x)) 

# <codecell>

tmpGBYear = list(papSim.groupby(['pap_year']).sim.values)
tmpYearMean = []
tmpYearCount = []
tmpPercentile = []
for i in tmpGBYear:
    tmpYearMean.append(np.mean(i))
    tmpYearCount.append(len(i))
    tmpPercentile.append(np.percentile(i,95))
years = list(set(papSim.pap_year))

fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(131)
ax1.scatter(years, tmpYearMean)
ax1.set_xlabel('Year')
ax1.set_ylabel('Mean Similarity')
ax1.set_title('Mean Abstract Similairty to RSA Paper by Year')


ax2 = fig.add_subplot(132)
ax2.scatter(years, tmpYearCount)
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Abstracts')
ax2.set_title('Count of ACM Abstracts by Year')


ax3 = fig.add_subplot(133)
ax3.scatter(years, tmpPercentile)
ax3.set_title('Similarity 95th Percentile')
plt.show()

# <codecell>

#papWork = papSim.copy()
print papWork.ix[0]
#papSim = papWork[(papWork.pap_year>1975)&(papWork.pap_year<1994)&(papWork.pap_journal_code=="JACM")]

# <codecell>

simRegs = pd.DataFrame(columns=['Title', 'Year', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
#simRegs[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']] = simRegs[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']].apply(np.float32)
def doMeanSimReg(x1,papST,dumYear,title,df):
    y = list(papST.groupby(['pap_year']).sim.mean())
    X = sm.add_constant(zip(x1), prepend=True)
    results = sm.OLS(y, X).fit()
    
    year = papST.rsa_year.mean()
    relDumYear = dumYear - year
    CCoef = round(results.params[0],5)
    CPVal = round(results.pvalues[0],5)
    DCoef = round(results.params[1],5)
    DPVal = round(results.pvalues[1],5)
    r2 = round(results.rsquared,5)
    probF = round(results.f_pvalue,5)
    dfRes = round(results.df_resid,5)

    row = pd.Series([title,year,dumYear,relDumYear,CCoef,CPVal,DCoef,DPVal,r2,probF,dfRes],index=['Title', 'Year', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
    df = df.append(row, ignore_index=True)
    # Changing dtypes back from object to floats
    df[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']] = df[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']].apply(np.float32)
    return df

paper = 'A method for obtaining digital signatures and public-key cryptosystems'
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1969.mean()),papSim,1969,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1970.mean()),papSim,1970,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1971.mean()),papSim,1971,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1972.mean()),papSim,1972,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1973.mean()),papSim,1973,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1974.mean()),papSim,1974,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1975.mean()),papSim,1975,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1976.mean()),papSim,1976,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1977.mean()),papSim,1977,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1978.mean()),papSim,1978,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1979.mean()),papSim,1979,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1980.mean()),papSim,1980,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1981.mean()),papSim,1981,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1982.mean()),papSim,1982,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1983.mean()),papSim,1983,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1984.mean()),papSim,1984,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1985.mean()),papSim,1985,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1986.mean()),papSim,1986,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1987.mean()),papSim,1987,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1988.mean()),papSim,1988,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1989.mean()),papSim,1989,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1990.mean()),papSim,1990,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1991.mean()),papSim,1991,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1992.mean()),papSim,1992,paper,simRegs)
simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1993.mean()),papSim,1993,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1994.mean()),papSim,1994,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1995.mean()),papSim,1995,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1996.mean()),papSim,1996,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1997.mean()),papSim,1997,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1998.mean()),papSim,1998,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D1999.mean()),papSim,1999,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D2000.mean()),papSim,2000,paper,simRegs)
#simRegs = doMeanSimReg(list(papSim.groupby(['pap_year']).D2001.mean()),papSim,2001,paper,simRegs)



simRegs.to_csv('acm_simRegRaw.csv')


#print simRegs
minRelYear = int(simRegs.RelDumYear.min())
print "minRelYear", minRelYear
maxRelYear = int(simRegs.RelDumYear.max())
print "maxRelYear", maxRelYear
simRelMeans = pd.DataFrame(pd.Series(range(minRelYear,maxRelYear+1)), columns=['RelDumYear'])


simRelMeans['CCoef'] = pd.Series(simRegs.groupby(['RelDumYear']).CCoef.mean().values)
simRelMeans['CPVal'] = pd.Series(simRegs.groupby(['RelDumYear']).CPVal.mean().values)
simRelMeans['DCoef'] = pd.Series(simRegs.groupby(['RelDumYear']).DCoef.mean().values)
simRelMeans['DPVal'] = pd.Series(simRegs.groupby(['RelDumYear']).DPVal.mean().values)
simRelMeans['R2'] = pd.Series(simRegs.groupby(['R2']).R2.mean().values)
simRelMeans['Pfstat'] = pd.Series(simRegs.groupby(['RelDumYear']).Pfstat.mean().values)
simRelMeans['DFResid'] = pd.Series(simRegs.groupby(['RelDumYear']).DFResid.mean().values)

print simRelMeans
simRelMeans.to_csv('paper_simRelMeans.csv')

#print "pval\n", simRegs.DPVal.describe()
#print "coef\n", simRegs.DCoef.describe()
print "min RelDumYear", simRegs.RelDumYear.min()
print "max RelDumYear", simRegs.RelDumYear.max()
print len(simRegs.groupby(['RelDumYear']).CCoef.mean().values)

relDumYears = range(int(min(simRegs.RelDumYear)),int(max(simRegs.RelDumYear))+1)

meanPval = []
meanCoef = []

for relYr in relDumYears:
    meanPval.append(simRegs[simRegs.RelDumYear==relYr].DPVal.mean())
    meanCoef.append(simRegs[simRegs.RelDumYear==relYr].DCoef.mean())

#print "relDumYears",relDumYears
#print "meanPval",meanPval
#print "meanCoef", meanCoef

# <codecell>

# Plot average across papers
fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
for i in range(len(meanPval)):
    if meanPval[i]<=0.05:
        ax1.scatter(relDumYears[i],meanPval[i], marker='o', c='b',s=40)
    else:
        ax1.scatter(relDumYears[i],meanPval[i], marker='x', c='r',s=50)
#ax1.scatter(relDumYears,meanPval)
ax1.set_ylabel('Mean P-Value')
ax1.set_xlabel('Dummy Year Relative to Publication')
ax1.set_title('Top AI Papers Dummy Regression P-Values\nMean OLS P-Values by Relative Dummy Year')
ax1.set_ylim(0,1)
ax1.axvline(x=0, color='r', ls='--', lw=2)
ax1.text(1,0.7,'Year Published',fontsize=12,)
ax1.grid()

ax2 = fig.add_subplot(122)
for i in range(len(meanCoef)):
    if meanPval[i]<=0.05:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='o', c='b',s=40)
    else:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='x', c='r',s=50)
#ax2.scatter(relDumYears,meanCoef)
ax2.set_ylabel('Mean Coefficient')
ax2.set_xlabel('Dummy Year Relative to Publication')
ax2.set_title('Top AI Papers Dummy Regression Coefficients\nMean OLS Coefficients by Relative Dummy Year')
ax2.axvline(x=0, color='r', ls='--', lw=2)
#ax2.text(1,-0.003,'Year Published',fontsize=12,)
ax2.grid()
#plt.savefig('pat_diffusion.png')
plt.show()

# <codecell>


fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
ax1.hist(papSim[(papSim.pap_rel_year<=0)].sim,bins=50)
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Similarity')
ax1.set_title('Histogram Top AI Papers Similarity\nSimilarity Measures Before Patent Publication')
meanBefore = papSim[(papSim.pap_rel_year<=0)].sim.mean()
ax1.axvline(x=meanBefore, color='r', ls='-', lw=2)
ax1.set_xlim(0,papSim.sim.max()*1.1)
ax1.grid()

ax2 = fig.add_subplot(122)
ax2.hist(papSim[(papSim.pap_rel_year>0)].sim,bins=50)
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Similarity')
ax2.set_title('Histogram Top AI Papers Similarity\nSimilarity Measures After Patent Publication')
meanAfter = papSim[(papSim.pap_rel_year>0)].sim.mean()
ax2.axvline(x=meanAfter, color='r', ls='-', lw=2)
ax2.set_xlim(0,papSim.sim.max()*1.1)
ax2.grid()
plt.show()

# <codecell>

papSim[papSim.pap_title=='A method for obtaining digital signatures and public-key cryptosystems']

# <codecell>

print papSim[papSim.pap_year>1960].groupby(['pap_year']).sim.mean().plot()
#print papSim[papSim.pap_year<1960]
#papSim.to_csv('test.csv')

