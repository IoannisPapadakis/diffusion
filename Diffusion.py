# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#import urllib
import re
#from bs4 import BeautifulSoup
import csv
import string
import operator
import nltk
import numpy as np
import time
import pandas as pd
from unicodedata import normalize
import matplotlib.pyplot as plt

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
    text = text.replace("<summary>","")
    text = text.replace("</summary>","")
    text = text.replace("<p>","")
    text = text.replace("</p>","")
    text = text.replace("\n"," ")
    text = text.replace("     ","")
    text = text.replace("<p>","")
    text = text.replace("</p>","")
    text = text.replace("<div class=\"pubabstract\">","")
    text = text.replace("</div>","")
    text = text.replace("<author> <name>","")
    text = text.replace("</name> </author>","")
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

# <codecell>

def similarity(x,y):
    # Clean input strings
    x = scrub(x)
    x = x.split()
    x = stopWordScrub(x)
    x = lemma(x)
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

def scrapePTOSearch():
    # Scrape patent numbers from patent search
    # First call to get number of results
    # URL for searching for "robot" in abstract
    url = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&u=%2Fnetahtml%2FPTO%2Fsearch-adv.htm&r=0&p=1&f=S&l=50&Query=ccl%2F706%2F%24+AND+APD%2F1%2F1%2F2003-%3E12%2F31%2F2003&d=PTXT'
    website = urllib.urlopen(url).read()
    soup = BeautifulSoup(website)
    
    totalResults = soup.findAll('strong')[2].string
    pages = float(totalResults)/50
    if int(pages)< pages:
        pages = int(pages) + 1
    else:
        pages = int(pages)
    
    serRes = []
    time.sleep(3)
    
    for page in range(1,pages+1):
        if page == 1:
            url = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&u=%2Fnetahtml%2FPTO%2Fsearch-adv.htm&r=0&p=1&f=S&l=50&Query=ccl%2F706%2F%24+AND+APD%2F1%2F1%2F2003-%3E12%2F31%2F2003&d=PTXT'
            website = urllib.urlopen(url).read()
            soup = BeautifulSoup(website)
            for i in soup.findAll('a'):
                if i.string is not None:
                    if len(i.string)==9:
                        serRes.append(str(i.string).replace(",",""))
        else:
            url = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&u=%2Fnetahtml%2FPTO%2Fsearch-adv.htm&r=0&f=S&l=50&d=PTXT&OS=ccl%2F706%2F%24+AND+APD%2F1%2F1%2F2003-%3E12%2F31%2F2003&RS=%28CCL%2F706%2F%24+AND+APD%2F20030101-%3E20031231%29&Query=ccl%2F706%2F%24+AND+APD%2F1%2F1%2F2003-%3E12%2F31%2F2003&TD=440&Srch1=%28706%2F%24.CCLS.+AND+%40AD%3E%3D20030101%3C%3D20031231%29&NextList2=Next+50+Hits'.format(page)
            website = urllib.urlopen(url).read()
            soup = BeautifulSoup(website)
            for i in soup.findAll('a'):
                if i.string is not None:
                    if len(i.string)==9:
                        serRes.append(str(i.string).replace(",",""))
        time.sleep(3)
    return serRes

# <codecell>

def scrapePTO(serRes):
    # Given a list of patent numbers scrapes data from the PTO search engine
    # Create dataframe for scraped data to be dumped into
    patdf = pd.DataFrame(columns=['PatentNum','Inventors','Abstract'])
    
    for pat in serRes:
        
        # Pull patent site
        url = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&d=PALL&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.htm&r=1&f=G&l=50&s1={0}.PN.&OS=PN/{0}&RS=PN/{0}'.format(pat)
        website = urllib.urlopen(url).read()
        soup = BeautifulSoup(website)
        
        # Grab all text raw, standardizing unicode ascii
        rawText = normalize('NFKD', soup.get_text()).encode('ASCII', 'ignore')
        
        # Define compile methods
        findClean1 = re.compile('\n+')
        findClean2 = re.compile('\s+')
        findAbs = re.compile('Abstract.+Inventors')
        findInv1 = re.compile('Inventors:.+Assignee:')
        findInv2 = re.compile('Inventors:.+Family ID:')
        findPatNo =  re.compile('United States Patent:\s\d{0,7}')
        findSummary = re.compile('SUMMARY OF THE INVENTION.+BRIEF DESCRIPTION')
        
        # Clean the raw text
        rawText = re.sub(findClean1," ", rawText)
        rawText = re.sub(findClean2," ", rawText)
        
        # Find the patent number, inventors and abstract
        find = re.search(findPatNo, rawText)
        patNo = find.group()
        patNo = patNo.replace("United States Patent: ", "")
        
        find = re.search(findInv1, rawText)
        # Patent may not be assigned, in which case a different compile method must be used
        if find is not None:
            inventors = find.group()
            inventors = inventors.replace(" Assignee:","")
            inventors = inventors.replace("Inventors: ","")
        else:
            find = re.search(findInv2, rawText)
            inventors = find.group()
            inventors = inventors.replace(" Family ID:","")
            inventors = inventors.replace("Inventors: ","")
            
        find = re.search(findAbs, rawText)
        abstract = find.group()
        abstract = abstract.replace(" Inventors","")
        abstract = abstract.replace("Abstract ","")
    
        row = pd.Series([patNo,inventors,abstract],index=['PatentNum','Inventors','Abstract'])
        patdf = patdf.append(row, ignore_index=True)
    
    return patdf

# <codecell>

# Scraping ACM from CiteSeer 
#cs = pd.DataFrame(columns=['Title','Authors','Abstract','JRef'])
cs = pd.DataFrame.from_csv('acm_abstracts2.csv')

url = 'http://citeseerx.ist.psu.edu/search?q=venue%3A%28Journal+of+the+ACM%29+AND+year%3A%5B1950+TO+2014%5D&t=doc&sort=ascdate&start=0'
website = urllib.urlopen(url).read()
soup = BeautifulSoup(website)
totalResults = int(soup.find(id='result_info').next.next.next.next.next.get_text())
print totalResults
if totalResults > 500:
    print "ERROR"
else:
    pages = range(0, totalResults, 10)
    
    for pg in pages:
        url = 'http://citeseerx.ist.psu.edu/search?q=venue%3A%28Journal+of+the+ACM%29+AND+year%3A%5B1950+TO+2014%5D&t=doc&sort=ascdate&start={0}'.format(pg)
        
        # Pull results list
        website = urllib.urlopen(url).read()
        soup = BeautifulSoup(website)
        
        resLinks = soup.find(id='result_list').findAll('a',{'class':'remove doc_details'})
        resultLinks = []
        for rs in resLinks:
            resultLinks.append(rs['href'])
        time.sleep(2)
        
        for link in resultLinks:
            url = 'http://citeseerx.ist.psu.edu' + link
            print url
            website = urllib.urlopen(url).read()
            soup = BeautifulSoup(website)
            
            if soup.findAll('div',{'class':'error'}):
                print "error"
            
            else:
                
                bibTex = soup.find(id='bibtex').get_text()
                                
                findAuthors = re.compile('author\s=\s\{[^\}]+')
                find = re.search(findAuthors,bibTex)
                authors = find.group()
                authors = authors.replace('author = {','')
                
                findTitle = re.compile('title\s=\s\{[^\}]+')
                find = re.search(findTitle,bibTex)
                title = find.group()
                title = title.replace('title = {','')
                
                findJRef1 = re.compile('journal.+')
                findJRef2 = re.compile('booktitle.+')
                findJRef3 = re.compile('year.+')
                try:
                    find = re.search(findJRef1,bibTex)
                    JRef = find.group() 
                except:
                    print "not JRef1"
                    JRef = ""
                if not JRef:
                    try:
                        find = re.search(findJRef2,bibTex)
                        JRef = find.group()
                    except:
                        print "not JRef2"
                        JRef = ""

                if not JRef:
                    try:
                        find = re.search(findJRef3,bibTex)
                        JRef = soup.find(id='docVenue').get_text()
                        JRef = JRef + find.group()
                        JRef = JRef.replace('Venue:', '')
                    except:
                        print "not JRef3"
                abstract = soup.find(id='abstract').p.get_text()
            
                row = pd.Series([title, authors, abstract, JRef],index=['Title','Authors','Abstract','JRef'])
                cs = cs.append(row, ignore_index=True)
                
                time.sleep(3)
            

# <codecell>

# Method for scraping data from the arXiv web API
def scrapeARXIV(maxRes,upperLimit,category, scrapeAll):
    # Make first call to get the total number of entries returned by the query  
    url = 'http://export.arxiv.org/api/query?search_query=cat:{0}&start=0&max_results=1&sortBy=submittedDate&sortOrder=ascending'.format(category)
    website = urllib.urlopen(url).read()
    soup = BeautifulSoup(website)
    totalResults = soup.findAll(re.compile('^opensearch:total'))[0].string
    total = int(totalResults)
    # Read and set the total number of entries matching the query 
    if scrapeAll:
        upLim = total
    else:
        upLim = upperLimit
    print "total", total
    print "upLim", upLim
    
    # Create empty dataframe to hold scraped data
    df = pd.DataFrame(columns=['Authors','Abstract','Title','JRef','SubmitDate','arxivID'])
    
    # Scrape data
    # Define the starting point for each call to the API
    starts = range(0,upLim/maxRes+1)
    starts = [x*maxRes for x in starts]
    
    # For each starting point make the call and scrape data from each entry returned
    for start in starts:
        if start + maxRes > total:
            maxRes = (upLim - start)-1
        url = 'http://export.arxiv.org/api/query?search_query=cat:{0}&start={1}&max_results={2}&sortBy=submittedDate&sortOrder=ascending'.format(category,start,maxRes)
        website = urllib.urlopen(url).read()
        soup = BeautifulSoup(website)
        
        # For each entry returned by the call find each variable and then append it to the data frame
        for ent in soup.findAll('entry'):
            # Define clean starting vars
            authors = []
            abstract = ""
            title = ""
            jref = []
            arxivID = ""
            # Find bibliographic variables
            authors = map(lambda x: cleanWebInput(str(x)), ent.findAll('author'))
            abstract = cleanWebInput(str(ent.summary))
            # Some titles have special ascii characters that have to be normalized 
            try: 
                # Run on titles that do not throw unicode error due to special characters
                title = str(ent.title.string)
            except:
                # Normalize titles with sepcial characters
                title = normalize('NFKD', ent.title.string).encode('ASCII', 'ignore')
            # Some entries have no journal reference and submission dates
            jRef = ent.findAll('arxiv:journal_ref')
            
            try:
                dSubmit = str(ent.published.string)
            except:
                dSubmit = "NONE"
            arxivID = str(ent.id.string)

            if len(jRef)>0:
                jRef = str(jRef[0].string)
            else:
                jRef = "None"
            # Write scraped variables out to the data frame
            row = pd.Series([authors,abstract,title,jRef,dSubmit,arxivID],index=['Authors','Abstract','Title','JRef','SubmitDate','arxivID'])
            df = df.append(row, ignore_index=True)
        # Sleep for 3 seconds to give the API server some rest
        time.sleep(3)
    return df

# <headingcell level=1>

# Scrape Biotech, AI, and Machine Learning Abstracts

# <codecell>

# Scrape data from Nature Biotechnology

#df = pd.DataFrame(columns=['Title','Authors','Abstract','JRef'])
df = pd.DataFrame.from_csv('bio_abstracts4.csv')
url = 'http://www.nature.com/nbt/archive/index.html'
website = urllib.urlopen(url).read()
soup = BeautifulSoup(website)
errorLinks = []
paperAbs = []
issueLinks = []
issueObs = soup.findAll("p", { "class" : "issue" })
for isobs in range(len(issueObs)):
    issueLinks.append(issueObs[isobs].a['href'])
issueLinks = issueLinks[195:384] # Subsetting the links to be considered

for isLink in issueLinks:
    url = 'http://www.nature.com' + isLink
    print url
    website = urllib.urlopen(url).read()
    soup = BeautifulSoup(website)
    paperLinks = []
    sproc = False
    ## Find all research paper links for the issue
    if soup.find(id='af'):
        if str(soup.find(id='af').next.next)=='\n':
            paperObs = soup.find(id='af').next.next.next.findAll("p", { "class" : "articlelinks" })
        else:
            paperObs = soup.find(id='af').next.next.findAll("p", { "class" : "articlelinks" })
        for papObs in range(len(paperObs)):
            paperLinks.append(paperObs[papObs].a['href'])
    elif soup.find(id='ra'):
        paperObs = soup.find(id='ra').findAll('p', { "class" : "articlelinks" })
        for papObs in range(len(paperObs)):
            if "abs" in paperObs[papObs].a['href']:
                paperLinks.append(paperObs[papObs].a['href'])
    elif soup.findAll('span',{'class':'categ_small'}):
        allHeaders = soup.findAll('span',{'class':'categ_small'})
        if [x for x in allHeaders if x.get_text()=='Research Articles']:
            resHeader = [x for x in allHeaders if x.get_text()=='Research Articles']
        elif [x for x in allHeaders if x.get_text()=='Research']:
            resHeader = [x for x in allHeaders if x.get_text()=='Research']
        elif [x for x in allHeaders if x.get_text()=='Research Papers']:
            resHeader = [x for x in allHeaders if x.get_text()=='Research Papers']
        else:
            resHeader = [x for x in allHeaders if x.get_text()=='Research Paper']
        if resHeader:
            paperObs =  resHeader[0].find_all_next("a", { "class" : "contentslink" })
            for papObs in range(len(paperObs)):
                paperLinks.append(paperObs[papObs]['href'])
            for papLink in range(len(paperLinks)):
                paperLinks[papLink] = paperLinks[papLink].replace('/pdf','/abs')
                paperLinks[papLink] = paperLinks[papLink].replace('.pdf','.html')
        sproc = True
        
    else:
        print "dead issue"
    time.sleep(1)
    
    if paperLinks:
        paperLinks = list(set(paperLinks))
        print paperLinks
        for link in paperLinks:
            url = 'http://www.nature.com' + link
            print url
            try:
                website = urllib.urlopen(url).read()
                soup = BeautifulSoup(website)
            except:
                print "Unauthorized"
                errorLinks.append(url)

            abstract = ""
            if sproc == False:
                # Abstract
                if soup.find(id='abs'):
                    abstract = soup.find(id='abs').get_text()
                    abstract = abstract.replace("Abstract", "")
                elif soup.find(id='abstract'):
                    abstract = soup.find(id='abstract').p.get_text()
                elif soup.find(id='first-paragraph'):
                    abstract = soup.find(id='first-paragraph').p.get_text()
                elif soup.find('span', { 'class' : 'articletext'} ):
                    abstract = soup.find('span', { 'class' : 'articletext'} ).get_text()
                elif soup.findAll("div", { "class" : "content" }):
                    abstract = soup.findAll("div", { "class" : "content" })[0].findAll('p')[1].get_text()
                else:
                    abstract = ""
                # Title
                if soup.find(id='atl'):
                    title = soup.find(id='atl').get_text()
                elif soup.findAll("h1", { "class" : "article-heading" }):
                    title = soup.findAll("h1", { "class" : "article-heading" })[0].get_text()
                else:
                    title = soup.findAll('h2')[0].get_text()
                # Journal Reference
                if soup.find(id='cite'):
                    JRef = soup.find(id='cite').get_text()
                elif soup.findAll("dl", { "class" : "citation"}):
                    JRef = soup.findAll("dl", { "class" : "citation"})[0].get_text()
                    JRef = JRef.replace('\t', " ")
                    JRef = JRef.replace('\n', " ")
                    JRef = JRef.replace("     ", " ")
                else:
                    JRef = soup.findAll('span',{'class':'blacksml'})[0].get_text()
                    JRef = JRef.replace('\t', " ")
                    JRef = JRef.replace('\n', " ")
                    JRef = JRef.replace("     ", " ")
                    JRef = JRef.replace("   - ", "")
                # Authors
                if soup.find(id='aug'):
                    authors = soup.find(id='aug').get_text()
                    authors = authors.replace("\n"," ")
                    authors = authors.replace("\t\t\t","")
                elif soup.findAll("li", { "class" : "vcard"}):
                    authors = []
                    authorObjs = soup.findAll("li", { "class" : "vcard"})
                    for auth in authorObjs:
                        authors.append(auth.a.get_text())
                else:
                    authors = soup.findAll('span',{'class':'author'})[0].get_text()
                
                if abstract:
                    row = pd.Series([title, authors, abstract, JRef],index=['Title','Authors','Abstract','JRef'])
                    df = df.append(row, ignore_index=True)
                
                time.sleep(1)
                
            else:
                if soup.find('h1', { 'class' : 'page-header'} ):
                    if "Research" in soup.find('h1', { 'class' : 'page-header'} ).get_text():
                        # Abstract
                        if soup.find(id='abs'):
                            abstract = soup.find(id='abs').get_text()
                            abstract = abstract.replace("Abstract", "")
                        elif soup.find(id='abstract'):
                            abstract = soup.find(id='abstract').p.get_text()
                        elif soup.find(id='first-paragraph'):
                            abstract = soup.find(id='first-paragraph').p.get_text()
                        elif soup.find('span', { 'class' : 'articletext'} ):
                            abstract = soup.find('span', { 'class' : 'articletext'} ).get_text()
                        elif soup.findAll("div", { "class" : "content" }):
                            abstract = soup.findAll("div", { "class" : "content" })[0].findAll('p')[1].get_text()
                        else:
                            abstract = ""
                        # Title
                        if soup.find(id='atl'):
                            title = soup.find(id='atl').get_text()
                        elif soup.findAll("h1", { "class" : "article-heading" }):
                            title = soup.findAll("h1", { "class" : "article-heading" })[0].get_text()
                        else:
                            title = soup.findAll('h2')[0].get_text()
                        # Journal Reference
                        if soup.find(id='cite'):
                            JRef = soup.find(id='cite').get_text()
                        elif soup.findAll("dl", { "class" : "citation"}):
                            JRef = soup.findAll("dl", { "class" : "citation"})[0].get_text()
                            JRef = JRef.replace('\t', " ")
                            JRef = JRef.replace('\n', " ")
                            JRef = JRef.replace("     ", " ")
                        else:
                            JRef = soup.findAll('span',{'class':'blacksml'})[0].get_text()
                            JRef = JRef.replace('\t', " ")
                            JRef = JRef.replace('\n', " ")
                            JRef = JRef.replace("     ", " ")
                            JRef = JRef.replace("   - ", "")
                        # Authors
                        if soup.find(id='aug'):
                            authors = soup.find(id='aug').get_text()
                            authors = authors.replace("\n"," ")
                            authors = authors.replace("\t\t\t","")
                        elif soup.findAll("li", { "class" : "vcard"}):
                            authors = []
                            authorObjs = soup.findAll("li", { "class" : "vcard"})
                            for auth in authorObjs:
                                authors.append(auth.a.get_text())
                        else:
                            authors = soup.findAll('span',{'class':'author'})[0].get_text()
                        
                        if abstract:
                            row = pd.Series([title, authors, abstract, JRef],index=['Title','Authors','Abstract','JRef'])
                            df = df.append(row, ignore_index=True)
                        
                        time.sleep(1)
                    else:
                        print "NOT CAPTURED"
                else:
                    print "NOT CAPTURED"

# <codecell>

# Storing Biotech Abstracts
df2 = df.copy()
for row_index, row in df2.iterrows():
    row['Title'] = normalize('NFKD', unicode(row['Title'])).encode('ASCII', 'ignore')
    row['Abstract'] = normalize('NFKD', unicode(row['Abstract'])).encode('ASCII', 'ignore')
    authors = []
    if type(row['Authors'])==list:
        for i in row['Authors']:
            authors.append(normalize('NFKD', unicode(i)).encode('ASCII', 'ignore'))
        row['Authors'] = authors
    else:
        row['Authors'] = normalize('NFKD', unicode(row['Authors'])).encode('ASCII', 'ignore')
    row['JRef'] = normalize('NFKD', unicode(row['JRef'])).encode('ASCII', 'ignore')
    
df2.to_csv('bio_abstracts5.csv')

# <codecell>

# Scrape arXiv for artificial intelligence and machine learning abstracts
df = scrapeARXIV(100, 0, 'cs.ai', True)
df.to_csv('ai_abstracts.csv')

df = scrapeARXIV(100, 0, 'cs.lg', True)
df.to_csv('lg_abstracts.csv')

df = scrapeARXIV(100, 0, 'stat.ml', True)
df.to_csv('ml_abstracts.csv')

# <headingcell level=1>

# Scrape Biotech and AI Patents

# <codecell>

def addPats2csv(df, yr,first,name):
    filename = '{0}_patents.csv'.format(name)
    if first:
        pfCSV = pd.DataFrame(columns=['PatentNum','PatYear','Inventors', 'Abstract'])
    else:
        # Add new patents to the csv
        pfCSV = pd.DataFrame.from_csv(filename)
    for row_index, row in df.iterrows():
        patNum = row['PatentNum']
        patYear = int(yr)
        patInv = row['Inventors']
        patAbs = row['Abstract']
        
        row = pd.Series([patNum,patYear,patInv,patAbs],index=['PatentNum','PatYear','Inventors', 'Abstract'])
        pfCSV = pfCSV.append(row, ignore_index=True)
        
    pfCSV.to_csv(filename)

# <codecell>

"""
#bioICL = pd.DataFrame(['A01H  100','A01H  400','A61K 3800','A61K 3900','A61K 4800','C02F  334','C07G 1100','C07G 1300','C07G 1500','C07K  400','C07K 1400','C07K 1600','C07K 1700','C07K 1900','G01N 27327','G01N 3353','G01N 33531','G01N 33532','G01N 33533','G01N 33534','G01N 33535','G01N 33536','G01N 33537','G01N 33538','G01N 33539','G01N 3354','G01N 33541','G01N 33542','G01N 33543','G01N 33544','G01N 33545','G01N 33546','G01N 33547','G01N 33548','G01N 33549','G01N 3355','G01N 33551','G01N 33552','G01N 33553','G01N 33554','G01N 33555','G01N 33556','G01N 33557','G01N 33558','G01N 33559','G01N 3357','G01N 33571','G01N 33572','G01N 33573','G01N 33574','G01N 33575','G01N 33576','G01N 33577','G01N 33578','G01N 33579','G01N 3368','G01N 3374','G01N 3376','G01N 3378','G01N 3388','G01N 3392'], columns=['icl'])
#bioICLclass = pd.DataFrame(['C12M','C12N','C12P','C12Q','C12S'],columns=['icl_class'])
#bioPats1 = pdf3.merge(bioICL,on=['icl'])
#bioPats2 = pdf3.merge(bioICLclass,on=['icl_class'])
#bioPats = pd.concat([bioPats1,bioPats2])

bioPats.groupby('gyear').size().plot()
bioPats.ncited.describe()


# find top x% of cited patents in a year
A = bioPats[(bioPats.gyear==1999)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.99)
A = bioPats[(bioPats.gyear==1999)&(bioPats.ncited>=quantCut)]
bioPatList1999 = list(A.patent)
print bioPatList1999

A = bioPats[(bioPats.gyear==1998)&(bioPats.subcat==33)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.99)
A = bioPats[(bioPats.gyear==1998)&(bioPats.ncited>=quantCut)]
bioPatList1998 = list(A.patent)
print bioPatList1998

A = bioPats[(bioPats.gyear==1997)&(bioPats.subcat==33)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.99)
A = bioPats[(bioPats.gyear==1997)&(bioPats.ncited>=quantCut)]
bioPatList1997 = list(A.patent)
print bioPatList1997

A = bioPats[(bioPats.gyear==1996)&(bioPats.subcat==33)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.99)
A = bioPats[(bioPats.gyear==1996)&(bioPats.ncited>=quantCut)]
bioPatList1996 = list(A.patent)
print bioPatList1996
"""

# <codecell>

# Create bio patents data set, top 2%
bioPatList1999 = [5928880, 5989835, 5866345, 5942443, 5948767, 5935576, 5858746, 5869336, 5871974, 5876997, 5877397, 5877399, 5932462, 5955358, 5968830, 5976833, 5976862, 6004788, 5861242, 5866336, 5869242, 5874219, 5876930, 5885775, 5888819, 5922537, 5925517, 5925525, 5939250, 5952172, 5958672, 5958694, 5972615, 5972619, 5981180, 6001574, 6004744, 5866363, 5928905, 5962258, 5965408, 5965410, 5856174, 5922591]
bioPatList1998 = [5750119, 5830464, 5750349, 5763192, 5814524, 5719060, 5709854, 5712146, 5723323, 5750376, 5753506, 5770417, 5770434, 5783431, 5801154, 5830721, 5851832, 5705348, 5714331, 5733729, 5744305, 5763239, 5770369, 5776672, 5780234, 5795714, 5800992, 5811238, 5824469, 5824473, 5824485, 5830655, 5837458, 5837832, 5843655, 5846708, 5846710, 5849486, 5736330, 5814476, 5817483, 5824513, 5824514, 5830696, 5834252, 5854033, 5716825, 5807522, 5843767]
bioPatList1997 = [5593846, 5674698, 5635358, 5658802, 5677195, 5677196, 5602040, 5670488, 5693622, 5665582, 5672491, 5591578, 5593838, 5593839, 5599668, 5601982, 5604097, 5605793, 5620850, 5622824, 5639603, 5667972, 5691141, 5695940, 5700637, 5700642, 5599695, 5604130, 5605662, 5632957, 5674743]
bioPatList1996 = [5510270, 5532128, 5523520, 5589466, 5536490, 5491084, 5500365, 5529914, 5573934, 5489508, 5492806, 5494810, 5508164, 5525464, 5527681, 5538848, 5545531, 5547839, 5552270, 5556752, 5565324, 5573906, 5573909, 5587128, 5512439, 5512463]

pf = scrapePTO(bioPatList1999)
addPats2csv(pf,1999,True,'bio')
pf = scrapePTO(bioPatList1998)
addPats2csv(pf,1998,False,'bio')
pf = scrapePTO(bioPatList1997)
addPats2csv(pf,1997,False,'bio')
pf = scrapePTO(bioPatList1996)
addPats2csv(pf,1996,False,'bio')


# <codecell>

"""
# find top x% of cited patents in a year
A = pdf3[(pdf3.gyear==1999)&(pdf3.nclass==706)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.90)
A = pdf3[(pdf3.gyear==1999)&(pdf3.nclass==706)&(pdf3.ncited>=quantCut)]
aiPatList1999 = list(A.patent)
print aiPatList1999

A = pdf3[(pdf3.gyear==1998)&(pdf3.nclass==706)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.90)
A = pdf3[(pdf3.gyear==1998)&(pdf3.nclass==706)&(pdf3.ncited>=quantCut)]
aiPatList1998 = list(A.patent)
print aiPatList1998

A = pdf3[(pdf3.gyear==1997)&(pdf3.nclass==706)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.90)
A = pdf3[(pdf3.gyear==1997)&(pdf3.nclass==706)&(pdf3.ncited>=quantCut)]
aiPatList1997 = list(A.patent)
print aiPatList1997

A = pdf3[(pdf3.gyear==1996)&(pdf3.nclass==706)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.90)
A = pdf3[(pdf3.gyear==1996)&(pdf3.nclass==706)&(pdf3.ncited>=quantCut)]
aiPatList1996 = list(A.patent)
print aiPatList1996
"""

# <codecell>

# Create ai patents data set, top 10%
aiPatList1999 = [5862304, 5870731, 5875285, 5884294, 5893083, 5903886, 5924086, 5933818, 5953713, 5970482, 5987443, 5995951, 5995956, 6003021]
aiPatList1998 = [5706402, 5720007, 5727128, 5727129, 5729661, 5729662, 5740326, 5742738, 5745382, 5745652, 5751914, 5754738, 5765028, 5778156, 5781703, 5784539, 5787234, 5799292, 5802253, 5805775, 5806056, 5809490, 5809493, 5819007, 5822744, 5822745, 5832182, 5835683, 5842194, 5845270, 5845272]
aiPatList1997 = [5594837, 5613039, 5630025, 5634087, 5640491, 5644686, 5649061, 5651099, 5659667, 5671333, 5671335, 5673369, 5675710, 5675711, 5677997, 5692107, 5694523, 5696885, 5701400, 5704011, 5704017]
aiPatList1996 = [5481647, 5483620, 5485550, 5488697, 5493729, 5504837, 5506937, 5515477, 5517405, 5546507, 5555346, 5561738, 5574828, 5581657, 5581664, 5586218, 5586219]

pf = scrapePTO(aiPatList1999)
addPats2csv(pf,1999,True,'ai')
pf = scrapePTO(aiPatList1998)
addPats2csv(pf,1998,False,'ai')
pf = scrapePTO(aiPatList1997)
addPats2csv(pf,1997,False,'ai')
pf = scrapePTO(aiPatList1996)
addPats2csv(pf,1996,False,'ai')

# <headingcell level=1>

# Calculate Similairty - Biotech

# <codecell>

# Analysis and similarity matrix
bioPats = pd.DataFrame.from_csv('bio_patents.csv')
bioAbs = pd.DataFrame.from_csv('bio_abstracts.csv')
sim = pd.DataFrame(columns=['PatNum', 'PatYear', 'PatInventors','PapTitle','PapAuthors','PapJRef','PapYear','Similarity'])
#sim = pd.DataFrame.from_csv('bio_similarity.csv')

# <codecell>

# Analysis and similarity matrix
findYear = re.compile('\(\d\d\d\d\)')
bioPats = bioPats
t0 = time.clock()

for pat_ind, pat in bioPats.iterrows():
    patNum = pat['PatentNum']
    patYear = pat['PatYear']
    patInv = pat['Inventors']
    patAbs = pat['Abstract']
    print "PatNum", patNum

    for row_index,row in bioAbs.iterrows():
        if row_index%100 == 0:
            print row_index
        papTitle = row['Title']
        papAuthors = row['Authors']
        papJRef = row['JRef']
        find = re.search(findYear, papJRef)
        papYear = int(find.group()[1:5])
        papAbs = row['Abstract']
    
        sml = similarity(patAbs,papAbs)
        
        row = pd.Series([patNum,patYear,patInv,papTitle,papAuthors,papJRef,papYear,sml],index=['PatNum','PatYear','PatInventors','PapTitle','PapAuthors','PapJRef','PapYear','Similarity'])
        sim = sim.append(row, ignore_index=True)
    
t = time.clock() - t0
print "Minutes Lapsed:", (t/60)
sim.to_csv('bio_similarity.csv')

# <headingcell level=1>

# Calculate Similarity - AI & Machine Learning

# <codecell>

aimlAbs = pd.DataFrame.from_csv('aiml_abstracts.csv')
aiPats = pd.DataFrame.from_csv('ai_patents.csv')
aiSim = pd.DataFrame(columns=['PatNum', 'PatYear', 'PatInventors','PapTitle','PapAuthors','PapJRef','PapID','PapYear','Similarity'])
#aiSim = pd.DataFrame.from_csv('ai_similarity.csv')

# <codecell>

# Analysis and similarity matrix
aiPats = aiPats[1:27]
t0 = time.clock()

for pat_ind, pat in aiPats.iterrows():
    patNum = pat['PatentNum']
    patYear = pat['PatYear']
    patInv = pat['Inventors']
    patAbs = pat['Abstract']
    print "PatNum", patNum

    for row_index,row in aimlAbs.iterrows():
        if row_index%500 == 0:
            print row_index
        papTitle = row['Title']
        papAuthors = row['Authors']
        papJRef = row['JRef']
        papID = row['arxivID']
        papYear = row['Year']
        papAbs = row['Abstract']
    
        sim = similarity(patAbs,papAbs)
        
        row = pd.Series([patNum,patYear,patInv,papTitle,papAuthors,papJRef,papID,papYear,sim],index=['PatNum','PatYear','PatInventors','PapTitle','PapAuthors','PapJRef','PapID','PapYear','Similarity'])
        aiSim = aiSim.append(row, ignore_index=True)
    
t = time.clock() - t0
print "Minutes Lapsed:", (t/60)
aiSim.to_csv('ai_similarity1.csv')

# <codecell>

print aimlAbs.Abstract

# <headingcell level=1>

# Calculate Similarity - Papers

# <codecell>

aimlAbs = pd.DataFrame.from_csv('aiml_abstracts.csv')
aiPapers = pd.DataFrame.from_csv('top_ai_papers.csv')
aiPaperSim = pd.DataFrame(columns=['TopPaperTitle', 'TopPaperAuthors', 'TopPaperYear','PapTitle','PapAuthors','PapJRef','PapYear','Similarity'])

# <codecell>

# Analysis and similarity matrix for papers
t0 = time.clock()

for toppap_ind,toppap in aiPapers.iterrows():
    toppapAbs = toppap['Abstract']
    toppapTitle = toppap['Title']
    toppapAuthors = toppap['Authors']
    toppapYear = toppap['Year']
    print "Paper", toppapTitle
    
    for row_index,row in aimlAbs.iterrows():
        if row_index%500 == 0:
            print row_index
        papTitle = row['Title']
        papAuthors = row['Authors']
        papJRef = row['JRef']
        papYear = row['PapYear']
        papAbs = row['Abstract']
        
        sim = similarity(toppapAbs,papAbs)
        
        row = pd.Series([toppapTitle,toppapAuthors,toppapYear,papTitle,papAuthors,papJRef,papYear,sim],index=['TopPaperTitle', 'TopPaperAuthors', 'TopPaperYear','PapTitle','PapAuthors','PapJRef','PapYear','Similarity'])
        aiPaperSim = aiPaperSim.append(row,ignore_index=True)
        
    
t = time.clock() - t0
print "Minutes Lapsed:", (t/60)
aiPaperSim.to_csv('paper_similarity2.csv')

# <codecell>

##########

# <codecell>

#sm = pd.DataFrame.from_csv('bio_similarity3.csv')
smPats = list(set(sm.PatNum))
years = list(set(sm.Year))
for pat in smPats: 
    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(111)
    ax1.plot(years, sm[sm.PatNum==pat].groupby(['Year']).Similarity.mean())
    ax1.set_title('') 
    ax1.set_ylabel('')
    ax1.set_xlabel('Year')
    ax1.set_ylim([0,0.5])
    plt.show()

# <codecell>

print cs
cs2 = cs.copy()
for row_index, row in cs2.iterrows():
    row['Title'] = normalize('NFKD', unicode(row['Title'])).encode('ASCII', 'ignore')
    row['Abstract'] = normalize('NFKD', unicode(row['Abstract'])).encode('ASCII', 'ignore')
    row['Authors'] = normalize('NFKD', unicode(row['Authors'])).encode('ASCII', 'ignore')
    row['JRef'] = normalize('NFKD', unicode(row['JRef'])).encode('ASCII', 'ignore')
    

cs2.to_csv('acm_abstracts2.csv')

# <codecell>

cs = pd.DataFrame.from_csv('acm_abstracts3.csv')
# drop if missing abstract
cs = cs.dropna(axis=0)
print cs
cs['AbsLen'] = int()
absLen = []
for row_index, row in cs.iterrows():
    absLen.append(len(row['Abstract']))
    cs.AbsLen[row_index] = len(row['Abstract'])

cs['PapYear'] = int()
findYear = re.compile('\{\d\d\d\d\}')
for row_index, row in cs.iterrows():
    JRef = row['JRef']
    find = re.search(findYear, JRef)
    year = int(find.group()[1:5])
    cs.PapYear[row_index] = year

plt.hist(absLen, bins=50)
plt.show()
#cs.to_csv('acm_abstracts4.csv')

# <codecell>

RSAAbs = 'An encryption method is presented with the novel property that publicly revealing an encryption key does not thereby reveal the corresponding decryption key. This has two important consequences: (1) Couriers or other secure means are not needed to transmit keys, since a message can be enciphered using an encryption key publicly revealed by the intented recipient. Only he can decipher the message, since only he knows the corresponding decryption key. (2) A message can be “signed” using a privately held decryption key. Anyone can verify this signature using the corresponding publicly revealed encryption key. Signatures cannot be forged, and a signer cannot later deny the validity of his signature. This has obvious applications in “electronic mail” and “electronic funds transfer” systems. A message is encrypted by representing it as a number M, raising M to a publicly specified power e, and then taking the remainder when the result is divided by the publicly specified product, n, of two large secret primer numbers p and q. Decryption is similar; only a different, secret, power d is used, where e * d ≡ 1(mod (p - 1) * (q - 1)). The security of the system rests in part on the difficulty of factoring the published divisor, n.'
RSAPat = 'A cryptographic communications system and method. The system includes a communications channel coupled to at least one terminal having an encoding device and to at least one terminal having a decoding device. A message-to-be-transferred is enciphered to ciphertext at the encoding terminal by first encoding the message as a number M in a predetermined set, and then raising that number to a first predetermined power (associated with the intended receiver) and finally computing the remainder, or residue, C, when the exponentiated number is divided by the product of two predetermined prime numbers (associated with the intended receiver). The residue C is the ciphertext. The ciphertext is deciphered to the original message at the decoding terminal in a similar manner by raising the ciphertext to a second predetermined power (associated with the intended receiver), and then computing the residue, M, when the exponentiated ciphertext is divided by the product of the two predetermined prime numbers associated with the intended receiver. The residue M corresponds to the original encoded message M.'

print len(RSAAbs)
print len(RSAPat)
print cs[(cs.AbsLen>(len(RSAPat)-150)) & (cs.AbsLen<(len(RSAPat)+150))]
print cs[(cs.AbsLen>(len(RSAAbs)-150)) & (cs.AbsLen<(len(RSAAbs)+150))]

sm = pd.DataFrame(columns=['RSATitle','RSAYear','RSAAuthors','PapTitle','PapAuthors','PapJRef','PapYear','Similarity'])

# <codecell>

# Analysis and similarity matrix
findYear = re.compile('\(\d\d\d\d\)')
ACMAbs = cs[(cs.AbsLen>(len(RSAPat)-150)) & (cs.AbsLen<(len(RSAPat)+150))]
t0 = time.clock()

RSAAbs = 'An encryption method is presented with the novel property that publicly revealing an encryption key does not thereby reveal the corresponding decryption key. This has two important consequences: (1) Couriers or other secure means are not needed to transmit keys, since a message can be enciphered using an encryption key publicly revealed by the intented recipient. Only he can decipher the message, since only he knows the corresponding decryption key. (2) A message can be “signed” using a privately held decryption key. Anyone can verify this signature using the corresponding publicly revealed encryption key. Signatures cannot be forged, and a signer cannot later deny the validity of his signature. This has obvious applications in “electronic mail” and “electronic funds transfer” systems. A message is encrypted by representing it as a number M, raising M to a publicly specified power e, and then taking the remainder when the result is divided by the publicly specified product, n, of two large secret primer numbers p and q. Decryption is similar; only a different, secret, power d is used, where e * d ≡ 1(mod (p - 1) * (q - 1)). The security of the system rests in part on the difficulty of factoring the published divisor, n.'
RSATitle = 'A method for obtaining digital signatures and public-key cryptosystems'
RSAYear = 1978
RSAAuthors = ['Ron Rivest', 'Adi Shamir', 'Leonard Adleman']

for row_index,row in ACMAbs.iterrows():
    if row_index%100 == 0:
        print row_index
    papTitle = row['Title']
    papAuthors = row['Authors']
    papJRef = row['JRef']   
    papAbs = row['Abstract']
    papYear = row['PapYear']
    
    sim = similarity(RSAAbs,papAbs)
    
    row = pd.Series([RSATitle,RSAYear,RSAAuthors,papTitle,papAuthors,papJRef,papYear,sim],index=['RSATitle','RSAYear','RSAAuthors','PapTitle','PapAuthors','PapJRef','PapYear','Similarity'])
    sm = sm.append(row, ignore_index=True)
    
t = time.clock() - t0
print "Minutes Lapsed:", (t/60)
#sm.to_csv('.csv')

