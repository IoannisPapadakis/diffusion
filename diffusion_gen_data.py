"""
Web Scraping Academic Abstracts and Patent Abstracts for Biotech and AI
Nathan Goldschlag
August 25, 2014
Version 1.0
Written in Python 2.7

This python program contains the methods used to scrape academic abstracts from the 
web and create a dataset of patent abstracts. These abstracts are scraped from 
Nature Biotechnology and arXiv. The patents are put together using the NBER patent 
database and yearly full text patent files, .json dictionaries with abstracts. 

These methods were used in the paper The Knowledge Diffusion Effects of Patents.

"""
## IMPORT LIBRARIES
import urllib
import re
from bs4 import BeautifulSoup
import time
import pandas as pd
from unicodedata import normalize
import json
import numpy as np

absPath = 'd:/patent_data/abstracts'
nberPath = 'd:/patent_data/nber_patent_data'
difPath = 'd:/diffusion_data'

def cleanWebInput(text):
    """
    Removes common html tags from a string of text
    """
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

def doNatureScrape(csvName):
    """
    Performs the scraping of publication data from the Nature Biotechnology website, returns a dataframe with the results.
    Scrapes paper title, paper authors, paper abstract, and journal reference. In the Nature Biotechnology archive the HTML structure
    changes periodically, so a significant amount of special processing if/elif/else statements are used to handle all of the exceptions in dealing with older issues
    Parameters:
    csvName = name of csv file saved after scraping
    """
    df = pd.DataFrame(columns=['Title','Authors','Abstract','JRef'])
    url = 'http://www.nature.com/nbt/archive/index.html'
    website = urllib.urlopen(url).read()
    soup = BeautifulSoup(website)
    errorLinks = []
    paperAbs = []
    issueLinks = []
    issueObs = soup.findAll("p", { "class" : "issue" })
    for isobs in range(len(issueObs)):
        issueLinks.append(issueObs[isobs].a['href'])
    #issueLinks = issueLinks[195:384] # Subsetting the links to be considered
    
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
    # Clean issues with unicode data
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
    # drop any duplicates
    df2 = df2.drop_duplicates('JRef')
    df2.to_csv(csvName)
    return df2

def scrapeARXIV(maxRes,upperLimit,category, scrapeAll):
    """
    Scrapes publication data from the arXiv API. Scrapes paper authors, paper abstract,paper title, journal reference, 
    submission date, and arxivID
    Parameters:
    maxRes = maximum number of results per page (ex:100)
    upperLimit = maximum number of results to be returned, if scrapeAll = True then upperLimit is disregarded and all results are returned
    category = category submitted in the search query (ex: cs.ai, cs.lg)
    scrapeAll = T/F whether all results should be scraped, if False then upperLimit used to limit the number of results scraped
    """
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
        print start,'of',total
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
                # Normalize titles with special characters
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

def getPats():
    """
    Loads NBER patent data, 1976-2006, and returns two data frames. One df with biotech patents, defined
    by the icl and icl_class, and one df with ai patents, defined by nclass.
    """
    # Import patent data
    orig7606 = pd.read_stata(nberPath+'/orig_gen_76_06.dta')
    pat7606 = pd.DataFrame.from_csv(nberPath+'/pat76_06_ipc.csv')

    # Since pat76-06 has multiple records for each patent = number of assigness, remove duplicates so there is one record per patent
    pat7606 = pat7606.drop_duplicates(cols=['patent'])
    pat7606 = pat7606.drop(['uspto_assignee','year'],axis=1)

    # Put originality, generality, and citation measures onto the pat76-06 dataframe
    pat7606 = pd.merge(pat7606, orig7606, on=['patent'], how='left')

    # Filter down to only bio-pats using hard-coded classifications from (REFERENCE)
    bioICL = pd.DataFrame(['A01H  100','A01H  400','A61K 3800','A61K 3900','A61K 4800','C02F  334','C07G 1100','C07G 1300','C07G 1500','C07K  400','C07K 1400','C07K 1600','C07K 1700','C07K 1900','G01N 27327','G01N 3353','G01N 33531','G01N 33532','G01N 33533','G01N 33534','G01N 33535','G01N 33536','G01N 33537','G01N 33538','G01N 33539','G01N 3354','G01N 33541','G01N 33542','G01N 33543','G01N 33544','G01N 33545','G01N 33546','G01N 33547','G01N 33548','G01N 33549','G01N 3355','G01N 33551','G01N 33552','G01N 33553','G01N 33554','G01N 33555','G01N 33556','G01N 33557','G01N 33558','G01N 33559','G01N 3357','G01N 33571','G01N 33572','G01N 33573','G01N 33574','G01N 33575','G01N 33576','G01N 33577','G01N 33578','G01N 33579','G01N 3368','G01N 3374','G01N 3376','G01N 3378','G01N 3388','G01N 3392'], columns=['icl'])
    bioICLclass = pd.DataFrame(['C12M','C12N','C12P','C12Q','C12S'],columns=['icl_class'])
    bioPats1 = pat7606.merge(bioICL,on=['icl'])
    bioPats2 = pat7606.merge(bioICLclass,on=['icl_class'])
    bioPats = pd.concat([bioPats1,bioPats2])
    
    # Filter down to ai pats
    aiPats = pat7606[pat7606.nclass==706]
    
    return bioPats, aiPats

def citeQuantCut(df,year,percentile):
    """
    Takes a dataframe of patent data, and returns a df containing only the given percentile
    by number of citations in a given year.
    Parameters:
    df = dataframe of nber patent data
    year = grant year
    percentile = 1 minus the percentile wanted, so .99 for top 1%
    """
    # find top x% of cited patents in a year
    dfFiltered = df[(df.gyear==year)]
    quantCut = dfFiltered.ncited.quantile(percentile)
    dfFiltered = dfFiltered[(dfFiltered.gyear==year)&(dfFiltered.ncited>=quantCut)]
    return dfFiltered

def loadPatentAbstracts(df):
    """
    Takes a df of patent data (patent number field named "patent") and returns a dataframe with
    the patent abstract and patent title added to the input dataframe. The abstract and title come from
    yearly .json files of full text patent data 1976-2014.
    Parameters:
    df = dataframe of patent data with at least patent numbers, patent number field named "patent"
    """
    # load patent abstract json's for the dataframe piece
    years =  list(set(df.gyear))
    allDict = {}
    for yr in years:
        f = open(absPath+'/patAbs{0}.json'.format(yr),'r')
        patDict = json.load(f)
        f.close()
        allDict = dict(allDict,**patDict)
        patDict = ''
    
    # put abstracts into dataframe
    df['patent_title'] = ''
    df['patent_abstract'] = ''
    for row_index, row in df.iterrows():
        try:
            abstract = allDict[str(row['patent'])]['abstract']
        except:
            abstract = 'NULL'
        try:
            title = allDict[str(row['patent'])]['title']
        except:
            title = 'NULL'
        df.patent_abstract[row_index] = abstract
        df.patent_title[row_index] = title
    allDict = ''
    return df


def main():
    # User input of processes to run
    params = {'doBioScrape':'null', 'doarXivScrape':'null', 'doPatents':'null'}
    for p in params.keys():
        print 'would you like to {0} (y/n)? '.format(p),
        answer = 'null'
        while answer not in ['y','n']:
            answer = raw_input()
            if answer in ['y','n']:
                params[p] = answer
            else:
                print 'invalid input'
    
    if params['doBioScrape']=='y': 
        ## Scrape biotech, AI, and machine learning abstracts
        # Storing Biotech Abstracts
        dfBio = doNatureScrape('bio_abstracts.csv')
        
        # post-processing
        # reindex before processing
        dfBio.index = np.arange(1,len(dfBio)+1)
        
        # extract year from journal ref
        findYear = re.compile('\(\d\d\d\d\)')
        dfBio['JRefYear'] = int()
        for row_index, row in dfBio.iterrows():
            papJRef = row['JRef']
            find = re.search(findYear, papJRef)
            dfBio.JRefYear[row_index] = int(find.group()[1:5])

        #drop records with duplicate journal refs
        dfBio = dfBio.drop_duplicates(['JRef'])
        
        # drop records outside the range of study, 1983-2013
        dfBio = dfBio[(dfBio.JRefYear>=1983)&(dfBio.JRefYear<=2013)]

        # reindex before storing
        dfBio.index = np.arange(1,len(dfBio)+1)
                
        dfBio.to_csv(difPath+'/bio_abstracts.csv')
        
    
    if params['doarXivScrape']=='y': 
        # Scrape arXiv for artificial intelligence and machine learning abstracts
        print 'Scrape arXiv AI'
        dfAI = scrapeARXIV(100, 0, 'cs.ai', True)
        dfAI.to_csv(difPath+'/arxiv_ai_abstracts.csv')
        print 'Scrape arXiv LG'
        dfLG = scrapeARXIV(100, 0, 'cs.lg', True)
        dfLG.to_csv(difPath+'/arxiv_lg_abstracts.csv')
        print 'Scrape arXiv ML'
        dfML = scrapeARXIV(100, 0, 'stat.ml', True)
        dfML.to_csv(difPath+'/arxiv_ml_abstracts.csv')
        
        # combine the abstract dataframes
        dfAIML = pd.concat([dfAI,dfLG,dfML])
        
        # post-processing
        # reindex before creating new vars
        dfAIML.index = np.arange(1,len(dfAIML)+1)
        
        # extract year from arXiv submit date 
        dfAIML['SubmitYear'] = int()
        for row_index, row in dfAIML.iterrows():
            dfAIML.SubmitYear[row_index] = row['SubmitDate'][0:4]
        
        # extract year  from JRef 
        findYear1 = re.compile('\(\d\d\d\d\)')
        findYear2 = re.compile('(?<!\d\d\d\d\-)\d\d\d\d(?!\-\d\d\d\d)')
        dfAIML['JRefYear'] = int()
        for row_index, row in dfAIML.iterrows():
            year = int()
            papJRef = row['JRef']
            find1 = re.findall(findYear1,papJRef)
            if find1:
                yrs = []
                for i in find1:
                    yrs.append(i[1:5])
                for i in yrs:
                    if int(i)<2015 and int(i)>1960:
                        year = int(i)
                        dfAIML.JRefYear[row_index] = int(year)
            find2 = re.findall(findYear2,papJRef)
            if find2 and not year:
                for i in find2:
                    if int(i)<2015 and int(i)>1960:
                        year = int(i)
                        dfAIML.JRefYear[row_index] = int(year)
        
        # Drop records where the year could not be extracted from the JRef
        dfAIML = dfAIML[dfAIML.JRefYear!=0]
        
        # drop duplicate abstracts (there may still be duplicate titles but with diff abstracts)
        dfAIML = dfAIML.drop_duplicates(['Abstract'])
        
        # drop records outside the range of study, 1993-2013
        dfAIML = dfAIML[(dfAIML.JRefYear>=1993)&(dfAIML.JRefYear<=2013)]
        
        # reindex before storing
        dfAIML.index = np.arange(1,len(dfAIML)+1)
        
        dfAIML.to_csv(difPath+'/aiml_abstracts.csv')
    
    if params['doPatents']=='y': 
        ## Load NBER patent data
        bioPats, aiPats = getPats()
        # find top 1% of cited bio patents for a set of years
        bioPatsTop = citeQuantCut(bioPats,1999,.99)
        bioPatsTop = pd.concat([bioPatsTop,citeQuantCut(bioPats,1998,.99)])
        bioPatsTop = pd.concat([bioPatsTop,citeQuantCut(bioPats,1997,.99)])
        bioPatsTop = pd.concat([bioPatsTop,citeQuantCut(bioPats,1996,.99)])
        # find top 10% of cited ai patents for a set of years
        aiPatsTop = citeQuantCut(aiPats,1999,.90)
        aiPatsTop = pd.concat([aiPatsTop,citeQuantCut(aiPats,1998,.90)])
        aiPatsTop = pd.concat([aiPatsTop,citeQuantCut(aiPats,1997,.90)])
        aiPatsTop = pd.concat([aiPatsTop,citeQuantCut(aiPats,1996,.90)]) 
        
        bioPatsTop = loadPatentAbstracts(bioPatsTop)
        bioPatsTop.to_csv(difPath+'/bio_patents.csv',encoding='utf-8')
        aiPatsTop = loadPatentAbstracts(aiPatsTop)
        aiPatsTop.to_csv(difPath+'/ai_patents.csv',encoding='utf-8')

    
if __name__=="__main__":
    main()



















    
