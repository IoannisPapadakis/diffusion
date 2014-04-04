# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import re
import csv
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# <headingcell level=1>

# # Describing Data

# <codecell>

# Merging AI abstract files
aiAbs = pd.DataFrame.from_csv('ai_abstracts.csv')
aiAbs = aiAbs[aiAbs.JRef!='None']
lgAbs = pd.DataFrame.from_csv('lg_abstracts.csv')
lgAbs = lgAbs[lgAbs.JRef!='None']
mlAbs = pd.DataFrame.from_csv('ml_abstracts.csv')
mlAbs = mlAbs[mlAbs.JRef!='None']

aimlAbs = pd.concat([aiAbs, lgAbs, mlAbs])
aimlAbs = aimlAbs.drop_duplicates('arxivID')
# Re-index to ensure that modifications based on index are applied to the correct rows
index = pd.Series(range(0,len(aimlAbs)))
aimlAbs.index=index

# Adding year from arXiv submit date to ai abstracts
aimlAbs['SubmitYear'] = int()
for row_index, row in aimlAbs.iterrows():
    aimlAbs.SubmitYear[row_index] = row['SubmitDate'][0:4]

# Adding year extracted from JRef to ai abstracts
findYear1 = re.compile('\(\d\d\d\d\)')
findYear2 = re.compile('(?<!\d\d\d\d\-)\d\d\d\d(?!\-\d\d\d\d)')
aimlAbs['JRefYear'] = int()
for row_index, row in aimlAbs.iterrows():
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
                aimlAbs.JRefYear[row_index] = int(year)
    find2 = re.findall(findYear2,papJRef)
    if find2 and not year:
        for i in find2:
            if int(i)<2015 and int(i)>1960:
                year = int(i)
                aimlAbs.JRefYear[row_index] = int(year)
# Drop records where the year could not be extracted from the JRef
aimlAbs = aimlAbs[aimlAbs.JRefYear!=0]
# Re-index to ensure that modifications based on index are applied to the correct rows
index = pd.Series(range(0,len(aimlAbs)))
aimlAbs.index=index

aimlAbs.to_csv('aiml_abstracts.csv')

# <codecell>

# Adding year to ai abstracts
aimlAbs['SubmitYear'] = int()
for row_index, row in aimlAbs.iterrows():
    aimlAbs.Year[row_index] = row['SubmitDate'][0:4]

findYear1 = re.compile('\(\d\d\d\d\)')
findYear2 = re.compile('(?<!\d\d\d\d\-)\d\d\d\d(?!\-\d\d\d\d)')
aimlAbs['JRefYear'] = int()
for row_index, row in aimlAbs.iterrows():
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
                aimlAbs.JRefYear[row_index] = int(year)
    find2 = re.findall(findYear2,papJRef)
    if find2 and not year:
        for i in find2:
            if int(i)<2015 and int(i)>1960:
                year = int(i)
                aimlAbs.JRefYear[row_index] = int(year)
aimlAbs.to_csv('aiml_abstracts.csv')

# <codecell>

bioAbs = pd.DataFrame.from_csv('bio_abstracts.csv')
findYear = re.compile('\(\d\d\d\d\)')
bioAbs['Year'] = int()
for row_index, row in bioAbs.iterrows():
    papJRef = row['JRef']
    find = re.search(findYear, papJRef)
    bioAbs.Year[row_index] = int(find.group()[1:5])
bioAbs.to_csv('bio_abstracts2.csv')

# <codecell>

bioAbs = pd.DataFrame.from_csv('bio_abstracts.csv')
aimlAbs = pd.DataFrame.from_csv('aiml_abstracts.csv')

aimlAbs['AbsLen'] = int()
for row_index, row in aimlAbs.iterrows():
    aimlAbs.AbsLen[row_index] = len(row['Abstract'])

findYear = re.compile('\(\d\d\d\d\)')
bioAbs['PapYear'] = int()
for row_index, row in bioAbs.iterrows():
    papJRef = row['JRef']
    find = re.search(findYear, papJRef)
    bioAbs.PapYear[row_index] = int(find.group()[1:5])
    
bioAbs['AbsLen'] = int()
for row_index, row in bioAbs.iterrows():
    bioAbs.AbsLen[row_index] = len(row['Abstract'])

# <codecell>

print aimlAbs.Year.describe()
print aimlAbs.AbsLen.describe()
print bioAbs.PapYear.describe()
print bioAbs.AbsLen.describe()

# <codecell>

aiPats = pd.DataFrame.from_csv('ai_patents.csv')
aiPats['AbsLen'] = int()
for row_index, row in aiPats.iterrows():
    aiPats.AbsLen[row_index] = len(row['Abstract'])
    
bioPats = pd.DataFrame.from_csv('bio_patents.csv')
bioPats['AbsLen'] = int()
for row_index, row in bioPats.iterrows():
    bioPats.AbsLen[row_index] = len(row['Abstract'])

# <codecell>

print aiPats.PatYear.describe()
print aiPats.AbsLen.describe()
print bioPats.PatYear.describe()
print bioPats.AbsLen.describe()

# <codecell>

# Load NBER Patent data
orig7606 = pd.read_stata('orig_gen_76_06.dta')
pat7606 = pd.DataFrame.from_csv('pat76_06_ipc.csv')

# <codecell>

# Put originality, generality, and citation measures onto the pat76-06 dataframe
pat7606 = pd.merge(pat7606, orig7606, on=['patent'])
# Since pat76-06 has multiple records for each patent = number of assigness, remove duplicates so there is one record per patent
pat7606 = pat7606.drop_duplicates(cols=['patent'])

# Merge NBER Patent columns onto bio patents data frame
bioPats.rename(columns={'PatentNum':'patent'}, inplace=True)
bioPats2 = pd.merge(bioPats,pat7606, on=['patent'])

# Merge NBER Patent columns onto ai patents data frame
aiPats.rename(columns={'PatentNum':'patent'}, inplace=True)
aiPats2 = pd.merge(aiPats,pat7606, on=['patent'])

# <codecell>

print aiPats2.ncited.describe()
print bioPats2.ncited.describe()

# <headingcell level=1>

# # Patent file analysis

# <codecell>

cat = 'bio'
#simDF = pd.DataFrame.from_csv('{0}_similarity.csv'.format(cat))
simDF = pd.DataFrame.from_csv('bio_simCosine.csv')
simDFPats = list(set(simDF.PatNum))
years = list(set(simDF.PapYear))
patYears = range(min(simDF.PatYear),max(simDF.PatYear)+1)
print patYears

# <codecell>

# Define dummy year variables
simDF['D_GT1993'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 1993:
        simDF.D_GT1993[row_index] = 1
    else:
        simDF.D_GT1993[row_index] = 0

simDF['D_GT1994'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 1994:
        simDF.D_GT1994[row_index] = 1
    else:
        simDF.D_GT1994[row_index] = 0

simDF['D_GT1995'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 1995:
        simDF.D_GT1995[row_index] = 1
    else:
        simDF.D_GT1995[row_index] = 0

simDF['D_GT1996'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 1996:
        simDF.D_GT1996[row_index] = 1
    else:
        simDF.D_GT1996[row_index] = 0

simDF['D_GT1997'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 1997:
        simDF.D_GT1997[row_index] = 1
    else:
        simDF.D_GT1997[row_index] = 0

simDF['D_GT1998'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 1998:
        simDF.D_GT1998[row_index] = 1
    else:
        simDF.D_GT1998[row_index] = 0

simDF['D_GT1999'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 1999:
        simDF.D_GT1999[row_index] = 1
    else:
        simDF.D_GT1999[row_index] = 0

simDF['D_GT2000'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2000:
        simDF.D_GT2000[row_index] = 1
    else:
        simDF.D_GT2000[row_index] = 0

        
simDF['D_GT2001'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2001:
        simDF.D_GT2001[row_index] = 1
    else:
        simDF.D_GT2001[row_index] = 0

simDF['D_GT2002'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2002:
        simDF.D_GT2002[row_index] = 1
    else:
        simDF.D_GT2002[row_index] = 0

simDF['D_GT2003'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2003:
        simDF.D_GT2003[row_index] = 1
    else:
        simDF.D_GT2003[row_index] = 0

simDF['D_GT2004'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2004:
        simDF.D_GT2004[row_index] = 1
    else:
        simDF.D_GT2004[row_index] = 0

simDF['D_GT2005'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2005:
        simDF.D_GT2005[row_index] = 1
    else:
        simDF.D_GT2005[row_index] = 0

simDF['D_GT2006'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2006:
        simDF.D_GT2006[row_index] = 1
    else:
        simDF.D_GT2006[row_index] = 0

simDF['D_GT2007'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2007:
        simDF.D_GT2007[row_index] = 1
    else:
        simDF.D_GT2007[row_index] = 0

simDF['D_GT2008'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2008:
        simDF.D_GT2008[row_index] = 1
    else:
        simDF.D_GT2008[row_index] = 0

simDF['D_GT2009'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2009:
        simDF.D_GT2009[row_index] = 1
    else:
        simDF.D_GT2009[row_index] = 0
        
simDF['D_GT2010'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2010:
        simDF.D_GT2010[row_index] = 1
    else:
        simDF.D_GT2010[row_index] = 0
        
simDF['D_GT2011'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2011:
        simDF.D_GT2011[row_index] = 1
    else:
        simDF.D_GT2011[row_index] = 0
        
simDF['D_GT2012'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2012:
        simDF.D_GT2012[row_index] = 1
    else:
        simDF.D_GT2012[row_index] = 0

simDF['D_GT2013'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2013:
        simDF.D_GT2013[row_index] = 1
    else:
        simDF.D_GT2013[row_index] = 0

simDF['D_GT2014'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] >= 2014:
        simDF.D_GT2014[row_index] = 1
    else:
        simDF.D_GT2014[row_index] = 0

# <codecell>

# Define dummy year variables
simDF['D_E1993'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 1993:
        simDF.D_E1993[row_index] = 1
    else:
        simDF.D_E1993[row_index] = 0

simDF['D_E1994'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 1994:
        simDF.D_E1994[row_index] = 1
    else:
        simDF.D_E1994[row_index] = 0

simDF['D_E1995'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 1995:
        simDF.D_E1995[row_index] = 1
    else:
        simDF.D_E1995[row_index] = 0

simDF['D_E1996'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 1996:
        simDF.D_E1996[row_index] = 1
    else:
        simDF.D_E1996[row_index] = 0

simDF['D_E1997'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 1997:
        simDF.D_E1997[row_index] = 1
    else:
        simDF.D_E1997[row_index] = 0

simDF['D_E1998'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 1998:
        simDF.D_E1998[row_index] = 1
    else:
        simDF.D_E1998[row_index] = 0

simDF['D_E1999'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 1999:
        simDF.D_E1999[row_index] = 1
    else:
        simDF.D_E1999[row_index] = 0

simDF['D_E2000'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2000:
        simDF.D_E2000[row_index] = 1
    else:
        simDF.D_E2000[row_index] = 0

        
simDF['D_E2001'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2001:
        simDF.D_E2001[row_index] = 1
    else:
        simDF.D_E2001[row_index] = 0

simDF['D_E2002'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2002:
        simDF.D_E2002[row_index] = 1
    else:
        simDF.D_E2002[row_index] = 0

simDF['D_E2003'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2003:
        simDF.D_E2003[row_index] = 1
    else:
        simDF.D_E2003[row_index] = 0

simDF['D_E2004'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2004:
        simDF.D_E2004[row_index] = 1
    else:
        simDF.D_E2004[row_index] = 0

simDF['D_E2005'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2005:
        simDF.D_E2005[row_index] = 1
    else:
        simDF.D_E2005[row_index] = 0

simDF['D_E2006'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2006:
        simDF.D_E2006[row_index] = 1
    else:
        simDF.D_E2006[row_index] = 0

simDF['D_E2007'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2007:
        simDF.D_E2007[row_index] = 1
    else:
        simDF.D_E2007[row_index] = 0

simDF['D_E2008'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2008:
        simDF.D_E2008[row_index] = 1
    else:
        simDF.D_E2008[row_index] = 0

simDF['D_E2009'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2009:
        simDF.D_E2009[row_index] = 1
    else:
        simDF.D_E2009[row_index] = 0
        
simDF['D_E2010'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2010:
        simDF.D_E2010[row_index] = 1
    else:
        simDF.D_E2010[row_index] = 0
        
simDF['D_E2011'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2011:
        simDF.D_E2011[row_index] = 1
    else:
        simDF.D_E2011[row_index] = 0
        
simDF['D_E2012'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2012:
        simDF.D_E2012[row_index] = 1
    else:
        simDF.D_E2012[row_index] = 0

simDF['D_E2013'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2013:
        simDF.D_E2013[row_index] = 1
    else:
        simDF.D_E2013[row_index] = 0

simDF['D_E2014'] = int()
for row_index, row in simDF.iterrows():
    if row['PapYear'] == 2014:
        simDF.D_E2014[row_index] = 1
    else:
        simDF.D_E2014[row_index] = 0

# <codecell>

simDF['RelYear'] = int()
for row_index,row in simDF.iterrows():
    simDF.RelYear[row_index] = row['PapYear'] - row['PatYear']

# <codecell>

# Descriptive Statistics
print "cat", cat
print "N Before", len(simDF[simDF.RelYear<=0].Similarity)
print "Mean Before", simDF[simDF.RelYear<=0].Similarity.mean()
print "Std Before", simDF[simDF.RelYear<=0].Similarity.std()
print "Min Before", simDF[simDF.RelYear<=0].Similarity.min()
print "Max Before", simDF[simDF.RelYear<=0].Similarity.max()
print "\n"
print "N After", len(simDF[simDF.RelYear>0].Similarity)
print "Mean After", simDF[simDF.RelYear>0].Similarity.mean()
print "Std After", simDF[simDF.RelYear>0].Similarity.std()
print "Min After", simDF[simDF.RelYear>0].Similarity.min()
print "Max After", simDF[simDF.RelYear>0].Similarity.max()

# <codecell>

# Spaghetti plot of mean similarity by year
fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
#ax1.set_ylim([0.1, 0.2])

for pat in simDFPats: 
    ax1.plot(years, simDF[simDF.PatNum==pat].groupby(['PapYear']).Similarity.mean())
    if cat == 'bio':
        ax1.set_title('Biotechnology: Mean Similarity by Year All Patents') 
    elif cat == 'ai':
        ax1.set_title('Artificial Intelligence: Mean Similarity by Year All Patents') 
    ax1.set_ylabel('Similarity')
    ax1.set_xlabel('Year')
plt.show()

# <codecell>

# Analyzing a single patent
fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1.plot(years, simDF[simDF.PatNum==simDFPats[0]].groupby(['PapYear']).Similarity.mean())
ax1.set_ylabel('')
ax1.set_xlabel('Year')

y = list(simDF[simDF.PatNum==simDFPats[0]].groupby(['PapYear']).Similarity.mean())
x1 = list(simDF[simDF.PatNum==simDFPats[0]].groupby(['PapYear']).D1999.mean())
x2 = list(set(simDF.PapYear))

slope, intercept, r_value, p_value, std_err = stats.linregress(x1,y)

print 'slope', slope
print 'intercept', intercept
print 'r value', r_value
print  'p_value', p_value
print 'standard deviation', std_err

# <codecell>

patYears = range(min(simDF.PatYear),max(simDF.PatYear)+1)
print patYears

for yr in patYears:
    x1Coef = []
    x1Pval = []
    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(111)

    yrPats = list(set(simDF[simDF.PatYear==yr].PatNum))
    for pat in yrPats:
        y = list(simDF[simDF.PatNum==pat].groupby(['PapYear']).Similarity.mean())
        if yr ==1996:
            x1 = list(simDF[simDF.PatNum==pat].groupby(['PapYear']).D1996.mean())
        elif yr == 1997:
            x1 = list(simDF[simDF.PatNum==pat].groupby(['PapYear']).D1997.mean())
        elif yr == 1998:
            x1 = list(simDF[simDF.PatNum==pat].groupby(['PapYear']).D1998.mean())
        elif yr == 1999:
            x1 = list(simDF[simDF.PatNum==pat].groupby(['PapYear']).D1999.mean())
        X = sm.add_constant(zip(x1), prepend=True)
        results = sm.OLS(y, X).fit()
    
        x1Coef.append(round(results.params[1],4))
        x1Pval.append(round(results.pvalues[1],4))
    if cat == 'bio':
        ax1.set_title('{0} Biotechnology Patents: \n {0} Dummy Regression Coefs and PVals'.format(yr)) 
    elif cat == 'ai':
        ax1.set_title('{0} Artificial Intelligence Patents: \n {0} Dummy Regression Coefs and PVals'.format(yr)) 
    ax1.scatter(x1Pval,x1Coef)
    ax1.set_ylabel('Coefficient')
    ax1.set_xlabel('P-Value')
    plt.show()

# <codecell>

y = list(simDF[simDF.PatNum==simDFPats[0]].groupby(['PapYear']).Similarity.mean())
x1 = list(simDF.groupby(['PapYear']).D1999.mean())
X = sm.add_constant(zip(x1), prepend=True)
results = sm.OLS(y, X).fit()
x1Coef.append(round(results.params[1],5))
x1Pval.append(round(results.pvalues[1],5))

patYear = simDF[simDF.PatNum==simDFPats[0]].PatYear.mean()
CCoef = round(results.params[0],5)
CPVal = round(results.pvalues[0],5)
DCoef = round(results.params[1],5)
DPVal = round(results.pvalues[1],5)
r2 = round(results.rsquared,5)
probF = round(results.f_pvalue,5)
dfRes = round(results.df_resid,5)
print CCoef
print CPVal
print DCoef
print DPVal
print r2
print probF
print dfRes
print results.summary()

# <codecell>

# Run similarity regressions using >= dummies
simRegs = pd.DataFrame(columns=['PatNum', 'PatYear', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
def doMeanSimReg(x1,y,patent,dumYear,patYear,df):
            
    X = sm.add_constant(zip(x1), prepend=True)
    results = sm.OLS(y, X).fit()
    
    patNum = patent
    patYear = patYear
    relDumYear = dumYear - patYear
    CCoef = round(results.params[0],5)
    CPVal = round(results.pvalues[0],5)
    DCoef = round(results.params[1],5)
    DPVal = round(results.pvalues[1],5)
    r2 = round(results.rsquared,5)
    probF = round(results.f_pvalue,5)
    dfRes = round(results.df_resid,5)

    row = pd.Series([patNum,patYear,dumYear,relDumYear,CCoef,CPVal,DCoef,DPVal,r2,probF,dfRes],index=['PatNum', 'PatYear', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
    df = df.append(row, ignore_index=True)
    return df


for pat in simDFPats:
    y = list(simDF[simDF.PatNum==pat].groupby(['PapYear']).Similarity.mean())
    patYear = patYear = simDF[simDF.PatNum==pat].PatYear.mean()
    if cat=='bio':
        # AI papers do not go back to 1993
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1993.mean()),y,pat,1993,patYear,simRegs)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1994.mean()),y,pat,1994,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1995.mean()),y,pat,1995,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1996.mean()),y,pat,1996,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1997.mean()),y,pat,1997,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1998.mean()),y,pat,1998,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1999.mean()),y,pat,1999,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2000.mean()),y,pat,2000,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2001.mean()),y,pat,2001,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2002.mean()),y,pat,2002,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2003.mean()),y,pat,2003,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2004.mean()),y,pat,2004,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2005.mean()),y,pat,2005,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2006.mean()),y,pat,2006,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2007.mean()),y,pat,2007,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2008.mean()),y,pat,2008,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2009.mean()),y,pat,2009,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2010.mean()),y,pat,2010,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2011.mean()),y,pat,2011,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2012.mean()),y,pat,2012,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2013.mean()),y,pat,2013,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2014.mean()),y,pat,2014,patYear,simRegs)
    
if cat=='bio':
    simRegs.to_csv('bio_simRegRaw.csv')
elif cat=='ai':
    simRegs.to_csv('ai_simRegRaw.csv')
    
print simRegs
minRelYear = int(simRegs.RelDumYear.min())
maxRelYear = int(simRegs.RelDumYear.max())

simRelMeans = pd.DataFrame(pd.Series(range(minRelYear,maxRelYear+1)), columns=['RelDumYear'])

simRelMeans['CCoef'] = pd.Series(simRegs.groupby(['RelDumYear']).CCoef.mean().values)
simRelMeans['CPVal'] = pd.Series(simRegs.groupby(['RelDumYear']).CPVal.mean().values)
simRelMeans['DCoef'] = pd.Series(simRegs.groupby(['RelDumYear']).DCoef.mean().values)
simRelMeans['DPVal'] = pd.Series(simRegs.groupby(['RelDumYear']).DPVal.mean().values)
simRelMeans['R2'] = pd.Series(simRegs.groupby(['R2']).R2.mean().values)
simRelMeans['Pfstat'] = pd.Series(simRegs.groupby(['RelDumYear']).Pfstat.mean().values)
simRelMeans['DFResid'] = pd.Series(simRegs.groupby(['RelDumYear']).DFResid.mean().values)

print simRelMeans
if cat=='bio':
    simRelMeans.to_csv('bio_simRelMeans.csv')
elif cat=='ai':
    simRelMeans.to_csv('ai_simRelMeans.csv')


relDumYears = list(simRelMeans.RelDumYear)

meanPval = []
meanCoef = []

for relYr in relDumYears:
    meanPval.append(simRegs[simRegs.RelDumYear==relYr].DPVal.mean())
    meanCoef.append(simRegs[simRegs.RelDumYear==relYr].DCoef.mean())

# <codecell>

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
for i in range(len(meanPval)):
    if meanPval[i]<=0.1:
        ax1.scatter(relDumYears[i],meanPval[i], marker='o', c='b',s=40)
    else:
        ax1.scatter(relDumYears[i],meanPval[i], marker='x', c='r',s=50)
#ax1.scatter(relDumYears,meanPval)
ax1.set_ylabel('Mean P-Value')
ax1.set_xlabel('Dummy Year Relative to Publication')
if cat == 'bio':
    ax1.set_title('Biotechnology Patent Dummy Regression P-Values\nMean OLS P-Values by Relative Dummy Year')
elif cat == 'ai':
    ax1.set_title('Artificial Intelligence Patent Dummy Regression P-Values\nMean OLS P-Values by Relative Dummy Year')
ax1.set_ylim(0,1)
ax1.axvline(x=0, color='r', ls='--', lw=2)
ax1.text(1,0.7,'Year Granted',fontsize=12,)
ax1.grid()

ax2 = fig.add_subplot(122)
for i in range(len(meanCoef)):
    if meanPval[i]<=0.1:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='o', c='b',s=40)
    else:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='x', c='r',s=50)
#ax2.scatter(relDumYears,meanCoef)
ax2.set_ylabel('Mean Coefficient')
ax2.set_xlabel('Dummy Year Relative to Publication')
if cat=='bio':
    ax2.set_title('Biotechnology Patent Dummy Regression Coefficients\nMean OLS Coefficients by Relative Dummy Year')
elif cat=='ai':
    ax2.set_title('Artificial Intelligence Patent Dummy Regression Coefficients\nMean OLS Coefficients by Relative Dummy Year')
ax2.axvline(x=0, color='r', ls='--', lw=2)
if cat=='bio':
    ax2.text(1,-0.003,'Year Granted',fontsize=12,)
elif cat=='ai':
    ax2.text(1,0.003,'Year Granted',fontsize=12,)
ax2.grid()
#plt.savefig('pat_diffusion.png')
plt.show()


# <codecell>

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
ax1.hist(simDF[simDF.RelYear<=0].Similarity,bins=50)
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Similarity')
if cat=='bio':
    ax1.set_title('Histogram Biotechnology Patent Similarity\nSimilarity Measures Before Patent Publication')
elif cat=='ai':
    ax1.set_title('Histogram of Artificial Intelligence Patent Similarity\nSimilarity Measures Before Patent Publication')
meanBefore = simDF[simDF.RelYear<=0].Similarity.mean()
ax1.axvline(x=meanBefore, color='r', ls='-', lw=2)
ax1.set_xlim(0,simDF.Similarity.max()*1.1)
ax1.grid()

ax2 = fig.add_subplot(122)
ax2.hist(simDF[simDF.RelYear>0].Similarity,bins=50)
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Similarity')
if cat=='bio':
    ax2.set_title('Histogram Biotechnology Patent Similarity\nSimilarity Measures After Patent Publication')
elif cat=='ai':
    ax2.set_title('Histogram of Artificial Intelligence Patent Similarity\nSimilarity Measures After Patent Publication')
meanAfter = simDF[simDF.RelYear>0].Similarity.mean()
ax2.axvline(x=meanAfter, color='r', ls='-', lw=2)
ax2.set_xlim(0,simDF.Similarity.max()*1.1)
ax2.grid()
plt.show()

# <codecell>

# Regression for top %10 of similarity
simRegs = pd.DataFrame(columns=['PatNum', 'PatYear', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
def doMeanSimReg(x1,patent,dumYear,df):
    # Find top %x by similarity
    y = []
    for yr in range(simDF.PapYear.min(),simDF.PapYear.max()+1):
        quantCut = simDF[(simDF.PatNum==pat)&(simDF.PapYear==yr)].Similarity.quantile(.90)
        y.append(simDF[(simDF.PatNum==pat)&(simDF.PapYear==yr)&(simDF.Similarity>=quantCut)].Similarity.mean())
        
    X = sm.add_constant(zip(x1), prepend=True)
    results = sm.OLS(y, X).fit()
    
    patNum = patent
    patYear = simDF[simDF.PatNum==pat].PatYear.mean()
    relDumYear = dumYear - patYear
    CCoef = round(results.params[0],5)
    CPVal = round(results.pvalues[0],5)
    DCoef = round(results.params[1],5)
    DPVal = round(results.pvalues[1],5)
    r2 = round(results.rsquared,5)
    probF = round(results.f_pvalue,5)
    dfRes = round(results.df_resid,5)

    row = pd.Series([patNum,patYear,dumYear,relDumYear,CCoef,CPVal,DCoef,DPVal,r2,probF,dfRes],index=['PatNum', 'PatYear', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
    df = df.append(row, ignore_index=True)
    return df


for pat in simDFPats:
    
    if cat=='bio':
        # AI papers do not go back to 1993
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1993.mean()),pat,1993,simRegs)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1994.mean()),pat,1994,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1995.mean()),pat,1995,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1996.mean()),pat,1996,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1997.mean()),pat,1997,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1998.mean()),pat,1998,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1999.mean()),pat,1999,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2000.mean()),pat,2000,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2001.mean()),pat,2001,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2002.mean()),pat,2002,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2003.mean()),pat,2003,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2004.mean()),pat,2004,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2005.mean()),pat,2005,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2006.mean()),pat,2006,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2007.mean()),pat,2007,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2008.mean()),pat,2008,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2009.mean()),pat,2009,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2010.mean()),pat,2010,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2011.mean()),pat,2011,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2012.mean()),pat,2012,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2013.mean()),pat,2013,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2014.mean()),pat,2014,simRegs)
    
print simRegs
minRelYear = int(simRegs.RelDumYear.min())
maxRelYear = int(simRegs.RelDumYear.max())

simRelMeans = pd.DataFrame(pd.Series(range(minRelYear,maxRelYear+1)), columns=['RelDumYear'])

simRelMeans['CCoef'] = pd.Series(simRegs.groupby(['RelDumYear']).CCoef.mean().values)
simRelMeans['CPVal'] = pd.Series(simRegs.groupby(['RelDumYear']).CPVal.mean().values)
simRelMeans['DCoef'] = pd.Series(simRegs.groupby(['RelDumYear']).DCoef.mean().values)
simRelMeans['DPVal'] = pd.Series(simRegs.groupby(['RelDumYear']).DPVal.mean().values)
simRelMeans['R2'] = pd.Series(simRegs.groupby(['R2']).R2.mean().values)
simRelMeans['Pfstat'] = pd.Series(simRegs.groupby(['RelDumYear']).Pfstat.mean().values)
simRelMeans['DFResid'] = pd.Series(simRegs.groupby(['RelDumYear']).DFResid.mean().values)

print simRelMeans

relDumYears = range(int(min(simRegs.RelDumYear)),int(max(simRegs.RelDumYear))+1)

meanPval = []
meanCoef = []

for relYr in relDumYears:
    meanPval.append(simRegs[simRegs.RelDumYear==relYr].DPVal.mean())
    meanCoef.append(simRegs[simRegs.RelDumYear==relYr].DCoef.mean())

# <codecell>

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
for i in range(len(meanPval)):
    if meanPval[i]<=0.1:
        ax1.scatter(relDumYears[i],meanPval[i], marker='o', c='b',s=40)
    else:
        ax1.scatter(relDumYears[i],meanPval[i], marker='x', c='r',s=50)
#ax1.scatter(relDumYears,meanPval)
ax1.set_ylabel('Mean P-Value')
ax1.set_xlabel('Dummy Year Relative to Publication')
if cat == 'bio':
    ax1.set_title('Biotechnology Patent Dummy Regression P-Values\nDependent Variable Mean of Top 10% Similarity\nMean OLS P-Values by Relative Dummy Year')
elif cat == 'ai':
    ax1.set_title('Artificial Intelligence Patent Dummy Regression P-Values\nDependent Variable Mean of Top 10% Similarity\nMean OLS P-Values by Relative Dummy Year')
ax1.set_ylim(0,1)
ax1.axvline(x=0, color='r', ls='--', lw=2)
ax1.text(1,0.7,'Year Granted',fontsize=12,)
ax1.grid()

ax2 = fig.add_subplot(122)
for i in range(len(meanCoef)):
    if meanPval[i]<=0.1:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='o', c='b',s=40)
    else:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='x', c='r',s=50)
#ax2.scatter(relDumYears,meanCoef)
ax2.set_ylabel('Mean Coefficient')
ax2.set_xlabel('Dummy Year Relative to Publication')
if cat=='bio':
    ax2.set_title('Biotechnology Patent Dummy Regression Coefficients\nDependent Variable Mean of Top 10% Similarity\nMean OLS Coefficients by Relative Dummy Year')
elif cat=='ai':
    ax2.set_title('Artificial Intelligence Patent Dummy Regression Coefficients\nDependent Variable Mean of Top 10% Similarity\nMean OLS Coefficients by Relative Dummy Year')
ax2.axvline(x=0, color='r', ls='--', lw=2)
if cat=='bio':
    ax2.text(1,-0.003,'Year Granted',fontsize=12,)
elif cat=='ai':
    ax2.text(1,0.003,'Year Granted',fontsize=12,)
ax2.grid()
#plt.savefig('pat_diffusion.png')

# <codecell>

# Run similarity regressions using == dummies
simRegs = pd.DataFrame(columns=['PatNum', 'PatYear', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
def doMeanSimReg(x1,y,patent,dumYear,patYear,df):
            
    X = sm.add_constant(zip(x1), prepend=True)
    results = sm.OLS(y, X).fit()
    
    patNum = patent
    patYear = patYear
    relDumYear = dumYear - patYear
    CCoef = round(results.params[0],5)
    CPVal = round(results.pvalues[0],5)
    DCoef = round(results.params[1],5)
    DPVal = round(results.pvalues[1],5)
    r2 = round(results.rsquared,5)
    probF = round(results.f_pvalue,5)
    dfRes = round(results.df_resid,5)

    row = pd.Series([patNum,patYear,dumYear,relDumYear,CCoef,CPVal,DCoef,DPVal,r2,probF,dfRes],index=['PatNum', 'PatYear', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
    df = df.append(row, ignore_index=True)
    return df


for pat in simDFPats:
    y = list(simDF[simDF.PatNum==pat].groupby(['PapYear']).Similarity.mean())
    patYear = patYear = simDF[simDF.PatNum==pat].PatYear.mean()
    if cat=='bio':
        # AI papers do not go back to 1993
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E1993.mean()),y,pat,1993,patYear,simRegs)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E1994.mean()),y,pat,1994,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E1995.mean()),y,pat,1995,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E1996.mean()),y,pat,1996,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E1997.mean()),y,pat,1997,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E1998.mean()),y,pat,1998,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E1999.mean()),y,pat,1999,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2000.mean()),y,pat,2000,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2001.mean()),y,pat,2001,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2002.mean()),y,pat,2002,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2003.mean()),y,pat,2003,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2004.mean()),y,pat,2004,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2005.mean()),y,pat,2005,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2006.mean()),y,pat,2006,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2007.mean()),y,pat,2007,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2008.mean()),y,pat,2008,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2009.mean()),y,pat,2009,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2010.mean()),y,pat,2010,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2011.mean()),y,pat,2011,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2012.mean()),y,pat,2012,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2013.mean()),y,pat,2013,patYear,simRegs)
    simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_E2014.mean()),y,pat,2014,patYear,simRegs)
    
if cat=='bio':
    simRegs.to_csv('bio_simRegRaw.csv')
elif cat=='ai':
    simRegs.to_csv('ai_simRegRaw.csv')
    
print simRegs
minRelYear = int(simRegs.RelDumYear.min())
maxRelYear = int(simRegs.RelDumYear.max())

simRelMeans = pd.DataFrame(pd.Series(range(minRelYear,maxRelYear+1)), columns=['RelDumYear'])

simRelMeans['CCoef'] = pd.Series(simRegs.groupby(['RelDumYear']).CCoef.mean().values)
simRelMeans['CPVal'] = pd.Series(simRegs.groupby(['RelDumYear']).CPVal.mean().values)
simRelMeans['DCoef'] = pd.Series(simRegs.groupby(['RelDumYear']).DCoef.mean().values)
simRelMeans['DPVal'] = pd.Series(simRegs.groupby(['RelDumYear']).DPVal.mean().values)
simRelMeans['R2'] = pd.Series(simRegs.groupby(['R2']).R2.mean().values)
simRelMeans['Pfstat'] = pd.Series(simRegs.groupby(['RelDumYear']).Pfstat.mean().values)
simRelMeans['DFResid'] = pd.Series(simRegs.groupby(['RelDumYear']).DFResid.mean().values)

print simRelMeans
if cat=='bio':
    simRelMeans.to_csv('bio_simRelMeans.csv')
elif cat=='ai':
    simRelMeans.to_csv('ai_simRelMeans.csv')


relDumYears = list(simRelMeans.RelDumYear)

meanPval = []
meanCoef = []

for relYr in relDumYears:
    meanPval.append(simRegs[simRegs.RelDumYear==relYr].DPVal.mean())
    meanCoef.append(simRegs[simRegs.RelDumYear==relYr].DCoef.mean())

# <codecell>

# Plot results for == dummies
fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
for i in range(len(meanPval)):
    if meanPval[i]<=0.1:
        ax1.scatter(relDumYears[i],meanPval[i], marker='o', c='b',s=40)
    else:
        ax1.scatter(relDumYears[i],meanPval[i], marker='x', c='r',s=50)
#ax1.scatter(relDumYears,meanPval)
ax1.set_ylabel('Mean P-Value')
ax1.set_xlabel('Dummy Year Relative to Publication')
if cat == 'bio':
    ax1.set_title('Biotechnology Patent Dummy Regression P-Values\nMean OLS P-Values by Relative Dummy Year')
elif cat == 'ai':
    ax1.set_title('Artificial Intelligence Patent Dummy Regression P-Values\nMean OLS P-Values by Relative Dummy Year')
ax1.set_ylim(0,1)
ax1.axvline(x=0, color='r', ls='--', lw=2)
ax1.text(1,0.7,'Year Granted',fontsize=12,)
ax1.grid()

ax2 = fig.add_subplot(122)
for i in range(len(meanCoef)):
    if meanPval[i]<=0.1:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='o', c='b',s=40)
    else:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='x', c='r',s=50)
#ax2.scatter(relDumYears,meanCoef)
ax2.set_ylabel('Mean Coefficient')
ax2.set_xlabel('Dummy Year Relative to Publication')
if cat=='bio':
    ax2.set_title('Biotechnology Patent Dummy Regression Coefficients\nMean OLS Coefficients by Relative Dummy Year')
elif cat=='ai':
    ax2.set_title('Artificial Intelligence Patent Dummy Regression Coefficients\nMean OLS Coefficients by Relative Dummy Year')
ax2.axvline(x=0, color='r', ls='--', lw=2)
if cat=='bio':
    ax2.text(1,-0.003,'Year Granted',fontsize=12,)
elif cat=='ai':
    ax2.text(1,0.003,'Year Granted',fontsize=12,)
ax2.grid()
#plt.savefig('pat_diffusion.png')
plt.show()

# <headingcell level=1>

# Analyze Paper Similarity Data

# <codecell>

#papSim = pd.DataFrame.from_csv('paper_similarity.csv')
papSim = pd.DataFrame.from_csv('paper_simCosine.csv')
print papSim
papers = list(set(papSim.TopPaperTitle))

# <codecell>

# Create dummy variables on paper similarity dataframe with >=
papSim['D_GT1994'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 1994:
        papSim.D_GT1994[row_index] = 1
    else:
        papSim.D_GT1994[row_index] = 0

papSim['D_GT1995'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 1995:
        papSim.D_GT1995[row_index] = 1
    else:
        papSim.D_GT1995[row_index] = 0

papSim['D_GT1996'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 1996:
        papSim.D_GT1996[row_index] = 1
    else:
        papSim.D_GT1996[row_index] = 0

papSim['D_GT1997'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 1997:
        papSim.D_GT1997[row_index] = 1
    else:
        papSim.D_GT1997[row_index] = 0

papSim['D_GT1998'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 1998:
        papSim.D_GT1998[row_index] = 1
    else:
        papSim.D_GT1998[row_index] = 0

papSim['D_GT1999'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 1999:
        papSim.D_GT1999[row_index] = 1
    else:
        papSim.D_GT1999[row_index] = 0

papSim['D_GT2000'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2000:
        papSim.D_GT2000[row_index] = 1
    else:
        papSim.D_GT2000[row_index] = 0

        
papSim['D_GT2001'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2001:
        papSim.D_GT2001[row_index] = 1
    else:
        papSim.D_GT2001[row_index] = 0

papSim['D_GT2002'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2002:
        papSim.D_GT2002[row_index] = 1
    else:
        papSim.D_GT2002[row_index] = 0

papSim['D_GT2003'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2003:
        papSim.D_GT2003[row_index] = 1
    else:
        papSim.D_GT2003[row_index] = 0

papSim['D_GT2004'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2004:
        papSim.D_GT2004[row_index] = 1
    else:
        papSim.D_GT2004[row_index] = 0

papSim['D_GT2005'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2005:
        papSim.D_GT2005[row_index] = 1
    else:
        papSim.D_GT2005[row_index] = 0

papSim['D_GT2006'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2006:
        papSim.D_GT2006[row_index] = 1
    else:
        papSim.D_GT2006[row_index] = 0

papSim['D_GT2007'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2007:
        papSim.D_GT2007[row_index] = 1
    else:
        papSim.D_GT2007[row_index] = 0

papSim['D_GT2008'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2008:
        papSim.D_GT2008[row_index] = 1
    else:
        papSim.D_GT2008[row_index] = 0

papSim['D_GT2009'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2009:
        papSim.D_GT2009[row_index] = 1
    else:
        papSim.D_GT2009[row_index] = 0
        
papSim['D_GT2010'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2010:
        papSim.D_GT2010[row_index] = 1
    else:
        papSim.D_GT2010[row_index] = 0
        
papSim['D_GT2011'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2011:
        papSim.D_GT2011[row_index] = 1
    else:
        papSim.D_GT2011[row_index] = 0
        
papSim['D_GT2012'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2012:
        papSim.D_GT2012[row_index] = 1
    else:
        papSim.D_GT2012[row_index] = 0

papSim['D_GT2013'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2013:
        papSim.D_GT2013[row_index] = 1
    else:
        papSim.D_GT2013[row_index] = 0

papSim['D_GT2014'] = int()
for row_index, row in papSim.iterrows():
    if row['PapJRefYear'] >= 2014:
        papSim.D_GT2014[row_index] = 1
    else:
        papSim.D_GT2014[row_index] = 0

papSim['RelYear'] = int()
for row_index,row in papSim.iterrows():
    papSim.RelYear[row_index] = row['PapJRefYear'] - row['TopPaperYear']

# <codecell>

# Create dummy variables on paper similarity dataframe with ==
papSim['D_E1994'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 1994:
        papSim.D_E1994[row_index] = 1
    else:
        papSim.D_E1994[row_index] = 0

papSim['D_E1995'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 1995:
        papSim.D_E1995[row_index] = 1
    else:
        papSim.D_E1995[row_index] = 0

papSim['D_E1996'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 1996:
        papSim.D_E1996[row_index] = 1
    else:
        papSim.D_E1996[row_index] = 0

papSim['D_E1997'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 1997:
        papSim.D_E1997[row_index] = 1
    else:
        papSim.D_E1997[row_index] = 0

papSim['D_E1998'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 1998:
        papSim.D_E1998[row_index] = 1
    else:
        papSim.D_E1998[row_index] = 0

papSim['D_E1999'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 1999:
        papSim.D_E1999[row_index] = 1
    else:
        papSim.D_E1999[row_index] = 0

papSim['D_E2000'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2000:
        papSim.D_E2000[row_index] = 1
    else:
        papSim.D_E2000[row_index] = 0

        
papSim['D_E2001'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2001:
        papSim.D_E2001[row_index] = 1
    else:
        papSim.D_E2001[row_index] = 0

papSim['D_E2002'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2002:
        papSim.D_E2002[row_index] = 1
    else:
        papSim.D_E2002[row_index] = 0

papSim['D_E2003'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2003:
        papSim.D_E2003[row_index] = 1
    else:
        papSim.D_E2003[row_index] = 0

papSim['D_E2004'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2004:
        papSim.D_E2004[row_index] = 1
    else:
        papSim.D_E2004[row_index] = 0

papSim['D_E2005'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2005:
        papSim.D_E2005[row_index] = 1
    else:
        papSim.D_E2005[row_index] = 0

papSim['D_E2006'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2006:
        papSim.D_E2006[row_index] = 1
    else:
        papSim.D_E2006[row_index] = 0

papSim['D_E2007'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2007:
        papSim.D_E2007[row_index] = 1
    else:
        papSim.D_E2007[row_index] = 0

papSim['D_E2008'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2008:
        papSim.D_E2008[row_index] = 1
    else:
        papSim.D_E2008[row_index] = 0

papSim['D_E2009'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2009:
        papSim.D_E2009[row_index] = 1
    else:
        papSim.D_E2009[row_index] = 0
        
papSim['D_E2010'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2010:
        papSim.D_E2010[row_index] = 1
    else:
        papSim.D_E2010[row_index] = 0
        
papSim['D_E2011'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2011:
        papSim.D_E2011[row_index] = 1
    else:
        papSim.D_E2011[row_index] = 0
        
papSim['D_E2012'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2012:
        papSim.D_E2012[row_index] = 1
    else:
        papSim.D_E2012[row_index] = 0

papSim['D_E2013'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2013:
        papSim.D_E2013[row_index] = 1
    else:
        papSim.D_E2013[row_index] = 0

papSim['D_E2014'] = int()
for row_index, row in papSim.iterrows():
    if row['PapYear'] == 2014:
        papSim.D_E2014[row_index] = 1
    else:
        papSim.D_E2014[row_index] = 0

papSim['RelYear'] = int()
for row_index,row in papSim.iterrows():
    papSim.RelYear[row_index] = row['PapYear'] - row['TopPaperYear']

# <codecell>

# Plot mean similarity, count of papers, and 95th percentile of similarity by year for each paper
for paper in papers:
    print paper
    papSimTmp = papSim[papSim.TopPaperTitle==paper]
    
    tmpGBYear = list(papSimTmp.groupby(['PapYear']).Similarity.values)
    tmpYearMean = []
    tmpYearCount = []
    tmpPercentile = []
    for i in tmpGBYear:
        tmpYearMean.append(np.mean(i))
        tmpYearCount.append(len(i))
        tmpPercentile.append(np.percentile(i,95))
    years = list(set(papSimTmp.PapYear))
    
    fig = plt.figure(figsize=(18,5))
    ax1 = fig.add_subplot(131)
    ax1.scatter(years, tmpYearMean)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mean Similarity')
    ax1.set_title('Mean Abstract Similairty to Focus Paper by Year')
    
    
    ax2 = fig.add_subplot(132)
    ax2.scatter(years, tmpYearCount)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Abstracts')
    ax2.set_title('Count of Machine Learning Abstracts by Year')
    
    
    ax3 = fig.add_subplot(133)
    ax3.scatter(years, tmpPercentile)
    ax3.set_title('Similarity 95th Percentile')
    plt.show()

# <codecell>

# Flagging the papers that visually show an effect
papers = ['Learning and Inferring Transportation Routines', 'Text Classification from Labeled and Unlabeled Documents using EM', 'FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem', 'Random Forests', 'Unsupervised Learning by Probabilistic Latent Semantic Analysis', 'Gene Selection for Cancer Classification using Support Vector Machines', 'Choosing Multiple Parameters for Support Vector Machines', 'Matching words and pictures']
papSim['flg'] = int()
for row_index, row in papSim.iterrows():
    if row['TopPaperTitle'] in papers:
        papSim.flg[row_index] = 1
    else:
        papSim.flg[row_index] = 0

# <codecell>

simRegs = pd.DataFrame(columns=['Title', 'Year', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
simRegs[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']] = simRegs[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']].apply(np.float32)
def doMeanSimReg(x1,y,dumYear,title,papYear,df):
    
    X = sm.add_constant(zip(x1), prepend=True)
    results = sm.OLS(y, X).fit()
    
    year = papYear
    relDumYear = dumYear - year
    CCoef = round(results.params[0],5)
    CPVal = round(results.pvalues[0],5)
    DCoef = round(results.params[1],5)
    DPVal = round(results.pvalues[1],5)
    r2 = round(results.rsquared,5)
    probF = round(results.f_pvalue,5)
    dfRes = round(results.df_resid,5)

    row = pd.Series([title,year,dumYear,relDumYear,CCoef,CPVal,DCoef,DPVal,r2,probF,dfRes],index=['Title','Year','DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
    df = df.append(row, ignore_index=True)
    # Changing dtypes back from object to floats
    df[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']] = df[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']].apply(np.float32)
    return df

for paper in papers:
    print paper
    y = list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).Similarity.mean())
    papYear = papSim[papSim.TopPaperTitle==paper].TopPaperYear.mean()
    
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1995.mean()),y,1995,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1996.mean()),y,1996,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1997.mean()),y,1997,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1998.mean()),y,1998,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1999.mean()),y,1999,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2000.mean()),y,2000,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2001.mean()),y,2001,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2002.mean()),y,2002,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2003.mean()),y,2003,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2004.mean()),y,2004,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2005.mean()),y,2005,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2006.mean()),y,2006,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2007.mean()),y,2007,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2008.mean()),y,2008,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2009.mean()),y,2009,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2010.mean()),y,2010,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2011.mean()),y,2011,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2012.mean()),y,2012,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2013.mean()),y,2013,paper,papYear,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2014.mean()),y,2014,paper,papYear,simRegs)

simRegs.to_csv('pap_simRegRaw.csv')

# <codecell>

print papers

# <codecell>

print simRegs
minRelYear = int(simRegs.RelDumYear.min())
#print "minRelYear", minRelYear
maxRelYear = int(simRegs.RelDumYear.max())
#print "maxRelYear", maxRelYear
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
#print "min RelDumYear", simRegs.RelDumYear.min()
#print "max RelDumYear", simRegs.RelDumYear.max()
#print len(simRegs.groupby(['RelDumYear']).CCoef.mean().values)

relDumYears = list(simRelMeans.RelDumYear)

meanPval = []
meanCoef = []

for relYr in relDumYears:
    meanPval.append(simRegs[simRegs.RelDumYear==relYr].DPVal.mean())
    meanCoef.append(simRegs[simRegs.RelDumYear==relYr].DCoef.mean())

#print "relDumYears",relDumYears
#print "meanPval",meanPval
#print "meanCoef", meanCoef

# <codecell>

# Plot for each paper
for paper in papers:
    print paper
    DPVal = list(simRegs[simRegs.Title==paper].DPVal)
    DCoef = list(simRegs[simRegs.Title==paper].DCoef)
    relDumYears = list(simRegs[simRegs.Title==paper].RelDumYear)
    
    fig = plt.figure(figsize=(13,5))
    ax1 = fig.add_subplot(121)
    for i in range(len(DPVal)):
        if DPVal[i]<=0.1:
            ax1.scatter(relDumYears[i],DPVal[i], marker='o', c='b',s=40)
        else:
            ax1.scatter(relDumYears[i],DPVal[i], marker='x', c='r',s=50)
    ax1.set_ylabel('Dummy P-Value')
    ax1.set_xlabel('Dummy Year Relative to Publication')
    ax1.set_title('{0}\nMean OLS P-Values by Relative Dummy Year'.format(paper))
    ax1.set_ylim(0,1)
    ax1.axvline(x=0, color='r', ls='--', lw=2)
    #ax1.text(1,0.8,'Year of Publication',fontsize=12,)
    ax1.grid()
    
    ax2 = fig.add_subplot(122)
    for i in range(len(DPVal)):
        if DPVal[i]<=0.1:
            ax2.scatter(relDumYears[i],DCoef[i], marker='o', c='b',s=40)
        else:
            ax2.scatter(relDumYears[i],DCoef[i], marker='x', c='r',s=50)
    ax2.set_ylabel('Dummy Coefficient')
    ax2.set_xlabel('Dummy Year Relative to Publication')
    ax2.set_title('Mean OLS Coefficients by Relative Dummy Year'.format(paper))
    ax2.axvline(x=0, color='r', ls='--', lw=2)
    #ax2.text(1,-0.003,'Year of Publication',fontsize=12,)
    ax2.grid()
    title = paper
    title = title.replace(':', '')
    print "title", title
    plt.savefig('papSim_{0}.png'.format(title))
    plt.show()

# <codecell>

# Plot average across papers
relDumYears = list(simRelMeans.RelDumYear)

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
for i in range(len(meanPval)):
    if meanPval[i]<=0.1:
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
    if meanPval[i]<=0.1:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='o', c='b',s=40)
    else:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='x', c='r',s=50)
#ax2.scatter(relDumYears,meanCoef)
ax2.set_ylabel('Mean Coefficient')
ax2.set_xlabel('Dummy Year Relative to Publication')
ax2.set_title('Top AI Papers Dummy Regression Coefficients\nMean OLS Coefficients by Relative Dummy Year')
ax2.axvline(x=0, color='r', ls='--', lw=2)
ax2.text(1,-0.003,'Year Published',fontsize=12,)
ax2.grid()
#plt.savefig('pat_diffusion.png')
plt.show()

# <codecell>

# Histograms
fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
ax1.hist(papSim[(papSim.RelYear<=0)&(papSim.flg==1)].Similarity,bins=50)
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Similarity')
ax1.set_title('Histogram Top AI Papers Similarity\nSimilarity Measures Before Patent Publication')
meanBefore = papSim[(papSim.RelYear<=0)&(papSim.flg==1)].Similarity.mean()
ax1.axvline(x=meanBefore, color='r', ls='-', lw=2)
ax1.set_xlim(0,papSim[papSim.flg==1].Similarity.max()*1.1)
ax1.grid()

ax2 = fig.add_subplot(122)
ax2.hist(papSim[(papSim.RelYear>0)&(papSim.flg==1)].Similarity,bins=50)
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Similarity')
ax2.set_title('Histogram Top AI Papers Similarity\nSimilarity Measures After Patent Publication')
meanAfter = papSim[(papSim.RelYear>0)&(papSim.flg==1)].Similarity.mean()
ax2.axvline(x=meanAfter, color='r', ls='-', lw=2)
ax2.set_xlim(0,papSim[papSim.flg==1].Similarity.max()*1.1)
ax2.grid()
plt.show()

# <codecell>

# Descriptive Statistics
papSim = papSim[papSim.flg==1]
print "N Before", len(papSim[papSim.RelYear<=0].Similarity)
print "Mean Before", papSim[papSim.RelYear<=0].Similarity.mean()
print "Std Before", papSim[papSim.RelYear<=0].Similarity.std()
print "Min Before", papSim[papSim.RelYear<=0].Similarity.min()
print "Max Before", papSim[papSim.RelYear<=0].Similarity.max()
print "\n"
print "N After", len(papSim[papSim.RelYear>0].Similarity)
print "Mean After", papSim[papSim.RelYear>0].Similarity.mean()
print "Std After", papSim[papSim.RelYear>0].Similarity.std()
print "Min After", papSim[papSim.RelYear>0].Similarity.min()
print "Max After", papSim[papSim.RelYear>0].Similarity.max()

# <codecell>

print aimlAbs.Year.min()

# <codecell>

# Regressions for top 10% for papers
simRegs = pd.DataFrame(columns=['Title', 'Year', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid'])
simRegs[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']] = simRegs[['DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']].apply(np.float32)
def doMeanSimReg(x1,dumYear,title,df):
    # Find top %x by similarity
    y = []
    for yr in range(papSim.PapJRefYear.min(),papSim.PapJRefYear.max()+1):
        quantCut = papSim[(papSim.TopPaperTitle==paper)&(papSim.PapJRefYear==yr)].Similarity.quantile(.90)
        y.append(papSim[(papSim.TopPaperTitle==paper)&(papSim.PapJRefYear==yr)&(papSim.Similarity>=quantCut)].Similarity.mean())
    
    X = sm.add_constant(zip(x1), prepend=True)
    results = sm.OLS(y, X).fit()
    
    year = papSim[papSim.TopPaperTitle==title].TopPaperYear.mean()
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

for paper in papers:
    print paper
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1995.mean()),1995,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1996.mean()),1996,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1997.mean()),1997,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1998.mean()),1998,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT1999.mean()),1999,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2000.mean()),2000,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2001.mean()),2001,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2002.mean()),2002,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2003.mean()),2003,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2004.mean()),2004,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2005.mean()),2005,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2006.mean()),2006,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2007.mean()),2007,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2008.mean()),2008,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2009.mean()),2009,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2010.mean()),2010,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2011.mean()),2011,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2012.mean()),2012,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2013.mean()),2013,paper,simRegs)
    simRegs = doMeanSimReg(list(papSim[papSim.TopPaperTitle==paper].groupby(['PapJRefYear']).D_GT2014.mean()),2014,paper,simRegs)

print simRegs

# <codecell>

print simRegs
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

print "relDumYears",relDumYears
print "meanPval",meanPval
print "meanCoef", meanCoef

# <codecell>

# Plot average across papers
fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
for i in range(len(meanPval)):
    if meanPval[i]<=0.1:
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
    if meanPval[i]<=0.1:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='o', c='b',s=40)
    else:
        ax2.scatter(relDumYears[i],meanCoef[i], marker='x', c='r',s=50)
#ax2.scatter(relDumYears,meanCoef)
ax2.set_ylabel('Mean Coefficient')
ax2.set_xlabel('Dummy Year Relative to Publication')
ax2.set_title('Top AI Papers Dummy Regression Coefficients\nMean OLS Coefficients by Relative Dummy Year')
ax2.axvline(x=0, color='r', ls='--', lw=2)
ax2.text(1,-0.003,'Year Published',fontsize=12,)
ax2.grid()
#plt.savefig('pat_diffusion.png')
plt.show()

# <codecell>

# Plot for each paper
for paper in papers:
    print paper
    DPVal = list(simRegs[simRegs.Title==paper].DPVal)
    DCoef = list(simRegs[simRegs.Title==paper].DCoef)
    relDumYears = list(simRegs[simRegs.Title==paper].RelDumYear)
    
    fig = plt.figure(figsize=(13,5))
    ax1 = fig.add_subplot(121)
    for i in range(len(DPVal)):
        if DPVal[i]<=0.1:
            ax1.scatter(relDumYears[i],DPVal[i], marker='o', c='b',s=40)
        else:
            ax1.scatter(relDumYears[i],DPVal[i], marker='x', c='r',s=50)
    ax1.set_ylabel('Dummy P-Value')
    ax1.set_xlabel('Dummy Year Relative to Publication')
    ax1.set_title('{0}\nMean OLS P-Values by Relative Dummy Year'.format(paper))
    ax1.set_ylim(0,1)
    ax1.axvline(x=0, color='r', ls='--', lw=2)
    #ax1.text(1,0.8,'Year of Publication',fontsize=12,)
    ax1.grid()
    
    ax2 = fig.add_subplot(122)
    for i in range(len(DPVal)):
        if DPVal[i]<=0.1:
            ax2.scatter(relDumYears[i],DCoef[i], marker='o', c='b',s=40)
        else:
            ax2.scatter(relDumYears[i],DCoef[i], marker='x', c='r',s=50)
    ax2.set_ylabel('Dummy Coefficient')
    ax2.set_xlabel('Dummy Year Relative to Publication')
    ax2.set_title('Mean OLS Coefficients by Relative Dummy Year'.format(paper))
    ax2.axvline(x=0, color='r', ls='--', lw=2)
    #ax2.text(1,-0.003,'Year of Publication',fontsize=12,)
    ax2.grid()
    title = paper
    title = title.replace(':', '')
    print "title", title
    plt.savefig('papSim_{0}.png'.format(title))
    plt.show()

