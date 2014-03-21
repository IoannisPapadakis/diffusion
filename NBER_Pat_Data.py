# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

orig7606 = pd.read_stata('orig_gen_76_06.dta')
pat7606 = pd.DataFrame.from_csv('pat76_06_ipc.csv')
#df = pd.DataFrame.from_csv('apat63_99.txt')
#cf = pd.DataFrame.from_csv('cite75_99.txt')

# <codecell>

import networkx as nx

# <codecell>

# Put originality, generality, and citation measures onto the pat76-06 dataframe
pdf2 = pd.merge(pdf, df, on=['patent'])
# Since pat76-06 has multiple records for each patent = number of assigness, remove duplicates so there is one record per patent
pdf3 = pdf2.drop_duplicates(cols=['patent'])

# <codecell>

# Put originality, generality, and citation measures onto the similarity dataframe
sim = pd.DataFrame.from_csv('bio_similarity6.csv')
sim.rename(columns={'PatNum':'patent'}, inplace=True)
sim2 = pd.merge(sim,pdf3,on=['patent'])
sim2.to_csv('bio_similarity_exp.csv')

# <codecell>

bioPats = pd.DataFrame.from_csv('bio_patents.csv')
bioPats.rename(columns={'PatentNum':'patent'}, inplace=True)
bioPats2 = pd.merge(bioPats,pdf3,on=['patent'])
bioPats2.to_csv('bio_patents_exp.csv')

# <codecell>

#bioICL = pd.DataFrame(['A01H  100','A01H  400','A61K 3800','A61K 3900','A61K 4800','C02F  334','C07G 1100','C07G 1300','C07G 1500','C07K  400','C07K 1400','C07K 1600','C07K 1700','C07K 1900','G01N 27327','G01N 3353','G01N 33531','G01N 33532','G01N 33533','G01N 33534','G01N 33535','G01N 33536','G01N 33537','G01N 33538','G01N 33539','G01N 3354','G01N 33541','G01N 33542','G01N 33543','G01N 33544','G01N 33545','G01N 33546','G01N 33547','G01N 33548','G01N 33549','G01N 3355','G01N 33551','G01N 33552','G01N 33553','G01N 33554','G01N 33555','G01N 33556','G01N 33557','G01N 33558','G01N 33559','G01N 3357','G01N 33571','G01N 33572','G01N 33573','G01N 33574','G01N 33575','G01N 33576','G01N 33577','G01N 33578','G01N 33579','G01N 3368','G01N 3374','G01N 3376','G01N 3378','G01N 3388','G01N 3392'], columns=['icl'])
#bioICLclass = pd.DataFrame(['C12M','C12N','C12P','C12Q','C12S'],columns=['icl_class'])
#bioPats1 = pdf3.merge(bioICL,on=['icl'])
#bioPats2 = pdf3.merge(bioICLclass,on=['icl_class'])
#bioPats = pd.concat([bioPats1,bioPats2])

bioPats.groupby('gyear').size().plot()
bioPats.ncited.describe()

# <codecell>

# find top x% of cited patents in a year
A = bioPats[(bioPats.gyear==1999)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.99)
A = bioPats[(bioPats.gyear==1999)&(bioPats.ncited>=quantCut)]
bioPatList1999 = list(A.patent)
print bioPatList1999

A = bioPats[(bioPats.gyear==1998)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.99)
A = bioPats[(bioPats.gyear==1998)&(bioPats.ncited>=quantCut)]
bioPatList1998 = list(A.patent)
print bioPatList1998

A = bioPats[(bioPats.gyear==1997)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.99)
A = bioPats[(bioPats.gyear==1997)&(bioPats.ncited>=quantCut)]
bioPatList1997 = list(A.patent)
print bioPatList1997

A = bioPats[(bioPats.gyear==1996)]
print A.ncited.describe()
quantCut = A.ncited.quantile(.99)
A = bioPats[(bioPats.gyear==1996)&(bioPats.ncited>=quantCut)]
bioPatList1996 = list(A.patent)
print bioPatList1996

# <codecell>

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

# <codecell>

#print df.describe()
#years = range(1963,2000)
#print years
df[(df.GYEAR==1995)&(df.NCLASS==706)&(df.CRECEIVE>10)].CRECEIVE

# <codecell>

numPatGYEAR = []
numUSPatGYEAR = []
numNONUSPatGYEAR = []
for i in years:
    dfYEAR = df[df.GYEAR == i]
    numPatGYEAR.append(len(dfYEAR))
    dfYEAR = df[(df.GYEAR == i) & (df.COUNTRY=='US')]
    numUSPatGYEAR.append(len(dfYEAR))
    dfYEAR = df[(df.GYEAR == i) & (df.COUNTRY!='US')]
    numNONUSPatGYEAR.append(len(dfYEAR))

# <codecell>

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1.plot(years, df.groupby(['GYEAR']).size(), lw=2, c='k')
ax1.plot(years, df[df.COUNTRY=='US'].groupby(['GYEAR']).size(), lw=2, ls='--', c='b')
ax1.plot(years, df[df.COUNTRY!='US'].groupby(['GYEAR']).size(), lw=2, ls='-.', c='r')
labels = ['Total', 'US', 'Non-US'] 
ax1.legend(labels, loc=0)
ax1.set_title('Number of Patents by Grant Year: \n US and non-US') 
ax1.set_ylabel('Number of Patents Granted')
ax1.set_xlabel('Year')
plt.show()

# <codecell>

cat = [1, 2, 3, 4, 5, 6]
catName = ['Chemical', 'Cmp&Cmm', 'Drgs&Med', 'Elec', 'Mech', 'Others']

numCat1G = []
numCat2G = []
numCat3G = []
numCat4G = []
numCat5G = []
numCat6G = []

numCat1A = []
numCat2A = []
numCat3A = []
numCat4A = []
numCat5A = []
numCat6A = []

for i in years:
    for j in cat:
        dfYEARG = df[(df.GYEAR == i) & (df.CAT==j)]
        dfYEARA = df[(df.APPYEAR == i) & (df.CAT==j)]
        if j==1:
            numCat1G.append(len(dfYEARG))
            numCat1A.append(len(dfYEARA))
        elif j==2:
            numCat2G.append(len(dfYEARG))
            numCat2A.append(len(dfYEARA))
        elif j==3:
            numCat3G.append(len(dfYEARG))
            numCat3A.append(len(dfYEARA))
        elif j==4:
            numCat4G.append(len(dfYEARG))
            numCat4A.append(len(dfYEARA))
        elif j==5:
            numCat5G.append(len(dfYEARG))
            numCat5A.append(len(dfYEARA))
        elif j==6:
            numCat6G.append(len(dfYEARG))
            numCat6A.append(len(dfYEARA))

# <codecell>

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
ax1.plot(years, numCat1G, lw=2, c='k')
ax1.plot(years, numCat2G, lw=2, c='b')
ax1.plot(years, numCat3G, lw=2, c='r')
ax1.plot(years, numCat4G, lw=2, c='g')
ax1.plot(years, numCat5G, lw=2, c='y')
ax1.plot(years, numCat6G, lw=2, c='m')
ax1.legend(catName, loc=0)
ax1.set_title('Number of Patents Granted: \n by Category and Grant Year') 
ax1.set_ylabel('Number of Patents Granted')
ax1.set_xlabel('Year')

ax2 = fig.add_subplot(122)
ax2.plot(years, numCat1A, lw=2, c='k')
ax2.plot(years, numCat2A, lw=2, c='b')
ax2.plot(years, numCat3A, lw=2, c='r')
ax2.plot(years, numCat4A, lw=2, c='g')
ax2.plot(years, numCat5A, lw=2, c='y')
ax2.plot(years, numCat6A, lw=2, c='m')
ax2.legend(catName, loc=0)
ax2.set_title('Number of Patents Applied For: \n by Category and Grant Year') 
ax2.set_ylabel('Number of Patents Applied For')
ax2.set_xlabel('Year')

plt.show()

# <codecell>

assTypeString = '1=unassigned\n2=US nongov\n3=non-US nongov\n4=US individual\n5=non-US individual\n6=US fed gov\n7=non-US gov'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
bins = range(1,8)
ax1.hist(df.ASSCODE, bins)
ax1.set_title('Histogram of Assignee Type') 
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Assignee Type')
ax1.text(4.5, 1300000, assTypeString, fontsize=12,horizontalalignment='left', verticalalignment='top',bbox=props)
plt.show()

# <codecell>

numUSPatCorp = []
numUSTot = []
numNOUSPatCorp = []
numNOUSTot = []

USCodes = [2, 4, 6]
NONUSCodes = [3, 5, 7]

for i in [1963, 1964]:
    dfYEAR = df[(df.APPYEAR == i) & (df.ASSCODE in USCodes)]
    numUSTot.append(len(dfYEAR))
    dfYEAR = df[(df.APPYEAR == i) & (df.ASSCODE == 2)]
    numUSPatCorp.append(len(dfYEAR))
    dfYEAR = df[(df.APPYEAR == i) & (df.ASSCODE in NONUSCodes)]
    numNOUSTot.append(len(dfYEAR))
    dfYEAR = df[(df.APPYEAR == i) & (df.ASSCODE == 3)]
    numNOUSPatCorp.append(len(dfYEAR))

# <codecell>

#groupGYear1 = df[(df.ASSCODE==1)&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
#groupGYear2 = df[(df.ASSCODE==2)&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
#groupGYear3 = df[(df.ASSCODE==3)&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
#groupGYear4 = df[(df.ASSCODE==4)&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
#groupGYear5 = df[(df.ASSCODE==5)&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
#groupGYear6 = df[(df.ASSCODE==6)&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
#groupGYear7 = df[(df.ASSCODE==7)&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)\

us2 = df[(df.ASSCODE==2)&(df.COUNTRY=='US')&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
us3 = df[(df.ASSCODE==3)&(df.COUNTRY=='US')&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
usPerCorp = (us2+us3)/df[(df.COUNTRY=='US')&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)

nus2 = df[(df.ASSCODE==2)&(df.COUNTRY!='US')&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
nus3 = df[(df.ASSCODE==3)&(df.COUNTRY!='US')&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)
nusPerCorp = (nus2+nus3)/df[(df.COUNTRY!='US')&(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1.plot(range(1965,2000), usPerCorp, lw=2, c='b')
ax1.plot(range(1965,2000), nusPerCorp, lw=2, ls='-.', c='r')
labels = ['US', 'Non-US'] 
ax1.legend(labels, loc=0)
ax1.set_title('Share of Patents Assigned to Corporations:\nby App Year, Non-Adjusted') 
ax1.set_ylabel('Share of Corporate Patents')
ax1.set_xlabel('Year')
ax1.set_ylim(0,1)
plt.show()


#print c
#numUSCorpPat = (groupGYear3)/(groupGYear1+groupGYear2+groupGYear3+groupGYear4+groupGYear5+groupGYear6+groupGYear7)
#numUSCorpPat = (groupGYear2)/(groupGYear2+groupGYear4+groupGYear6)
#print numUSCorpPat
#groupGYear = df[(df.COUNTRY=='US')&(df.APPYEAR>1964)].groupby(['APPYEAR','ASSCODE']).size().apply(float)
#print groupGYear
#print (groupGYear[2]+groupGYear[3])/(groupGYear[1]+groupGYear[2]+groupGYear[3]+groupGYear[4]+groupGYear[5]+groupGYear[6]+groupGYear[7])

# <codecell>

print df.groupby(['GYEAR','ASSCODE']).size()

# <codecell>

catName = ['Chemical', 'Cmp&Cmm', 'Drgs&Med', 'Elec', 'Mech', 'Others']

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
ax1.plot(years, df[df.CAT==1].groupby(['GYEAR']).size(), lw=2, c='k')
ax1.plot(years, df[df.CAT==2].groupby(['GYEAR']).size(), lw=2, c='b')
ax1.plot(years, df[df.CAT==3].groupby(['GYEAR']).size(), lw=2, c='r')
ax1.plot(years, df[df.CAT==4].groupby(['GYEAR']).size(), lw=2, c='g')
ax1.plot(years, df[df.CAT==5].groupby(['GYEAR']).size(), lw=2, c='y')
ax1.plot(years, df[df.CAT==6].groupby(['GYEAR']).size(), lw=2, c='m')
ax1.set_title('Number of Patents Granted: \n by Category and Grant Year') 
ax1.set_ylabel('Number of Patents Granted')
ax1.set_xlabel('Year')
ax1.legend(catName, loc=0)

ax2 = fig.add_subplot(122)
ax2.plot(years, df[(df.CAT==1)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size(), lw=2, c='k')
ax2.plot(years, df[(df.CAT==2)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size(), lw=2, c='b')
ax2.plot(years, df[(df.CAT==3)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size(), lw=2, c='r')
ax2.plot(years, df[(df.CAT==4)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size(), lw=2, c='g')
ax2.plot(years, df[(df.CAT==5)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size(), lw=2, c='y')
ax2.plot(years, df[(df.CAT==6)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size(), lw=2, c='m')
ax2.set_title('Number of Patents Granted: \n by Category and Grant Year') 
ax2.set_ylabel('Number of Patents Granted')
ax2.set_xlabel('Year')
ax2.legend(catName, loc=0)

plt.show()

# <codecell>

catName = ['Chemical', 'Cmp&Cmm', 'Drgs&Med', 'Elec', 'Mech', 'Others']
totalG = df.groupby(['GYEAR']).size().apply(float)
totalA = df[(df.APPYEAR>1964)].groupby(['APPYEAR']).size().apply(float)

fig = plt.figure(figsize=(16,5))
ax1 = fig.add_subplot(121)
ax1.plot(years, df[df.CAT==1].groupby(['GYEAR']).size()/totalG, lw=2, c='k')
ax1.plot(years, df[df.CAT==2].groupby(['GYEAR']).size()/totalG, lw=2, c='b')
ax1.plot(years, df[df.CAT==3].groupby(['GYEAR']).size()/totalG, lw=2, c='r')
ax1.plot(years, df[df.CAT==4].groupby(['GYEAR']).size()/totalG, lw=2, c='g')
ax1.plot(years, df[df.CAT==5].groupby(['GYEAR']).size()/totalG, lw=2, c='y')
ax1.plot(years, df[df.CAT==6].groupby(['GYEAR']).size()/totalG, lw=2, c='m')
ax1.set_title('Share of Patents by Tech Category: \n by Grant Year') 
ax1.set_ylabel('Share of Patents')
ax1.set_xlabel('Year')
ax1.set_ylim(0,0.5)
ax1.legend(catName, loc=0)

ax2 = fig.add_subplot(122)
ax2.plot(years, df[(df.CAT==1)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size()/totalA, lw=2, c='k')
ax2.plot(years, df[(df.CAT==2)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size()/totalA, lw=2, c='b')
ax2.plot(years, df[(df.CAT==3)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size()/totalA, lw=2, c='r')
ax2.plot(years, df[(df.CAT==4)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size()/totalA, lw=2, c='g')
ax2.plot(years, df[(df.CAT==5)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size()/totalA, lw=2, c='y')
ax2.plot(years, df[(df.CAT==6)&(df.APPYEAR>1962)].groupby(['APPYEAR']).size()/totalA, lw=2, c='m')
ax2.set_title('Share of Patents by Tech Category: \n by App Year') 
ax2.set_ylabel('Share of Patents')
ax2.set_xlabel('Year')
ax2.set_ylim(0,0.5)
ax2.legend(catName, loc=0)

plt.show()

# <codecell>

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1.plot(range(1965,2000), df[(df.APPYEAR>1964)].groupby(['APPYEAR'])['CMADE'].mean(), lw=2, c='k')
ax1.plot(range(1965,2000), df[(df.APPYEAR>1964)].groupby(['APPYEAR'])['CRECEIVE'].mean(), lw=2, ls='--', c='b')
ax1.plot(range(1965,2000), df[(df.GYEAR>1964)].groupby(['GYEAR'])['CMADE'].mean(), lw=2, ls='-.', c='r')
labels = ['Cit Made by App Year', 'Cit Received by App Year', 'Cit Made by Grant Year'] 
ax1.legend(labels, loc=0)
ax1.set_title('Citations Made and Received:\nShowing Truncation') 
ax1.set_ylabel('Citations')
ax1.set_xlabel('Year')
plt.show()

# <codecell>

catName = ['Chemical', 'Cmp&Cmm', 'Drgs&Med', 'Elec', 'Mech', 'Others']

fig = plt.figure(figsize=(16,5))
ax1 = fig.add_subplot(121)
ax1.plot(range(1965,2000), df[(df.CAT==1)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CMADE'].mean(), lw=2, c='k')
ax1.plot(range(1965,2000), df[(df.CAT==2)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CMADE'].mean(), lw=2, c='b')
ax1.plot(range(1965,2000), df[(df.CAT==3)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CMADE'].mean(), lw=2, c='r')
ax1.plot(range(1965,2000), df[(df.CAT==4)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CMADE'].mean(), lw=2, c='g')
ax1.plot(range(1965,2000), df[(df.CAT==5)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CMADE'].mean(), lw=2, c='y')
ax1.plot(range(1965,2000), df[(df.CAT==6)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CMADE'].mean(), lw=2, c='m')
labels = ['Cit Made by App Year', 'Cit Received by App Year', 'Cit Made by Grant Year'] 
ax1.legend(catName, loc=0)
ax1.set_title('Citations Made and Received:\nby Technology Category') 
ax1.set_ylabel('Citations Made')
ax1.set_xlabel('Year')

ax2 = fig.add_subplot(122)
ax2.plot(range(1965,2000), df[(df.CAT==1)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CRECEIVE'].mean(), lw=2, c='k')
ax2.plot(range(1965,2000), df[(df.CAT==2)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CRECEIVE'].mean(), lw=2, c='b')
ax2.plot(range(1965,2000), df[(df.CAT==3)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CRECEIVE'].mean(), lw=2, c='r')
ax2.plot(range(1965,2000), df[(df.CAT==4)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CRECEIVE'].mean(), lw=2, c='g')
ax2.plot(range(1965,2000), df[(df.CAT==5)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CRECEIVE'].mean(), lw=2, c='y')
ax2.plot(range(1965,2000), df[(df.CAT==6)&(df.APPYEAR>1964)].groupby(['APPYEAR'])['CRECEIVE'].mean(), lw=2, c='m')
ax2.legend(catName, loc=0)
ax2.set_title('Citations Made and Received:\nby Technology Category') 
ax2.set_ylabel('Citations Made')
ax2.set_xlabel('Year')

plt.show()


# <codecell>

fig = plt.figure(figsize=(14,5))
ax1 = fig.add_subplot(121)
ax1.hist(df[(df.APPYEAR>1975)&(df.BCKGTLAG<=50)]['BCKGTLAG'],bins=range(0,51), normed=True) 
ax1.set_title('Histogram of Mean Backward Citation Lag') 
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Mean Backward Citation Lag')

ax2 = fig.add_subplot(122)
ax2.hist(df[(df.APPYEAR>1975)&(df.FWDAPLAG<=50)]['FWDAPLAG'],bins=range(0,25), normed=True) 
ax2.set_title('Histogram of Mean Forward Citation Lag\n(Exhibits trunctation)') 
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Mean Forward Citation Lag')

plt.show()

# <codecell>

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1.plot(range(1963,2000), df[df.SUBCAT==33].groupby(['GYEAR']).size(), lw=2, c='r')
ax1.set_title('Number of Biotech Patents Granted per Year') 
ax1.set_ylabel('Number of Patents Granted')
ax1.set_xlabel('Year')
plt.show()

# <codecell>

pltYears = [1970, 1975, 1980, 1985, 1990, 1995]
for yr in pltYears:
    cf['CITING'] = cf.index
    cfGrouped = cf.groupby('CITED').size().order()
    nodes = df[(df.GYEAR==yr)&(df.SUBCAT==33)].index
    nodes = list(nodes)
    ndf = pd.DataFrame(nodes, columns=['CITED'])
    mdf = pd.merge(cf,ndf, on=['CITED'])
    
    newNodes = mdf.CITED
    newNodes = newNodes.append(mdf.CITING)
    newNodes = newNodes.drop_duplicates()
    
    DG = nx.DiGraph()
    for i in newNodes:
        DG.add_node(i)
    for index, row in mdf.iterrows():
        DG.add_edge(row['CITED'],row['CITING'])
    G = DG.to_undirected()

    clo_cen = nx.closeness_centrality(G)
    bet_cen = nx.betweenness_centrality(G)
    cloList = []
    betList = []
    for key, value in clo_cen.iteritems():
        cloList.append(value)
    for key, value in bet_cen.iteritems():
        betList.append(value)
    
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121)
    ax1.hist(cloList)
    ax1.set_title('Histogram {0} Closeness Centrality'.format(yr)) 
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Closeness Centrality')
    ax1.set_xlim(0,0.1)
    ax1.set_ylim(0,3000)
    ax2 = fig.add_subplot(122)
    ax2.hist(betList)
    ax2.set_title('Histogram {0} Betweenness Centrality'.format(yr)) 
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Betweenness Centrality')
    ax2.set_xlim(0,0.25)
    ax2.set_ylim(0,6000)
    fileName = "clo_bet_hist_{0}".format(yr)
    plt.savefig(fileName)
    plt.close()

    centrality_scatter(bet_cen, clo_cen, ylab="Closeness Centrality",xlab="Betweenness Centrality",title="Scatter {0} Closeness-Betweenness".format(yr),line=False, fName = "clo_bet_scat_{0}".format(yr))

# <codecell>

coreNodes = df[(df.GYEAR==1980)&(df.SUBCAT==33)&(df.ORIGINAL>0.75)].index
print coreNodes

# <codecell>

cf['CITING'] = cf.index
coreNodes = df[(df.GYEAR==1980)&(df.SUBCAT==33)&(df.ORIGINAL>0.75)].index
coreNodes = list(coreNodes)
print "len(coreNodes)", len(coreNodes)

# Layer 1
# OUT  citations for coreNodes
ndf1 = pd.DataFrame(coreNodes, columns=['CITING'])
mdf1 = pd.merge(cf,ndf1, on=['CITING'])
# IN citations for coreNodes
ndf2 = pd.DataFrame(coreNodes, columns=['CITED'])
mdf2 = pd.merge(cf,ndf2, on=['CITED'])
# All citations for coreNodes
netdf = mdf1.append(mdf2, ignore_index=True)
netdf = netdf.drop_duplicates()

nodesOut = list(set(netdf['CITING']))
nodesIn = list(set(netdf['CITED']))
finalNodes = list(set(nodesOut) | set(nodesIn))
print "Layer 1: len(finalNodes)", len(finalNodes)

# Layer 2
# OUT  citations for coreNodes
ndf1 = pd.DataFrame(finalNodes, columns=['CITING'])
mdf1 = pd.merge(cf,ndf1, on=['CITING'])
# IN citations for coreNodes
ndf2 = pd.DataFrame(finalNodes, columns=['CITED'])
mdf2 = pd.merge(cf,ndf2, on=['CITED'])
# All citations for coreNodes
netdf = mdf1.append(mdf2, ignore_index=True)
netdf = netdf.drop_duplicates()

nodesOut = list(set(netdf['CITING']))
nodesIn = list(set(netdf['CITED']))
finalNodes = list(set(nodesOut) | set(nodesIn))
print "Layer 2: len(finalNodes)", len(finalNodes)

# <codecell>

DG = nx.DiGraph()
for i in finalNodes:
    DG.add_node(i)
for index, row in netdf.iterrows():
    DG.add_edge(row['CITING'],row['CITED'])

G = DG.to_undirected()

# <codecell>

# Spider through the network starting with core nodes, seems to not stop
coreNodes = df[(df.GYEAR==1980)&(df.SUBCAT==41)&(df.ORIGINAL>0.8)].index
coreNodes = list(coreNodes)
finalNodes = [x for x in coreNodes]

newNodes = [x for x in coreNodes]
while newNodes:
    # First put all new nodes into final nodes
    tmpNodes = list( (set(finalNodes) | set(newNodes)) - set(finalNodes))
    for node in tmpNodes:
        finalNodes.append(node)
    # For the first node in newNodes get in and out
    nodeOut = list(cf[cf.CITING==newNodes[0]].CITED)
    nodeIn = list(cf[cf.CITED==newNodes[0]].CITING)
    for node in nodeOut:
        if node not in finalNodes:
            if node not in newNodes:
                newNodes.append(node)
    for node in nodeIn:
        if node not in finalNodes:
            if node not in newNodes:
                newNodes.append(node)
    newNodes.remove(newNodes[0])
print finalNodes

# <codecell>

clo_cen = nx.closeness_centrality(G)
bet_cen = nx.betweenness_centrality(G)
cloList = []
betList = []
for key, value in clo_cen.iteritems():
    cloList.append(value)
for key, value in bet_cen.iteritems():
    betList.append(value)
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax1.hist(cloList)
ax1.set_title('Histogram 1970 Closeness Centrality') 
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Closeness Centrality')
ax2 = fig.add_subplot(122)
ax2.hist(betList)
ax2.set_title('Histogram 1970 Betweenness Centrality') 
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Betweenness Centrality')
#plt.savefig("clo_bet_hist_1970.png")
plt.show()

# <codecell>

def centrality_scatter(dict1,dict2, ylab="",xlab="",title="",line=False, fName =""):
    # Create figure and drawing axis
    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(111)
    # Create items and extract centralities
    items1 = sorted(dict1.items())
    items2 = sorted(dict2.items())
    xdata=[b for a,b in items1]
    ydata=[b for a,b in items2] 
    
    # Add each actor to the plot by ID 
    for p in xrange(len(items1)):
        ax1.text(x=xdata[p], y=ydata[p],s=str(items1[p][0]), color="b")
    if line: 
        # use NumPy to calculate the best fit
        slope, yint = plt.polyfit(xdata,ydata,1)
        xline = plt.xticks()[0] 
        yline = map(lambda x: slope*x+yint,xline)
        ax1.plot(xline,yline,ls='--',color='b')
    # Set new x- and y-axis limits
    #plt.xlim((0.0,max(xdata)+(.15*max(xdata))))
    #plt.ylim((0.0,max(ydata)+(.15*max(ydata)))) 
    # Add labels and save
    ax1.set_title(title)
    ax1.set_xlabel(xlab) 
    ax1.set_ylabel(ylab)
    ax1.set_ylim(0,0.12)
    ax1.set_xlim(0,0.3)
    plt.savefig(fName)
    plt.close()

# <codecell>

in_degrees = DG.in_degree() # dictionary node:degree
in_values = sorted(set(in_degrees.values())) 
in_hist = [in_degrees.values().count(x) for x in in_values]

out_degrees = DG.out_degree()
out_values = sorted(set(out_degrees.values()))
out_hist = [out_degrees.values().count(x) for x in out_values]


fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1.loglog(in_values, in_hist, lw=2, c='r')
ax1.loglog(out_values, out_hist, lw=2, ls='--', c='b')
labels = ['In-Degree', 'Out-Degree'] 
ax1.legend(labels, loc=0)
ax1.set_title('Network In and Out Degree Analysis') 
ax1.set_ylabel('Number of Nodes')
ax1.set_xlabel('Degree')
plt.show()
"""
clo_cen = nx.closeness_centrality(G)
clo_values = sorted(set(clo_cen.values()))
clo_hist = [clo_cen.values().count(x) for x in clo_values]

bet_cen = nx.betweenness_centrality(G)
bet_values = sorted(set(bet_cen.values()))
bet_hist = [bet_cen.values().count(x) for x in bet_values]

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax1.loglog(clo_values, clo_hist, lw=2, c='r')
ax1.set_title('Histogram 1970 Closeness Centrality') 
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Closeness Centrality')
ax2 = fig.add_subplot(122)
ax2.loglog(bet_values, bet_hist, lw=2, c='r')
ax2.set_title('Histogram 1970 Betweenness Centrality') 
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Betweenness Centrality')
#plt.savefig("clo_bet_hist_1970.png")
plt.show()
"""

# <codecell>

in_all_values = in_degrees.values()
print np.std(in_all_values)
print np.mean(in_all_values)
print np.max(in_all_values)
print np.percentile(in_all_values, 95)
print 
ccs = nx.average_clustering(G)
print ccs

# <codecell>

print coreNodes
for node in finalNodes:
    if node in coreNodes:
        G.node[node]['color'] = 'b'
    else:
        G.node[node]['color'] = 'r'
plt.figure(figsize=(20,20))
nx.draw_networkx(G, with_labels=False, node_size=50, c={4230676:0}, node_color = [G.node[node]['color'] for node in G.nodes()])

