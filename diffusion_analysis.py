"""
Perform diffusion analysis
Nathan Goldschlag
August 25, 2014
Version 1.0
Written in Python 2.7

This python program prepares the csv files for plotting and runs similarity regressions.

"""
## IMPORT LIBRARIES
import re
import csv
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import datetime
difPath = 'd:/diffusion_data'

def genDummies(df,indepYear,depYear):
    # Define greater than dummy year variables
    df['D_GT1993'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 1993:
            df.D_GT1993[row_index] = 1
        else:
            df.D_GT1993[row_index] = 0

    df['D_GT1994'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 1994:
            df.D_GT1994[row_index] = 1
        else:
            df.D_GT1994[row_index] = 0

    df['D_GT1995'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 1995:
            df.D_GT1995[row_index] = 1
        else:
            df.D_GT1995[row_index] = 0

    df['D_GT1996'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 1996:
            df.D_GT1996[row_index] = 1
        else:
            df.D_GT1996[row_index] = 0

    df['D_GT1997'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 1997:
            df.D_GT1997[row_index] = 1
        else:
            df.D_GT1997[row_index] = 0

    df['D_GT1998'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 1998:
            df.D_GT1998[row_index] = 1
        else:
            df.D_GT1998[row_index] = 0

    df['D_GT1999'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 1999:
            df.D_GT1999[row_index] = 1
        else:
            df.D_GT1999[row_index] = 0

    df['D_GT2000'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2000:
            df.D_GT2000[row_index] = 1
        else:
            df.D_GT2000[row_index] = 0

    df['D_GT2001'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2001:
            df.D_GT2001[row_index] = 1
        else:
            df.D_GT2001[row_index] = 0

    df['D_GT2002'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2002:
            df.D_GT2002[row_index] = 1
        else:
            df.D_GT2002[row_index] = 0

    df['D_GT2003'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2003:
            df.D_GT2003[row_index] = 1
        else:
            df.D_GT2003[row_index] = 0

    df['D_GT2004'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2004:
            df.D_GT2004[row_index] = 1
        else:
            df.D_GT2004[row_index] = 0

    df['D_GT2005'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2005:
            df.D_GT2005[row_index] = 1
        else:
            df.D_GT2005[row_index] = 0

    df['D_GT2006'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2006:
            df.D_GT2006[row_index] = 1
        else:
            df.D_GT2006[row_index] = 0

    df['D_GT2007'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2007:
            df.D_GT2007[row_index] = 1
        else:
            df.D_GT2007[row_index] = 0

    df['D_GT2008'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2008:
            df.D_GT2008[row_index] = 1
        else:
            df.D_GT2008[row_index] = 0

    df['D_GT2009'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2009:
            df.D_GT2009[row_index] = 1
        else:
            df.D_GT2009[row_index] = 0
            
    df['D_GT2010'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2010:
            df.D_GT2010[row_index] = 1
        else:
            df.D_GT2010[row_index] = 0
            
    df['D_GT2011'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2011:
            df.D_GT2011[row_index] = 1
        else:
            df.D_GT2011[row_index] = 0
            
    df['D_GT2012'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2012:
            df.D_GT2012[row_index] = 1
        else:
            df.D_GT2012[row_index] = 0

    df['D_GT2013'] = int()
    for row_index, row in df.iterrows():
        if row[indepYear] >= 2013:
            df.D_GT2013[row_index] = 1
        else:
            df.D_GT2013[row_index] = 0
   
    df['RelYear'] = int()
    for row_index,row in df.iterrows():
        df.RelYear[row_index] = row[indepYear] - row[depYear]
    
    return df


def doMeanSimReg(x1,y,obj,dumYear,objYear,df,resultCols):
    X = sm.add_constant(zip(x1), prepend=True)
    results = sm.OLS(y, X).fit()
    
    relDumYear = dumYear - objYear
    CCoef = round(results.params[0],5)
    CPVal = round(results.pvalues[0],5)
    DCoef = round(results.params[1],5)
    DPVal = round(results.pvalues[1],5)
    r2 = round(results.rsquared,5)
    probF = round(results.f_pvalue,5)
    dfRes = round(results.df_resid,5)

    row = pd.Series([obj,objYear,dumYear,relDumYear,CCoef,CPVal,DCoef,DPVal,r2,probF,dfRes],index=resultCols)
    df = df.append(row, ignore_index=True)
    return df

def runGTDummyRegs(simDF,cat,top):
    if cat in ['ai','bio']:
        resultCols = ['PatNum', 'PatYear', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']
        simRegs = pd.DataFrame(columns=resultCols)
        simDFObj = list(set(simDF.PatNum))        
    elif cat=='tc':
        resultCols = ['PapTC', 'PapYearTC', 'DumYear','RelDumYear','CCoef','CPVal','DCoef','DPVal','R2','Pfstat','DFResid']
        simRegs = pd.DataFrame(columns=resultCols)
        simDFObj = list(set(simDF.PapTitleTC))
    
    for obj in simDFObj:
        if cat in ['ai','bio']:        
            if top:
                y = []
                for yr in range(simDF.PapYear.min(),simDF.PapYear.max()+1):
                    quantCut = simDF[(simDF.PatNum==obj)&(simDF.PapYear==yr)].Similarity.quantile(.90)
                    y.append(simDF[(simDF.PatNum==obj)&(simDF.PapYear==yr)&(simDF.Similarity>=quantCut)].Similarity.mean())
                objYear =  simDF[simDF.PatNum==obj].PatYear.mean()
            else:
                y = list(simDF[simDF.PatNum==obj].groupby(['PapYear']).Similarity.mean())
                objYear =  simDF[simDF.PatNum==obj].PatYear.mean()
        elif cat=='tc':
            y = list(simDF[simDF.PapTitleTC==obj].groupby(['PapYear']).Similarity.mean())
            objYear =  simDF[simDF.PapTitleTC==obj].PapYearTC.mean()

        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1994.mean()),y,obj,1994,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1995.mean()),y,obj,1995,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1996.mean()),y,obj,1996,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1997.mean()),y,obj,1997,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1998.mean()),y,obj,1998,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT1999.mean()),y,obj,1999,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2000.mean()),y,obj,2000,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2001.mean()),y,obj,2001,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2002.mean()),y,obj,2002,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2003.mean()),y,obj,2003,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2004.mean()),y,obj,2004,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2005.mean()),y,obj,2005,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2006.mean()),y,obj,2006,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2007.mean()),y,obj,2007,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2008.mean()),y,obj,2008,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2009.mean()),y,obj,2009,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2010.mean()),y,obj,2010,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2011.mean()),y,obj,2011,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2012.mean()),y,obj,2012,objYear,simRegs,resultCols)
        simRegs = doMeanSimReg(list(simDF.groupby(['PapYear']).D_GT2013.mean()),y,obj,2013,objYear,simRegs,resultCols)
    #print simRegs
    if cat=='bio':
        simRegs.to_csv(difPath+'/bio_simRegRaw.csv')    
    elif cat=='ai':
        simRegs.to_csv(difPath+'/ai_simRegRaw.csv')
    elif cat=='tc':
        simRegs.to_csv(difPath+'/simTC_simRegRaw.csv')
    
    if cat in ['ai', 'bio']:
        # from regression results create mean regression results dataset
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
        
        if cat=='bio':
            if top:
                simRelMeans.to_csv(difPath+'/bio_simTopRegMeans.csv')
            else:
                simRelMeans.to_csv(difPath+'/bio_simRegMeans.csv')
        elif cat=='ai':
            if top:
                simRelMeans.to_csv(difPath+'/ai_simTopRegMeans.csv')
            else:
                simRelMeans.to_csv(difPath+'/ai_simRegMeans.csv')
        
def main():
    # prep abstract and patent data for analysis    
    # import abstracts
    bioAbs = pd.DataFrame.from_csv(difPath+'/bio_abstracts.csv')
    aiAbs = pd.DataFrame.from_csv(difPath+'/aiml_abstracts.csv')
    # import patents
    bioPats = pd.DataFrame.from_csv(difPath+'/bio_patents.csv')
    aiPats = pd.DataFrame.from_csv(difPath+'/ai_patents.csv')

    # calculate abstract lengths
    bioAbs['AbsLen'] = int()
    for row_index, row in bioAbs.iterrows():
        bioAbs.AbsLen[row_index] = len(row['Abstract'])
    aiAbs['AbsLen'] = int()
    for row_index, row in aiAbs.iterrows():
        aiAbs.AbsLen[row_index] = len(row['Abstract'])
    bioPats['AbsLen'] = int()
    for row_index, row in bioPats.iterrows():
        bioPats.AbsLen[row_index] = len(row['patent_abstract'])
    aiPats['AbsLen'] = int()
    for row_index, row in aiPats.iterrows():
        aiPats.AbsLen[row_index] = len(row['patent_abstract'])
    
    # store prep files
    bioAbs.to_csv(difPath+'/bio_abstracts_prep.csv')
    aiAbs.to_csv(difPath+'/ai_abstracts_prep.csv')
    bioPats.to_csv(difPath+'/bio_patents_prep.csv')
    aiPats.to_csv(difPath+'/ai_patents_prep.csv')
    
    ## prep similarity data for analysis
    # import bio similarity data
    bioSim = pd.DataFrame.from_csv(difPath+'/bio_similarity.csv')
    # put greater than dummies on similarity df
    print 'gen bio dummies'
    bioSim = genDummies(bioSim,'PapYear','PatYear')
    # store updated bio similarity df
    bioSim.to_csv(difPath+'/bio_similarity_prep.csv')
    # run bio similarity regressions
    print 'run bio regs'
    runGTDummyRegs(bioSim,'bio',False)
    # run top bio similarity regressions
    print 'run top bio regs'
    runGTDummyRegs(bioSim,'bio',True)
    
    
    # import ai similarity data
    aiSim = pd.DataFrame.from_csv(difPath+'/ai_similarity.csv')
    # put greater than dummies on similarity df
    print 'gen ai dummies'
    aiSim = genDummies(aiSim,'PapYear','PatYear')
    # store updated ai similarity df
    aiSim.to_csv(difPath+'/ai_similarity_prep.csv')
    # run ai similarity regressions
    print 'run ai regs'
    runGTDummyRegs(aiSim,'ai',False)
    # run top ai similarity regressions
    print 'run top ai regs'
    runGTDummyRegs(aiSim,'ai',True)
    
    # import sim test case data
    tcSim = pd.DataFrame.from_csv(difPath+'/simTC_similarity.csv')
    # drop duplicate by title
    tcSim = tcSim.drop_duplicates(['PapTitle'])
    # put greater than dummies on similarity df
    print 'gen tc dummies'
    tcSim = genDummies(tcSim,'PapYear','PapYearTC')
    # store updated tv similarity df
    tcSim.to_csv(difPath+'/simTC_similarity_prep.csv')
    # run test case similarity regressions
    print 'run tc regs'
    runGTDummyRegs(tcSim,'tc',False)
    
    
    
    
if __name__=="__main__":
    main()
