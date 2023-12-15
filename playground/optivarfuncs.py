import pandas as pd
import numpy as np
import itertools

#--------------------------------------get sample dataframe
def getdf(numb_days= 5, stocks= [0,1,3], numb_secs_day= 4, n_nulls= 1, drop_days= [1,2], target='target'):
    '''
    generate a dataframe
    num_days: how many days in dataframe
    stocks: stock ids
    n_nulls: first n rows per day have near and far price = NaN
    drop_days: remove these days from dataframe
    '''
    stockl = np.concatenate(numb_days*[list(itertools.repeat(n, numb_secs_day)) for n in stocks], axis=0).tolist()
    dayl = [np.trunc(d/(numb_secs_day*len(stocks))) for d in list(range(numb_days*numb_secs_day*len(stocks)))]
    vall = [10*i for i in list(range(len(stocks)*numb_days*numb_secs_day))]
    secl = [d % numb_secs_day for d in list(range(numb_days*numb_secs_day*len(stocks)))]

    df = pd.DataFrame(data={'stock_id': stockl, 'date_id': dayl, 'far_price': vall, 'near_price': [x+1 for x in vall], 'seconds_in_bucket': secl})

    #make first n entries in near_price and far_price NaN
    df.loc[((df.seconds_in_bucket <= n_nulls)), ['near_price', 'far_price']] = np.nan

    #create a target column
    df[target]=list(range(len(df)))

    #get rid of some days (drop_days=[1,2], .index gives index of the rows
    df.drop(df[df.date_id.isin(drop_days)].index, inplace=True)
    return df
#--------------------------------------get splits (no unit tests)
import numpy as np

def getsplitday( X, val_size, verbose = True):
    '''
    find split that seperates by day
    ex 
    tstdf=of.getdf(numb_days= 10, drop_days= [2,3])
    getsplitday( df, val_sze=0.1)
    9
    '''
    days=X.date_id.unique()
    index=int(np.floor(len(days)*(1-val_size)))
    if (verbose):
        print(f"For days{days} and val_size={val_size}, split day={days[index]}")
    return days[index]

def splitDataset(X, val_size, copy=False,verbose=True):
    '''
    ensures datasets do not split accross a day
    returns train and val
    ex 
    tstdf=of.getdf(numb_days= 10, drop_days= [2,3])
    X_train, X_val = splitDataset(tstdf, 0.2, verbose=False):
    '''
    #get the day to split on
    day=getsplitday( X,val_size, verbose)
    # print(f'df shape={X.shape}, min_date={X.date_id.min()},max_date={X.date_id.max()}, val_start={val_start}, tst_start={tst_start}')

    #get train, val
    if(copy==True):
        X_train=X[X.eval(f"(date_id<{day})")].copy()
        X_val=X.loc[(X.date_id>=day)].copy()
    else:
        X_train=X[X.eval(f"(date_id<{day})")]
        X_val=X.loc[(X.date_id>=day)]
    return  X_train, X_val

def splitTarget(X, dep_var='target',verbose=True):
    '''
    split target from X
    ex
    X_train, y_train = splitTarget(X_train, dep_var)
    '''
    y = X[dep_var]
    X.drop(columns=[dep_var],inplace=True)
    return X,y

def get2_DatasetAndTarget(X, dep_var='target', val_size=0.2,copy=False, verbose=True):
    '''
    split into 2 datasets
    ex 
    tstdf=of.getdf(numb_days= 10, drop_days= [2,3])
    X_train, X_val, y_train, y_val = get2_DatasetAndTarget(tstdf,dep_var='target', val_size, verbose=False):
    '''
    X_train, X_val = splitDataset(X, val_size, copy,verbose)
    X_train, y_train = splitTarget(X_train, dep_var)
    X_val, y_val = splitTarget(X_val, dep_var)
    if (verbose):
        print(f"len(X_train)={len(X_train)} and len(X_val)={len(X_val)}")
        tots=len(X_val)+len(X_train)
        print(f"train size={len(X_train)/tots}, val size={len(X_val)/tots}")

    return X_train, X_val, y_train, y_val

def get3_DatasetAndTarget(X, dep_var='target', val_size=0.1, test_size=0.2,copy=False,verbose = True):
    '''
    split into 3 datasets
    ex 
    tstdf=of.getdf(numb_days= 10, drop_days= [2,3])
    X_train, X_val, X_tst, y_train, y_val, y_tst = get3_DatasetAndTarget(tstdf, dep_var='target', .1,.2 verbose=False):
    '''
 
    X_train, X_val= splitDataset(X, val_size+test_size,copy, verbose)
    X_val, X_tst = splitDataset(X_val, test_size/(val_size+test_size),copy, verbose)

    X_train, y_train = splitTarget(X_train, dep_var)
    X_val, y_val = splitTarget(X_val, dep_var)
    X_tst, y_tst = splitTarget(X_tst, dep_var)

    if (verbose):
        print(f"len(X_train)={len(X_train)} and len(X_val)={len(X_val)}, len(X_tst)={len(X_tst)}")
        tots=len(X_val)+len(X_train)+len(X_tst)
        print(f"train size={len(X_train)/tots}, val size={len(X_val)/tots}, tst size={len(X_tst)/tots}")

    return X_train, X_val, X_tst, y_train, y_val, y_tst


#--------------------------------------infer missing columns
class autocol():
    '''
    adds 1 boolean column to df for each column in cols.
    used to track whether column value was inferred
    ex.
    a=autocol(['near_price','far_price'],df)
    a.remove_syn_columns(df)
    df
    '''
    def __init__(self, columns, df):
        self.columns=columns
        for c in self.columns:
            if 'syn_'+c not in df.columns:
                #0 means c is not inferred, 1 means it is
                df['syn_'+c] = 0
    def remove_syn_columns(self,df):
        l=[]
        for c in self.columns:
            l.append( 'syn_'+c)
        df.drop(columns=l, axis=1, inplace=True)


class bfs():
    '''
    hosts one bf object per column being backfilled
    WARNING: syn_colname is added to df for each column in columns
    ex.     df=of.getdf()
            bs=of.bfs(['near_price','far_price'], df)
            df = df.apply(bs.backfill, axis=1)
    '''
    def __init__(self, columns, df, dfgb=None):
        self.columns = columns

        if(df is not None):
            self.doautocol(df)
            
        if (dfgb is not None):
            dfgb.fillna(0, inplace=True)
            # self.days=list(self.dfgb.index.get_level_values(1).unique() )
        self.bfs = [bf(col, dfgb) for col in self.columns]

    def doautocol(self, df):
        #add appropriate syn_ columns to df
        self.a = autocol(self.columns, df)
        
    def backfill(self, x):
        for b in self.bfs:
            x = b.backfill(x)
        return x

    def remove_syn_columns(self, df):
        self.a.remove_syn_columns(df)

    def _debuginfo(self):
        for b in self.bfs:
            print(f"col={b.column}")
            print(f"  b.lastvaliddays={b.lastvaliddays}")
            print(f"  b.vals={b.vals}")


class bf():
    '''
    backfills NaNs in column with last value registered in that column (likely from the day before)

    '''    
    def __init__(self,column,dfgb=None):
        '''
        column: column to operate on
        '''
        self.column=column  #column to operate on
        self.syn_column='syn_'+column  #mark as 1 if synthetic value generated
        self.dfgb=dfgb
        self.lastvaliddays={}
        self.vals = {}  #cache of column val for (stock_id, date_id)
        
    def backfill(self, x):
        '''
        x: row of data
        '''
        #if null then continue
        if(pd.isnull(x[self.column])):
            x[self.column] = self._getlastvalidvalue(x.stock_id, x.date_id)
            x[self.syn_column] = 1
        else:  
            self.lastvaliddays[x.stock_id] = x.date_id
            self.vals[(x.stock_id,x.date_id)] = x[self.column]
        return x
    
    def _getlastvalidvalue(self, stock_id, date_id):
        try:
            if self.dfgb is None:
                return self.vals[(stock_id, self.lastvaliddays[stock_id])]
            else:
                # print(f"stock_id={stock_id}, self.lastvaliddays[stock_id]={self.lastvaliddays[stock_id]}")
                return self.dfgb.loc[(stock_id, self.lastvaliddays[stock_id])][self.column]
        except:
            return 0
# class bfs():
#     '''
#     hosts one bf object percolumn being backfilled
#     make sure start date is the actual starting date in the dataframe
#     WARNING: syn_colname is added to df for each column in columns
#     ex.     df=of.getdf()
#             bs=of.bfs(['near_price','far_price'], df)
#             df = df.apply(bs.backfill, axis=1)
#     '''
#     def __init__(self,columns, df):
#         self.date_id_min=df.date_id.min()
#         self.a=autocol(columns, df)       
#         self.bfs=[bf(col,self.date_id_min) for col in columns]
        
#     def backfill(self, x):
#         for b in self.bfs:
#             x=b.backfill(x)
#         return x
        
#     def remove_syn_columns(self,df):
#         self.a.remove_syn_columns(df)
    
#     def _debuginfo(self):
#         for b in self.bfs:
#             print(f"col={b.column}")
#             print(f"  b.lastvaliddays={b.lastvaliddays}")
#             print(f"  b.vals={b.vals}")
        
       
# class bf():
#     '''
#     backfills NaNs in column with last value registered in that column (likely from the day before)

#     '''    
#     def __init__(self,column,date_id_min=0):
#         '''
#         column: column to operate on
#         date_id_min: initial start_date of df
#         '''
#         self.column=column  #column to operate on
#         self.syn_column='syn_'+column  #mark as 1 if synthetic value generated
#         self.lastvaliddays={}
#         # self.days=set() #days that are present
#         self.vals={}  #cache of column val for (stock_id,date_id)
#         self.date_id_min=date_id_min
        
#     def backfill(self, x):
#         '''
#         x: row of data
#         '''
#         #if null then continue
#         if(pd.isnull(x[self.column])):
#             x[self.column] = self._getlastvalidvalue(x.stock_id, x.date_id)
#             x[self.syn_column]=1
#         else:  
#             self.lastvaliddays[x.stock_id]=x.date_id
#             self.vals[(x.stock_id,x.date_id)]=x[self.column]
#         return x
    
#     def _getlastvalidvalue(self, stock_id, date_id):
#         try:
#             return self.vals[(stock_id, self.lastvaliddays[stock_id])]
#         except:
#             return 0

# does not work
#class last_val_dd(last_val):
#     '''
#     backfills NaNs in near_price and far_price with last value
#     dfgb- groupby opject with summary stats for each stock
#     ex. df.groupby(['stock_id','date_id']).median()
#     '''    
#     def __init__(self):
#         super().__init__()
#         self.days=set()
        
#     def backfill(self, x):
#         #save current config
#         if(pd.notnull(x.far_price)):
#             self.days.add(x.date_id)
#         self.prices[(x.stock_id,x.date_id)]=(x.far_price, x.near_price)
#         return super().backfill(x)
    
#     def _lookup(self, stock_id, yesterday):
#         if(yesterday < 0):
#             return 0,0

#         if (yesterday not in self.days):
#             yesterday=list(self.days).pop()
            
#         row = self.prices[(stock_id, (yesterday))]
#         #cache for later if not null
#         if( pd.notnull(row[0]) & pd.notnull(row[1])):
#             self.prices[(stock_id,yesterday)] = (row[0], row[1])
#         return row[0], row[1]
# from abc import ABC, abstractmethod 
# class last_val(ABC):
#     '''
#     caches last values found, cuts down on expensive lookups in dfgb
#     '''

#     def __init__(self):
#         self.prices = {}
#         self.hit=0
#         self.miss=0
#         self.total=0
        
#     def backfill(self, x):
#         self.total+=1
#         if pd.isnull(x.far_price):
#             x.synthetic_far_price=1

#             #get both far_and near price, near price used in next if
#             x.far_price,near_price=self._getlast(x.stock_id, x.date_id)
 
#         if pd.isnull(x.near_price):
#             x.synthetic_near_price=1
#             x.near_price=near_price
#         return x
        
#     def _getlast(self, stock_id, date_id):
#         #look for yesterdays date
#         yesterday=date_id-1
#         try:
#             #see if in cache
#             far_price, near_price=self.prices[(stock_id,yesterday)]
#             self.hit+=1
#         except:
#             #keyerror
#             self.miss+=1
#             # print(yesterday)

#             #find
#             far_price, near_price = self._lookup(stock_id, yesterday)
#         return far_price, near_price 
    
#     @abstractmethod
#     def _lookup(self, stock_id, date_id):
#         pass

         
#     def __repr__(self):
#         return (f"total processed={self.total}, totalNaNs infered={self.hit+ self.miss}; cache hits={self.hit} cache misses={self.miss}")


# class last_val_gb(last_val):

#     def __init__(self, dfgb):
#         super().__init__()
#         self.dfgb = dfgb
        
#         #replace the nulls with 0
#         self.dfgb.fillna(0,inplace=True)
#         self.days=list(self.dfgb.index.get_level_values(1).unique() )

#     def _lookup(self, stock_id, date_id):
#         if(date_id <= 0):
#             return 0,0
        
#         if(date_id not in self.days):
#             #find closest earlier match
#             date_id=self.days[len([x for x in (self.days - date_id) if x<0])-1]

#         #get row
#         row = self.dfgb.loc[(stock_id, (date_id))]
#         #cache for later if not null
#         if( pd.notnull(row.far_price) & pd.notnull(row.near_price)):
#             self.prices[(stock_id,date_id)] = (row.far_price, row.near_price)
#             return row.far_price, row.near_price
#         else:
#             return 0,0
