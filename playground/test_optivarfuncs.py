import pandas as pd
import numpy as np
import itertools
import optivarfuncs as of
import unittest
from gc import collect;

 
class TestOptivar(unittest.TestCase):
        # def setUp(self):
        #     self.df=getdf()

        def test_df(self):
            df=of.getdf()
            self.assertEqual(len(df), 36, f"DF should be 36, is {len(df)}")
            self.assertEqual(df.isnull().sum().sum(),36, f"DF should have 36 NaNs has {df.isnull().sum().sum() }")

        #split datasets
        def test_2splits(self):
            df = of.getdf(numb_days = 10, drop_days= [2,3])

            X_train, X_val, y_train, y_val=of.get2_DatasetAndTarget(df, dep_var='target', val_size=0.2, verbose=False)
            self.assertEqual(len(X_train), 72)
            self.assertEqual(len(X_val), 24)
            self.assertEqual(len(y_train), 72)
            self.assertEqual(len(y_val), 24)

        #split datasets
        def test_3splits(self):
            df = of.getdf(numb_days = 10, drop_days= [2,3])
            X_train, X_val, X_tst, y_train, y_val, y_tst = of.get3_DatasetAndTarget(df, dep_var='target', val_size=0.1, test_size=0.2, verbose=False)
            self.assertEqual(len(X_train), 60)
            self.assertEqual(len(X_val), 12)
            self.assertEqual(len(X_tst), 24)
            self.assertEqual(len(y_train), 60)
            self.assertEqual(len(y_val), 12)
            self.assertEqual(len(y_tst), 24)


        def test_bfs_dfgb(self):
            df=of.getdf()

            # dfgb = dfc.groupby(['stock_id','date_id']).median()
            dfgb = df.groupby(['stock_id','date_id']).median()
            # print(dfgb)
            # print(dfgb.index)
            # print(dfgb.loc[(0,0.0)])

            bs=of.bfs(['near_price','far_price'],df,dfgb)
            df = df.apply(bs.backfill, axis=1)
            # print(df)

            #the first set of NaNs should be 0
            val = df[(df.stock_id==1) & (df.date_id==0) & (df.seconds_in_bucket<=1)].near_price.sum()
            self.assertEqual(val,0, f"first 2 rows fhould be 0 is {val}")

            #the second set should be 25
            val = df[(df.stock_id==1) & (df.date_id==3) & (df.seconds_in_bucket<=1)].near_price.sum()/2
            # print(df)
            self.assertEqual(val,66)

            #should have no nulls left
            self.assertEqual(df.isnull().sum().sum(),0)
            # print(lv) 

        def test_bfs_autocol(self):
            df=of.getdf()

            cols=['near_price','far_price']
            bs=of.bfs(cols,df)

            #make sure columns are there
            for c in cols:
                self.assertTrue(('syn_'+c in df.columns))

            #make sure columns removed
            bs.remove_syn_columns(df)
            for c in cols:
                self.assertFalse(('syn_'+c in df.columns))

        def test_bfs(self):
            df=of.getdf()

            bs=of.bfs(['near_price','far_price'],df)
            # bs=of.bfs(['near_price'])
            df = df.apply(bs.backfill, axis=1)

            # print(df )
            # print(bs._debuginfo())
            
            #the first set of NaNs should be 0
            val = df[(df.stock_id==1) & (df.date_id==0) & (df.seconds_in_bucket<=1)].near_price.sum()
            self.assertEqual(val,0, f"first 2 rows fhould be 0 is {val}")

            #also should have set syn_column to 1
            val = df[(df.stock_id==1) & (df.date_id==0) & (df.seconds_in_bucket<=1)].syn_near_price.mean()
            self.assertEqual(val,1, f"first 2 rows syn_near_price should be 1 is {val}")
        
            #the second set should be 25
            val = df[(df.stock_id==1) & (df.date_id==3) & (df.seconds_in_bucket<=1)].near_price.sum()/2
            self.assertEqual(val,71)

            val = df[(df.stock_id==1) & (df.date_id==0) & (df.seconds_in_bucket<=1)].syn_far_price.mean()
            self.assertEqual(val,1)
          
            #should have no nulls left
            self.assertEqual(df.isnull().sum().sum(),0)
            # print(df)

        def test_bfs_concat(self):
            df=of.getdf()
            
            bs=of.bfs(['near_price','far_price'],df)
            df = df.apply(bs.backfill, axis=1)
        
            df2=of.getdf()
            df2["date_id"] = df2["date_id"] + 10 #make them in the future
            bs.doautocol(df2)  #add the syn_ columns

            df2=df2.apply(bs.backfill,axis=1)

            df3=pd.concat([df,df2])

            self.assertEqual(len(df3),len(df)+len(df2))

            op = df3.loc[((df3['stock_id']==0) & (df3['date_id']==13) & (df3['seconds_in_bucket']==3)),'near_price'].values[0]
            op1 = df3.loc[((df3['stock_id']==0) & (df3['date_id']==14) & (df3['seconds_in_bucket']==0)),'near_price'].values[0]
            self.assertEqual(op,op1)

        def test_bfs_concat1(self):
            df=of.getdf()
            
            bs=of.bfs(['near_price','far_price'],df)
            df = df.apply(bs.backfill, axis=1)
        
            df2=of.getdf()
            df2["date_id"] = df2["date_id"] + 10 #make them in the future
            bs.doautocol(df2)  #add the syn_ columns

            df3=pd.concat([df,df2])
            df3[-len(df2):]=df3[-len(df2):].apply(bs.backfill,axis=1)

            print(len(df3))
            self.assertEqual(len(df3),len(df)+len(df2))

            op = df3.loc[((df3['stock_id']==0) & (df3['date_id']==13) & (df3['seconds_in_bucket']==3)),'near_price'].values[0]
            op1 = df3.loc[((df3['stock_id']==0) & (df3['date_id']==14) & (df3['seconds_in_bucket']==0)),'near_price'].values[0]
            self.assertEqual(op,op1)
            print(df3)

        def test_cache(self):
            df=of.getdf()
            df2=of.getdf()
            df2["date_id"] = df2["date_id"] + 10 #make them in the future
            df3=pd.concat([df,df2])
            
            print("start")
            # print(df3)
            df3=df3[-(len(df2)+10):]
            del df2
            df2=None
            collect()
            
            print (df3)

            #TODO proper testing here
 
        
            

    
        # def test_last_val_dd(self):
        #     df=of.getdf()

        #     lv=of.last_val_dd()
        #     df = df.apply(lv.backfill, axis=1)

        #     #the first set of NaNs should be 0
        #     val = df[(df.stock_id==1) & (df.date_id==0) & (df.seconds_in_bucket<=1)].near_price.sum()
        #     self.assertEqual(val,0, f"first 2 rows fhould be 0 is {val}")

        #     #the second set should be 25
        #     val = df[(df.stock_id==1) & (df.date_id==3) & (df.seconds_in_bucket<=1)].near_price.sum()/2
        #     self.assertEqual(val,31)

        #     #should have no nulls left
        #     print(df)
        #     print(lv.prices)
        #     print(lv.days)
        #     self.assertEqual(df.isnull().sum().sum(),0)
       # def test_last_val_gb(self):
        #     df=of.getdf()

        #     # dfgb = dfc.groupby(['stock_id','date_id']).median()
        #     dfgb = df.groupby(['stock_id','date_id']).median()

        #     lv=of.last_val_gb(dfgb)
        #     df = df.apply(lv.backfill, axis=1)

        #     #the first set of NaNs should be 0
        #     val = df[(df.stock_id==1) & (df.date_id==0) & (df.seconds_in_bucket<=1)].near_price.sum()
        #     self.assertEqual(val,0, f"first 2 rows fhould be 0 is {val}")

        #     #the second set should be 25
        #     val = df[(df.stock_id==1) & (df.date_id==3) & (df.seconds_in_bucket<=1)].near_price.sum()/2
        #     # print(df)
        #     self.assertEqual(val,66)

        #     #should have no nulls left
        #     self.assertEqual(df.isnull().sum().sum(),0)
        #     # print(lv)
            

if __name__=="__main__":
    unittest.main()
    
    
    