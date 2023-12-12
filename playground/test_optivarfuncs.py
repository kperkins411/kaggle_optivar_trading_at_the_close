import pandas as pd
import numpy as np
import itertools
import optivarfuncs as of
import unittest
 
class TestOptivar(unittest.TestCase):
        # def setUp(self):
        #     self.df=getdf()

        def test_df(self):
            df=of.getdf()
            self.assertEqual(len(df), 36, f"DF should be 36, is {len(df)}")
            self.assertEqual(df.isnull().sum().sum(),36, f"DF should have 36 NaNs has {df.isnull().sum().sum() }")

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
    
    
    