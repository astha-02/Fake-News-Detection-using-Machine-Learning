import pandas as pd
import numpy as np
import csv
'''
    def makeDataset(file,label):
        with open(file) as data:
            content = data.readlines()
            content = [x.strip('\n') for x in content]
        print(content)
'''
def makeDataset(file,label):
    res=np.array([])
    line1=True
    with open(file) as data:
        for line in data:
            if line1:
                line1=False
                
                
                res= np.array([[line,label]])
                
              
                
            else:
                
               
                
                newrow = np.array([[line, label]])
                res = np.append(res, newrow, axis=0)
                
    return res                
                
    
    
      

if __name__ == '__main__':
    
    import pandas as pd

    my_df=pd.DataFrame(makeDataset('./fake-news',1))
    df=pd.DataFrame(makeDataset('./real-news',0))
    df_merge=pd.concat([my_df,df],ignore_index=True)
    df_merge=df_merge.sample(frac=1).reset_index(drop=True)
    df_merge.to_csv('dataset.csv',index=False,header=('Headline','Label'))

