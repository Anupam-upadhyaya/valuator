import numpy as np
import pandas as pd

final2 = {
    'ConditionRating':[4,5],
    'OriginalPrice':[80000,20000],
    'Vendor':[1,2],
    'Model':[6,3],
    'RAM':[6,6],
    'ROM':[64,32],
    'UsageTime':[0.4,0.1],
    'Warranty':[1,1],
    'DentsRating':[0,0],
    'ScratchesRating':[0,0],
    'CameraWorking':[1,1],
    'ScreenCracks':[0,0],
    'DIsplayWorks':[1,1],
    'PredictedPrice':[79000,20000],
    
}
df = pd.DataFrame(final2)
df.index[1]
df.to_csv('final8.csv')
print(df)
