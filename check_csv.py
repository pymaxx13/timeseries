import pandas as pd
df = pd.read_csv(r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\script-darts\Solar_EngressionTS_uniform_metrics.csv')
if 'Max Steps/Epochs' in df.columns:
    print(df[['Model', 'Max Steps/Epochs']])
elif 'N_Epochs' in df.columns:
    print(df[['Model', 'N_Epochs']])
else:
    print(df.columns)
