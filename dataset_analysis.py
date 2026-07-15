import pandas as pd

path = 'credit.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('\ncolumns:')
print(df.columns.tolist())
print('\nmissing values:')
print(df.isna().sum())
print('\nvalue counts for target:')
print(df['Approved'].value_counts(dropna=False))
print('\nfeature sample:')
print(df.head(10).to_string(index=False))
