import numpy as np
import pandas as pd
import math


def sig(dav):
    a = 1859*(10**-6)
    b = 1437
    sig = 140/(1+ math.exp(a(dav-b))) + 25
    return sig

data = np.load('dav_values.npy')

df = pd.DataFrame(data)
# df.to_excel('output.xlsx', engine='openpyxl', index=False)




# Replace 'your_file_path.xlsx' with the actual path to your Excel file
excel_file_path = 'extracted_ib_tracks.xlsx'


# Load the Excel file into a pandas DataFrame
df_ib = pd.read_excel(excel_file_path)

# Check if 'usa_wind' column exists in the DataFrame
if 'USA_WIND' in df_ib.columns:
    usa_wind_column = df_ib['USA_WIND']
    print(usa_wind_column)

# for column in df:
#     for i in range(len(column)):
        
print(sig(2314.16))