import pandas as pd
from config import cfg

file = sorted((cfg.RAW_DATA).glob("*.parquet"))[0]
df = pd.read_parquet(file)

conversation = df.iloc[0]["conversation"]
print(conversation)