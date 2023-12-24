import pandas as pd
import json

bm_df = pd.read_csv('./data/sloane_places_nov_2023-1.csv', low_memory=False)

toponyms = bm_df['Place Name'].str.title().unique().tolist()
aliases = bm_df['Aliases'].apply(lambda i: i.split(";")[0].split(",")[0].strip()).unique().tolist()

places = list(set(toponyms + aliases))

places_dict = [{"label": "GPE", "pattern": place} for place in places]

with open('hc_bm_gazetteer.jsonl', 'w') as f:
    for item in places_dict:
        json.dump(item, f)
        f.write("\n")
