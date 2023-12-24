from hc_nlp.pipeline import ThesaurusMatcher, EntityFilter, MapEntityTypes
from hc_nlp.spacy_helpers import display_ner_annotations

import spacy
from spacy import displacy
from spacy.matcher import Matcher
import pandas as pd

# https://github.com/LinkedPasts/LaNC-workshop/tree/main/heritageconnector
nlp = spacy.load('en_core_web_sm')
thesaurus_matcher = ThesaurusMatcher(nlp, thesaurus_path='./outputs/hc_bm_gazetteer.jsonl',  case_sensitive=False)

csv_file = './data/bm-dataset-298_cut.csv'
df = pd.read_csv(csv_file)

nlp_1 = spacy.load("en_core_web_sm")
nlp_1.add_pipe(thesaurus_matcher, before='ner')

nlp_2 = spacy.load("en_core_web_sm")
nlp_2.add_pipe(thesaurus_matcher, after='ner')

# for the third pipeline we create a new instance of ThesaurusMatcher with the extra argument overwrite_ents=True
nlp_3 = spacy.load("en_core_web_sm")
thesaurus_matcher_overwrite = ThesaurusMatcher(nlp, thesaurus_path='./outputs/hc_bm_gazetteer.jsonl',  case_sensitive=False, overwrite_ents=True)
nlp_3.add_pipe(thesaurus_matcher_overwrite, after='ner')

# we also add an extra component which filters out some obvious false positives
entityfilter = EntityFilter(ent_labels_ignore=["DATE", "CARDINAL"])

for pipe in nlp, nlp_1, nlp_2, nlp_3:
    pipe.add_pipe(entityfilter, last=True)

def extract_place_names(this_nlp, row, text_col_name, places_col_name):
    text_col_cell = row.loc[text_col_name]
    places_col_cell = row.loc[places_col_name]

    if not isinstance(text_col_cell, str):
        # print(f"The variable is not a string. Breaking.")
        if isinstance(places_col_cell, list):
            return places_col_cell
        else:
            print(places_col_cell)
            return []

    doc = this_nlp(text_col_cell)
    matcher = Matcher(this_nlp.vocab)

    # Define a pattern for location entities
    pattern = [{"ENT_TYPE": "GPE"}]
    # Doesn't work, comes back with Venus # pattern = [{"ENT_TYPE": "LOC"}]

    matcher.add("Location", [pattern])
    matches = matcher(doc)
    places = [doc[start:end].text for match_id, start, end in matches]

    # Checks
    # if row.loc['Unique ID'] == 'P_Y-2-133':
    #     print(places)
    # if text_col_name == 'Comment':
    #     items_not_in_list2 = set(places) - set(places_col_cell)
    #     result_list = list(items_not_in_list2)
    #     print(result_list)
    #     print(row.loc['Unique ID'])

    if isinstance(places_col_cell, list) and isinstance(places, list):
        places_col_cell.extend(places)
    else:
        print(places_col_cell)
        print(places)

    return list(set(places_col_cell))

text_col_names = ['Physical description', 'Object History Note / Acquisition Note', 'Comment', 'Inscription - Quoted']

nlps = [nlp, nlp_1, nlp_2, nlp_3]

for i, this_nlp in enumerate(nlps):
    places_col_name = f'place_names_nlp_{i}'
    df[places_col_name] = [[] for _ in range(len(df))]

for i, this_nlp in enumerate(nlps):
    places_col_name = f'place_names_nlp_{i}'
    for text_col_name in text_col_names:
        places_col_name_apply = df.apply(lambda row: extract_place_names(this_nlp, row, text_col_name, places_col_name), axis=1)
        df[places_col_name] = places_col_name_apply

df.to_csv('./outputs/bm-dataset-298_out.csv', index=False)
