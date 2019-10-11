import forecast_ARLR
from multiprocessing import Pool
import pdb
national = 4
region = 4
state = 4
regions = {"national": national, "1": region, "2": region, "3": region, "4": region, "5": region, "6": region, "7": region, "8": region, "9": region, "10": region, "Alabama": state, "Alaska": state, "Arizona": state, "Arkansas": state,
"California": state, "Colorado": state, "Connecticut": state, "Delaware": state, "Georgia": state, "Hawaii": state, "Idaho": state, "Illinois": state, "Indiana": state, "Iowa": state, "Kansas": state, "Kentucky": state,
"Louisiana": state, "Maine": state, "Maryland": state, "Massachusetts": state, "Michigan": state, "Minnesota": state, "Mississippi": state, "Missouri": state, "Montana": state, "Nebraska": state, "Nevada": state, "New Hampshire": state,
"New Jersey": state, "New Mexico": state, "New York": state, "North Carolina": state, "North Dakota": state, "Ohio": state, "Oklahoma": state, "Oregon": state, "Pennsylvania": state, "Rhode Island": state, "South Carolina": state,
"South Dakota": state, "Tennessee": state, "Texas": state, "Utah": state, "Vermont": state, "Virginia": state, "Washington": state, "West Virginia": state, "Wisconsin": state, "Wyoming": state}

for key in regions.keys():
    pdb.set_trace()
    forecast_ARLR.main({"REGION": key, "TARGET": "ili", "STARTDATE": "2017EW40", "ENDDATE": "2016EW39"})
