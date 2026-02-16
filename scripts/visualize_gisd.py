import pandas as pd
import altair as alt
import json
from urllib.request import urlopen

# Load the GISD data
gisd_data = pd.read_csv('/Users/AwotoroE-Dev/Desktop/AMR_rough/datasets/GISD_Bundesland_Updated.csv')

# Load the GeoJSON data
geojson_url = 'https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/refs/heads/main/2_bundeslaender/1_sehr_hoch.geo.json'
with urlopen(geojson_url) as response:
    bundesland_geojson = json.load(response)

# Ensure the GeoJSON properties match the GISD data
for feature in bundesland_geojson['features']:
    state_name = feature['properties']['name']
    gisd_score = gisd_data[gisd_data['state'] == state_name]['gisd_score'].mean()
    feature['properties']['gisd_score'] = gisd_score

# Define the color scale
color_scale = alt.Scale(
    domain=[0, 0.25, 0.5, 0.75, 1],
    range=['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']
)

# Create the map visualization
map_chart = alt.Chart(alt.Data(values=bundesland_geojson['features'])).mark_geoshape().encode(
    color=alt.Color('properties.gisd_score:Q', scale=color_scale),
    tooltip=['properties.name:N', 'properties.gisd_score:Q']
).project(
    type='identity', reflectY=True
).properties(
    width=800,
    height=900,
    title='GISD Scores by Bundesland'
)

# Save the chart
map_chart.save('/Users/AwotoroE-Dev/Desktop/AMR_rough/visualizations/gisd_map.html')
