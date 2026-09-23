# Global Urban Thermal Comfort Explorer

Interactive WebGIS for exploring historical urban thermal comfort patterns and migration strategies across 120 global cities.

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://global-urban-thermal-comfort-zxxax69fft5iuh5ucwml6a.streamlit.app/)

[Launch the live Streamlit application](https://global-urban-thermal-comfort-zxxax69fft5iuh5ucwml6a.streamlit.app/)

## Overview

The application presents a global, map-first view of monthly thermal comfort from 1995 to 2024. It combines comfortable-day classification, monthly mean UTCI, long-term city trends, and migration-strategy visualization in a single interactive interface.

## Key Features

- Interactive global choropleth map for 120 cities
- Historical month slider covering 1995–2024
- Comfortable-day classification by city and month
- Monthly mean UTCI in city tooltips
- City selection directly from the interactive map
- 30-year monthly UTCI trend exploration
- 30-year comfortable-day trend exploration
- Migration route visualization with interactive map layers
- Spatial and temporal exploration through Streamlit and PyDeck

## Tech Stack

- Python
- Streamlit
- PyDeck / deck.gl
- Pandas
- GeoJSON
- Shapely
- Git and GitHub

## Data Availability

The research datasets used by the deployed application are not distributed through this public repository.

This repository provides the application source code and project documentation for portfolio and reproducibility-oriented demonstration. The interactive results can be explored through the live application.

## Project Structure

```text
global-urban-thermal-comfort-showcase/
├── Home.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── README.md
```

## Deployment

The live application is deployed separately using Streamlit Community Cloud.

[Open the deployed application](https://global-urban-thermal-comfort-zxxax69fft5iuh5ucwml6a.streamlit.app/)
