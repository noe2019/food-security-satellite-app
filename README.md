# Food Security Monitoring with Satellite Data – KwaZulu-Natal

Tracks the effects of drought and COVID-19 lockdowns on smallholder and commercial agriculture in KwaZulu-Natal using vegetation indices (NDVI) and rainfall.

**Live app:** https://noe-food-security-c984fe0b9737.herokuapp.com/
**Portfolio:** https://noe2019.github.io/DA-Portfolio-Website/

## Features
- Monthly NDVI time series by district and farming system
- NDVI anomalies vs. 2015–2022 climatology
- District anomaly map
- Regression estimating drought and lockdown effects

## Tech stack
Python, Remote sensing (NDVI), scikit-learn, Streamlit, Plotly

## Data
All data in this app is **simulated** (generated in `data.py` with a fixed random seed) to reproduce the structure of the original project without exposing real or confidential records.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Heroku (stack `heroku-24`, Python 3.12) using the `Procfile` in this repository.

---
Author: Noé Carème Fouotsa Manfouo
