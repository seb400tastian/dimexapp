#!/bin/bash
cd /home/site/wwwroot
streamlit run main.py --server.port 8000 --server.enableCORS false

