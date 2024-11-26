#!/bin/bash
# Mover al directorio donde está tu app
cd /home/site/wwwroot
# Ejecutar Streamlit
streamlit run main.py --server.port 8000 --server.enableCORS false
