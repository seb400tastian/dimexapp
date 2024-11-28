
import streamlit as st
import pandas as pd
from datetime import date
from fpdf import FPDF
from datetime import datetime
def cargar_estilos():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Llama a la función para cargar los estilos CSS personalizados
cargar_estilos()
# Simulación de usuarios registrados
usuarios = {
    "user1@example.com": {"password": "password1", "role": "admin"},
    "user2@example.com": {"password": "password2", "role": "call_center"},
}

# Cargar el archivo CSV
try:
    df_mo = pd.read_csv('df_app_fin (1).csv')
    df_mo.columns = df_mo.columns.str.strip()

    if 'Solicitud_id' not in df_mo.columns:
        st.error("No se encontró la columna 'Solicitud_id'.")
        st.stop()
    df_mo['Solicitud_id'] = df_mo['Solicitud_id'].astype(int)

except Exception as e:
    st.error(f"Error al cargar el archivo CSV: {e}")
    st.stop()

# Diccionarios
atraso_dict = {
    3: 'atraso_1_29',
    4: 'atraso_30_59',
    5: 'atraso_60_89',
    6: 'atraso_90_119',
    0: 'atraso_120_149',
    1: 'atraso_150_179',
    2: 'atraso_180_más'
}


estatus_dict = {
    0: 'Cuenta Contenida',
    1: 'Cuenta Deteriorada',
    2: 'Cuenta Regularizada'
}
mensualidades_dict = {
    'atraso_1_29': 1,
    'atraso_30_59': 2,
    'atraso_60_89': 3,
    'atraso_90_119': 4,
    'atraso_120_149': 5,
    'atraso_150_179': 6,
    'atraso_180_más': 7
}
# Inicializar el estado de sesión
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'interacciones' not in st.session_state:
    st.session_state.interacciones = []
if 'mostrar_formulario' not in st.session_state:
    st.session_state.mostrar_formulario = False

def login(email, password):
    """Función para autenticar usuarios"""
    # Verificar si el email existe en el diccionario
    if email in usuarios:
        # Verificar si la contraseña coincide
        if usuarios[email]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.rol = usuarios[email]["role"]  # Guardar el rol del usuario
            return True
    return False

def mostrar_lista_usuarios(filtro_id=None):  # Aseguramos que filtro_id sea un parámetro opcional
    """Muestra la lista de usuarios con filtros, probabilidad de deterioro, nivel de atraso y botones de orden."""
    st.title("Listado de Usuarios")

    # Filtrar por Solicitud ID si se proporciona
    if filtro_id:
        df_filtrado = df_mo[df_mo['Solicitud_id'] == filtro_id]
    else:
        df_filtrado = df_mo.copy()

    # Filtros
    filtro_credito = st.number_input("Línea de Crédito mínima", min_value=0, step=1000, value=0)
    filtro_prob_deterioro = st.selectbox("Probabilidad de Deterioro", ["Todos", "Bajo", "Medio", "Alto"])
    filtro_nivel_atraso = st.selectbox("Nivel de Atraso", ["Todos"] + list(atraso_dict.values()))

    # Aplicar filtros al DataFrame
    if filtro_credito > 0:
        df_filtrado = df_filtrado[df_filtrado['Linea credito'] >= filtro_credito]
    if filtro_prob_deterioro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Prob Dete'] == filtro_prob_deterioro]
    if filtro_nivel_atraso != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Nivel_Atraso_encoded'].map(atraso_dict) == filtro_nivel_atraso]

    # Renombrar columnas para visualización
    df_filtrado.rename(columns={
        'Solicitud_id': 'Solicitud ID',
        'Linea credito': 'Línea de Crédito',
        'Nivel_Atraso_encoded': 'Nivel de Atraso',
        'Prob Dete': 'Probabilidad de Deterioro'
    }, inplace=True)

    # Botones para ordenar
    col1, col2 = st.columns(2)
    with col1:
        ordenar_ascendente = st.button("Ordenar Ascendente")
    with col2:
        ordenar_descendente = st.button("Ordenar Descendente")

    # Ordenar el DataFrame según los botones presionados
    if ordenar_ascendente:
        df_filtrado = df_filtrado.sort_values(by="Línea de Crédito", ascending=True)
    elif ordenar_descendente:
        df_filtrado = df_filtrado.sort_values(by="Línea de Crédito", ascending=False)

    # Verificar si el DataFrame tiene datos antes de la paginación
    if df_filtrado.empty:
        st.info("No hay datos disponibles para los filtros seleccionados.")
        return  # Salir de la función si no hay datos

    # Filas por página (para paginación)
    filas_por_pagina = 10
    total_paginas = max(1, (len(df_filtrado) + filas_por_pagina - 1) // filas_por_pagina)  # Asegurarse de que sea al menos 1
    pagina_actual = st.number_input("Página", min_value=1, max_value=total_paginas, step=1, value=1)

    inicio = (pagina_actual - 1) * filas_por_pagina
    fin = inicio + filas_por_pagina
    df_paginado = df_filtrado.iloc[inicio:fin]

    # Tabla estilizada con colores para Probabilidad de Deterioro
    def renderizar_tabla_html(df):
        table_html = """
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 15px;
                font-family: Arial, sans-serif;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            }
            th {
                background-color: #3b8dbd; /* Azul profesional */
                color: white;
                text-align: left;
                padding: 12px;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #f2f2f2;
                text-align: left;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:nth-child(odd) {
                background-color: #ffffff;
            }
            tr:hover {
                background-color: #f1f1f1; /* Hover suave */
            }
            .rojo {
                color: #e53935;
                font-weight: bold;
            }
            .amarillo {
                color: #fbc02d;
                font-weight: bold;
            }
            .verde {
                color: #43a047;
                font-weight: bold;
            }
        </style>
        <table>
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """
        # Generar los encabezados
        headers = ''.join(f'<th>{col}</th>' for col in df.columns)

        # Generar las filas con colores según `Probabilidad de Deterioro`
        rows = ''
        for _, row in df.iterrows():
            # Asignar colores según el nivel de probabilidad de deterioro
            if row['Probabilidad de Deterioro'] == "Alto":
                prob_class = 'rojo'
            elif row['Probabilidad de Deterioro'] == "Medio":
                prob_class = 'amarillo'
            elif row['Probabilidad de Deterioro'] == "Bajo":
                prob_class = 'verde'
            else:
                prob_class = ''

            rows += (
                f"<tr>"
                + ''.join(
                    f"<td class='{prob_class if col == 'Probabilidad de Deterioro' else ''}'>{val}</td>"
                    for col, val in row.items()
                )
                + "</tr>"
            )

        return table_html.replace("{headers}", headers).replace("{rows}", rows)

    # Renderizar la tabla paginada con opción de selección
    columnas_para_mostrar = ['Solicitud ID', 'Línea de Crédito', 'Nivel de Atraso', 'Probabilidad de Deterioro']
    tabla_html = renderizar_tabla_html(df_paginado[columnas_para_mostrar])
    st.markdown(tabla_html, unsafe_allow_html=True)

    # Botones de selección en mostrar_lista_usuarios
    for _, row in df_paginado.iterrows():
        if st.button(f"Seleccionar {row['Solicitud ID']}", key=row['Solicitud ID']):
            st.session_state["solicitud_seleccionada"] = row['Solicitud ID']
            st.session_state.page = "informacion_usuario"  # Change to the information page



df_mo['Capacidad_Pago'] = df_mo['Capacidad_Pago'].apply(lambda x: f"{x:.2%}")
import streamlit as st
import os
from PIL import Image
IMAGES_FOLDER = 'Fotos'
# Ruta a la carpeta donde están las imágenes descargadas
def obtener_imagen_usuario(solicitud_id):
    """Obtiene una imagen de usuario basada en el ID de solicitud de manera repetitiva."""
    # Verificar si la carpeta existe
    if not os.path.exists(IMAGES_FOLDER):
        st.error(f"La carpeta '{IMAGES_FOLDER}' no existe.")
        return None
    
    # Obtener la lista de archivos de imagen en la carpeta (incluyendo .webp)
    try:
        imagenes = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    except Exception as e:
        st.error(f"Error al listar archivos en la carpeta: {e}")
        return None

    # Verificar si la lista de imágenes está vacía
    if not imagenes:
        st.warning("No se encontraron imágenes en la carpeta especificada.")
        return None

    # Asegurarse de que solicitud_id sea un entero
    try:
        solicitud_id = int(solicitud_id)  # Convertir a entero si no lo es
        index_imagen = solicitud_id % len(imagenes)
        imagen_seleccionada = os.path.join(IMAGES_FOLDER, imagenes[index_imagen])
    except ValueError:
        st.error("El ID de solicitud no es un número válido.")
        return None
    except Exception as e:
        st.error(f"Error al calcular el índice de la imagen: {e}")
        return None

    # Verificar si la imagen realmente existe
    if not os.path.isfile(imagen_seleccionada):
        st.error(f"No se pudo encontrar la imagen: {imagen_seleccionada}")
        return None

    return imagen_seleccionada
def mostrar_lista_usuarios_callcenter(filtro_id=None):
    """Muestra la lista de usuarios con filtros, ordenamiento, paginación y visualización gráfica en pestañas."""
    st.title("Listado de Usuarios Call Center")

    # Filtrar por ID si se proporciona
    if filtro_id:
        df_filtrado = df_mo[df_mo['Solicitud_id'] == filtro_id]
    else:
        df_filtrado = df_mo.copy()

    # Filtros adicionales
    filtro_credito = st.number_input("Línea de Crédito mínima", min_value=0, step=1000, value=0)
    filtro_prob_deterioro = st.selectbox("Probabilidad de Deterioro", ["Todos", "Bajo", "Medio", "Alto"])

    # Aplicar filtros
    if filtro_credito > 0:
        df_filtrado = df_filtrado[df_filtrado['Linea credito'] >= filtro_credito]
    if filtro_prob_deterioro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Prob Dete'] == filtro_prob_deterioro]

    # Calcular los meses adeudados usando el diccionario
    df_filtrado['Mensualidades_Adeudadas'] = df_filtrado['Nivel_Atraso_encoded'].map(
        lambda x: mensualidades_dict.get(atraso_dict.get(x, ''), 0)
    )

    # Renombrar columnas para visualización
    df_filtrado.rename(columns={
        'Solicitud_id': 'ID',
        'Linea credito': 'Línea de Crédito',
        'Mensualidades_Adeudadas': 'Mensualidades Adeudadas',
        'Prob Dete': 'Probabilidad de Deterioro'
    }, inplace=True)

    # Botones para ordenar
    col1, col2 = st.columns(2)
    with col1:
        ordenar_ascendente = st.button("Ordenar Ascendente")
    with col2:
        ordenar_descendente = st.button("Ordenar Descendente")

    # Ordenar según el botón presionado
    if ordenar_ascendente:
        df_filtrado = df_filtrado.sort_values(by="Línea de Crédito", ascending=True)
    elif ordenar_descendente:
        df_filtrado = df_filtrado.sort_values(by="Línea de Crédito", ascending=False)

    # Paginación de resultados
    filas_por_pagina = 10
    total_paginas = (len(df_filtrado) // filas_por_pagina) + 1
    pagina_actual = st.number_input("Página", min_value=1, max_value=total_paginas, step=1, value=1)

    inicio = (pagina_actual - 1) * filas_por_pagina
    fin = inicio + filas_por_pagina
    df_paginado = df_filtrado.iloc[inicio:fin]

    # Mostrar tabla paginada
    st.subheader(f"Usuarios Filtrados (Página {pagina_actual} de {total_paginas})")
    st.table(df_paginado[['ID', 'Línea de Crédito', 'Mensualidades Adeudadas', 'Probabilidad de Deterioro']])

    # Crear pestañas para gráficas
    tabs = st.tabs(["Distribución de Mensualidades", "Distribución de Probabilidad de Deterioro", "Proporción de Probabilidad de Deterioro"])

    # Gráfico: Distribución de Mensualidades Adeudadas
    with tabs[0]:
        st.subheader("Distribución de Mensualidades Adeudadas")
        if not df_filtrado.empty:
            import matplotlib.pyplot as plt

            distribucion = df_filtrado['Mensualidades Adeudadas'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(distribucion.index, distribucion.values, color='skyblue')
            ax.set_title("Distribución de Mensualidades Adeudadas", fontsize=16)
            ax.set_xlabel("Mensualidades Adeudadas", fontsize=12)
            ax.set_ylabel("Cantidad de Usuarios", fontsize=12)

            # Añadir valores encima de las barras
            for i, v in enumerate(distribucion.values):
                ax.text(distribucion.index[i], v + 0.5, str(v), ha='center', va='bottom')

            st.pyplot(fig)
        else:
            st.info("No hay datos para mostrar en la gráfica.")

    # Gráfico: Distribución de Probabilidad de Deterioro
    with tabs[1]:
        st.subheader("Distribución de Probabilidad de Deterioro")
        if not df_filtrado.empty:
            distribucion = df_filtrado['Probabilidad de Deterioro'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#43a047', '#fbc02d', '#e53935']  # Verde, Amarillo, Rojo
            distribucion = distribucion.reindex(["Bajo", "Medio", "Alto"], fill_value=0)
            ax.bar(distribucion.index, distribucion.values, color=colors)
            ax.set_title("Distribución de Probabilidad de Deterioro", fontsize=16)
            ax.set_xlabel("Nivel de Probabilidad", fontsize=12)
            ax.set_ylabel("Cantidad de Usuarios", fontsize=12)

            # Añadir valores encima de las barras
            for i, v in enumerate(distribucion.values):
                ax.text(i, v + 0.5, str(v), ha='center', va='bottom')

            st.pyplot(fig)
        else:
            st.info("No hay datos para mostrar en la gráfica.")

    # Gráfico: Proporción de Probabilidad de Deterioro
    with tabs[2]:
        st.subheader("Proporción de Probabilidad de Deterioro")
        prob_deterioro_data = df_filtrado['Probabilidad de Deterioro'].value_counts()
        if not prob_deterioro_data.empty:
            labels = prob_deterioro_data.index
            sizes = prob_deterioro_data.values

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#43a047', '#fbc02d', '#e53935'])
            ax.set_title("Proporción de Probabilidad de Deterioro", fontsize=14)
            st.pyplot(fig)
        else:
            st.info("No hay información sobre probabilidad de deterioro disponible.")


    


def mostrar_informacion_usuario(solicitud_id):
    """Muestra la información del usuario y un mapa con la ubicación."""
    solicitud_data = df_mo[df_mo['Solicitud_id'] == solicitud_id]
    if solicitud_data.empty:
        st.warning("No se encontró información para este ID.")
        return

    # Mostrar imagen del usuario
    imagen_usuario = obtener_imagen_usuario(solicitud_id)
    if imagen_usuario:
        try:
            # Abrir y redimensionar la imagen
            imagen = Image.open(imagen_usuario)
            
            # Redimensionar la imagen manteniendo la relación de aspecto
            nuevo_ancho = 300
            relacion_aspecto = imagen.width / imagen.height
            nuevo_alto = int(nuevo_ancho / relacion_aspecto)
            imagen = imagen.resize((nuevo_ancho, nuevo_alto))
            
            # Convertir la imagen redimensionada a base64 para usar en HTML
            from io import BytesIO
            import base64

            buffered = BytesIO()
            imagen.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Mostrar la imagen centrada usando HTML y st.markdown
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{img_str}" alt="Imagen del usuario" width="{nuevo_ancho}px">
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error al cargar la imagen: {e}")
    
    # Mostrar la probabilidad de deterioro destacada
    st.markdown("<h2 style='text-align: center;'>Probabilidad de Deterioro</h2>", unsafe_allow_html=True)

    prob_deterioro = solicitud_data['Prob Dete'].values[0]
    color = "#e53935" if prob_deterioro == "Alto" else "#fbc02d" if prob_deterioro == "Medio" else "#43a047"
    st.markdown(
        f"""
        <div style="
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            background-color: {color};
            color: white;
            padding: 10px;
            border-radius: 8px;
        ">
            {prob_deterioro}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mostrar métricas principales
    st.subheader("Información Financiera del Cliente")
    col1, col2, col3 = st.columns(3)
    col1.metric("Línea de Crédito", f"${solicitud_data['Linea credito'].values[0]:,.0f}")
    col2.metric("Pago Mensual", f"${solicitud_data['Pago'].values[0]:,.0f}")
    col3.metric("Plazo en Meses", solicitud_data['Plazo_Meses'].values[0])

    col4, col5, col6 = st.columns(3)
    col4.metric("Capacidad de Pago", solicitud_data['Capacidad_Pago'].values[0])
    col5.metric("Ingreso Mensual", f"${solicitud_data['Ingreso_Bruto'].values[0]:,}")
    col6.metric("Mensualidades Adeudadas", 
                solicitud_data['Nivel_Atraso_encoded'].map(lambda x: mensualidades_dict.get(atraso_dict.get(x, ''), 0)).values[0])

    # Mostrar mapa si hay coordenadas
    st.subheader("Ubicación del Cliente")
    if 'latitude' in solicitud_data.columns and 'longitude' in solicitud_data.columns:
        coordenadas = solicitud_data[['latitude', 'longitude']].dropna()
        if not coordenadas.empty:
            st.map(coordenadas)
        else:
            st.info("No hay coordenadas disponibles para mostrar en el mapa.")
    else:
        st.warning("Las columnas de coordenadas no están disponibles en el DataFrame.")

   
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import json

import matplotlib.pyplot as plt

def generar_graficos_interacciones(df, solicitud_id):
    """Genera gráficos específicos basados en el historial de interacciones para un ID específico."""
    solicitud_data = df[df['Solicitud_id'] == solicitud_id]
    if solicitud_data.empty:
        st.warning("No se encontró información para este ID.")
        return

    # Crear pestañas
    tabs = st.tabs([
        "Gestiones por Canal",
        "Resultados de Interacciones",
        "Promesas de Pago",
        "Línea de Crédito vs Pago Mensual",
        "Probabilidad de Contención"
    ])

    # Gráfico de barras: Total de gestiones por canal
    with tabs[0]:
        st.subheader("Total de Gestiones por Canal")
        canales = ['Tipo_Gestión Puerta a Puerta', 'Tipo_Agencias Especializadas', 'Tipo_Call Center']
        valores = [
            solicitud_data['Tipo_Gestión Puerta a Puerta'].iloc[0],
            solicitud_data['Tipo_Agencias Especializadas'].iloc[0],
            solicitud_data['Tipo_Call Center'].iloc[0]
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        barras = ax.bar(canales, valores, color=['blue', 'green', 'orange'])
        ax.set_ylabel("Número de Gestiones", fontsize=12)
        ax.set_title(f"Gestiones por Canal para ID {solicitud_id}", fontsize=14)

        for bar, valor in zip(barras, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{valor}", ha='center', va='bottom', fontsize=12)

        st.pyplot(fig)

    # Gráfico de pastel: Resultados de las interacciones
    with tabs[1]:
        st.subheader("Resultados de las Interacciones")
        resultados = {
            "Atendió Cliente": solicitud_data['Resultado_Atendió cliente'].iloc[0],
            "Atendió un Tercero": solicitud_data['Resultado_Atendió un tercero'].iloc[0],
            "No Localizado": solicitud_data['Resultado_No localizado'].iloc[0]
        }
        labels = list(resultados.keys())
        sizes = list(resultados.values())

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.set_title(f"Resultados de las Interacciones para ID {solicitud_id}", fontsize=14)
        st.pyplot(fig)

    # Gráfico de barras apiladas: Promesas de pago por canal
    with tabs[2]:
        st.subheader("Promesas de Pago por Canal")
        canales = ['Call Center', 'Puerta a Puerta', 'Agencias']
        promesas_si = [
            solicitud_data['PromesaSiCallCenter'].iloc[0],
            solicitud_data['PromesaSiPuerta'].iloc[0],
            solicitud_data['PromesaSiAgencia'].iloc[0]
        ]
        promesas_no = [
            solicitud_data['PromesaNoCallCenter'].iloc[0],
            solicitud_data['PromesaNoPuerta'].iloc[0],
            solicitud_data['PromesaNoAgencia'].iloc[0]
        ]
        promesas_none = [
            solicitud_data['PromesaNoneCallCenter'].iloc[0],
            solicitud_data['PromesaNonePuerta'].iloc[0],
            solicitud_data['PromesaNoneAgencia'].iloc[0]
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        bar1 = ax.bar(canales, promesas_si, label="Promesa Sí", color='green')
        bar2 = ax.bar(canales, promesas_no, bottom=promesas_si, label="Promesa No", color='red')
        bar3 = ax.bar(canales, promesas_none, bottom=[i + j for i, j in zip(promesas_si, promesas_no)], label="Sin Promesa", color='gray')

        ax.set_ylabel("Número de Promesas", fontsize=12)
        ax.set_title(f"Promesas de Pago por Canal para ID {solicitud_id}", fontsize=14)
        ax.legend()

        for bars, values in zip([bar1, bar2, bar3], [promesas_si, promesas_no, promesas_none]):
            for bar, valor in zip(bars, values):
                if valor > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_y(), f"{valor}", ha='center', va='bottom', fontsize=10)

        st.pyplot(fig)

    # Gráfico de barras: Línea de Crédito vs Pago Mensual
    with tabs[3]:
        st.subheader("Línea de Crédito vs Pago Mensual")
        fig, ax = plt.subplots(figsize=(8, 5))
        categorias = ['Línea de Crédito', 'Pago Mensual']
        valores = [
            solicitud_data['Linea credito'].iloc[0],
            solicitud_data['Pago'].iloc[0]
        ]
        barras = ax.bar(categorias, valores, color=['blue', 'green'])

        for bar, valor in zip(barras, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                    f"${valor:,.0f}", ha='center', va='bottom', fontsize=12)

        ax.set_ylabel("Monto", fontsize=12)
        ax.set_title(f"Comparación de Línea de Crédito y Pago Mensual para ID {solicitud_id}", fontsize=14)
        st.pyplot(fig)

    # Gráfico de barras: Probabilidad de Contención vs Predicción Óptima
    with tabs[4]:
        st.subheader("Probabilidad de Contención y Predicción Óptima")
        fig, ax = plt.subplots(figsize=(8, 5))
        probabilidad_contencion = solicitud_data['Probabilidad contención'].iloc[0]
        prediccion_optima = solicitud_data['Prediccion Optima'].iloc[0]
        categorias = ['Probabilidad Contención', 'Predicción Óptima']
        valores = [probabilidad_contencion, 1]

        barras = ax.bar(categorias, valores, color=['purple', 'orange'])

        ax.text(barras[0].get_x() + barras[0].get_width() / 2, barras[0].get_height() + 0.05,
                f"{probabilidad_contencion:.2f}", ha='center', va='bottom', fontsize=12)
        ax.text(barras[1].get_x() + barras[1].get_width() / 2, barras[1].get_height() + 0.05,
                prediccion_optima, ha='center', va='bottom', fontsize=12)

        ax.set_ylabel("Valor / Categoría", fontsize=12)
        ax.set_title(f"Análisis para ID {solicitud_id}", fontsize=14)
        st.pyplot(fig)




def mostrar_informacion_usuario_callcenter(solicitud_id):
    """Muestra la información detallada del usuario con gráficos adicionales para Call Center."""
    solicitud_data = df_mo[df_mo['Solicitud_id'] == int(solicitud_id)]
    if not solicitud_data.empty:
        st.write("Interacciones previas registradas en el sistema:")
        # Aquí va la lógica para mostrar interacciones previas
    else:
        st.warning("No se encontró información para este ID.")

    # Diccionario de meses adeudados según el nivel de atraso
    meses_adeudados_dict = {
        'atraso_1_29': 1,
        'atraso_30_59': 2,
        'atraso_60_89': 3,
        'atraso_90_119': 4,
        'atraso_120_149': 5,
        'atraso_150_179': 6,
        'atraso_180_más': 7,
    }
    entidades_federativas = {
        "AGS": "Aguascalientes",
        "BCN": "Baja California",
        "BCS": "Baja California Sur",
        "CAM": "Campeche",
        "CHS": "Chiapas",
        "CHI": "Chihuahua",
        "COA": "Coahuila",
        "COL": "Colima",
        "CDMX": "Ciudad de México",
        "DGO": "Durango",
        "EM": "Estado de México",
        "GTO": "Guanajuato",
        "GRO": "Guerrero",
        "HGO": "Hidalgo",
        "JAL": "Jalisco",
        "MICH": "Michoacán",
        "MOR": "Morelos",
        "NAY": "Nayarit",
        "NL": "Nuevo León",
        "OAX": "Oaxaca",
        "PUE": "Puebla",
        "QR": "Quintana Roo",
        "QRO": "Querétaro",
        "SIN": "Sinaloa",
        "SLP": "San Luis Potosí",
        "SON": "Sonora",
        "TAB": "Tabasco",
        "TAM": "Tamaulipas",
        "TLAX": "Tlaxcala",
        "VER": "Veracruz",
        "YUC": "Yucatán",
        "ZAC": "Zacatecas"
    }

    # Determinar meses adeudados
    nivel_atraso_encoded = solicitud_data['Nivel_Atraso_encoded'].values[0]
    nivel_atraso = atraso_dict.get(nivel_atraso_encoded, "Desconocido")
    meses_adeudados = meses_adeudados_dict.get(nivel_atraso, 0)

    # Mostrar imagen del usuario
    imagen_usuario = obtener_imagen_usuario(solicitud_id)
    if imagen_usuario:
        try:
            # Abrir y redimensionar la imagen
            imagen = Image.open(imagen_usuario)
            nuevo_ancho = 300
            relacion_aspecto = imagen.width / imagen.height
            nuevo_alto = int(nuevo_ancho / relacion_aspecto)
            imagen = imagen.resize((nuevo_ancho, nuevo_alto))

            # Convertir la imagen redimensionada a base64 para usar en HTML
            from io import BytesIO
            import base64
            buffered = BytesIO()
            imagen.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Mostrar la imagen centrada usando HTML y st.markdown
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{img_str}" alt="Imagen del usuario" width="{nuevo_ancho}px">
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error al cargar la imagen: {e}")

    # Mostrar Probabilidad de Deterioro destacada
    prob_deterioro = solicitud_data['Prob Dete'].values[0]
    color = "#e53935" if prob_deterioro == "Alto" else "#fbc02d" if prob_deterioro == "Medio" else "#43a047"
    st.markdown(
        f"""
        <div style="
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            background-color: {color};
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
        ">
            Probabilidad de Deterioro: {prob_deterioro}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mostrar métricas principales
    st.subheader("Información Financiera del Cliente")
    col1, col2, col3 = st.columns(3)
    col1.metric("Línea de Crédito", f"${solicitud_data['Linea credito'].values[0]:,.0f}")
    col2.metric("Pago Mensual", f"${solicitud_data['Pago'].values[0]:,.0f}")
    col3.metric("Plazo en Meses", solicitud_data['Plazo_Meses'].values[0])

    col4, col5, col6 = st.columns(3)
    col4.metric("Capacidad de Pago", f"{solicitud_data['Capacidad_Pago'].values[0]}")
    col5.metric("Ingreso Mensual", f"${solicitud_data['Ingreso_Bruto'].values[0]:,.0f}")
    col6.metric("Meses Adeudados", f"{meses_adeudados} mes(es)")

    # Mostrar entidad federativa
    st.metric("Entidad Federativa", solicitud_data["Entidad_federativa"].map(entidades_federativas).values[0])

    # Gráficos adicionales
    st.subheader("Análisis Gráfico del Usuario")
    generar_graficos_interacciones(df_mo, solicitud_id)


    
def mostrar_historial_interacciones_callcenter(solicitud_id):
    """Muestra el historial de interacciones específicas para Call Center de un ID"""
    st.subheader(f"Historial de Interacciones para ID: {solicitud_id}")
    
    # Filtrar los datos para el ID proporcionado
    solicitud_data = df_mo[df_mo['Solicitud_id'] == solicitud_id]
    if not solicitud_data.empty:
        st.write("Interacciones previas registradas en el sistema:")

        # Columnas requeridas específicas para Call Center
        required_columns = [
            'Tipo_Call Center', 'Resultado_Atendió cliente',
            'Promesa_Sí', 'Promesa_No'
        ]
        
        if all(col in solicitud_data.columns for col in required_columns):
            for _, row in solicitud_data.iterrows():
                col1, col2, col3, col4 = st.columns(4)
                
                # Mostrar las columnas relevantes para Call Center
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            text-align: center; 
                            display: flex; 
                            flex-direction: column; 
                            justify-content: space-between; 
                            align-items: center; 
                            height: 100px;
                        ">
                            <div style="font-size: 16px;">Gestión Call Center</div>
                            <div style="font-size: 24px; font-weight: bold; margin-top: auto;">{row['Tipo_Call Center']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div style="
                            text-align: center; 
                            display: flex; 
                            flex-direction: column; 
                            justify-content: space-between; 
                            align-items: center; 
                            height: 100px;
                        ">
                            <div style="font-size: 16px;">Resultado Atendió Cliente</div>
                            <div style="font-size: 24px; font-weight: bold; margin-top: auto;">{row['Resultado_Atendió cliente']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col3:
                    st.markdown(
                        f"""
                        <div style="
                            text-align: center; 
                            display: flex; 
                            flex-direction: column; 
                            justify-content: space-between; 
                            align-items: center; 
                            height: 100px;
                        ">
                            <div style="font-size: 16px;">Promesa</div>
                            <div style="font-size: 24px; font-weight: bold; margin-top: auto;">{row['Promesa_Sí']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col4:
                    st.markdown(
                        f"""
                        <div style="
                            text-align: center; 
                            display: flex; 
                            flex-direction: column; 
                            justify-content: space-between; 
                            align-items: center; 
                            height: 100px;
                        ">
                            <div style="font-size: 16px;">No hubo promesa</div>
                            <div style="font-size: 24px; font-weight: bold; margin-top: auto;">{row['Promesa_No']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.write("---")
        else:
            st.warning("Las columnas necesarias no están presentes en el archivo CSV.")
    
    # Mostrar interacciones creadas por el usuario (en la sesión actual)
    st.write("Interacciones creadas por el usuario:")
    interacciones_filtradas = [i for i in st.session_state.interacciones if i["Solicitud_id"] == solicitud_id]
    
    if interacciones_filtradas:
        df_interacciones = pd.DataFrame(interacciones_filtradas)
        st.table(df_interacciones)
    else:
        st.info("No hay interacciones creadas para este ID.")

    # Botón para crear una nueva interacción
    if st.button("Crear nueva interacción (Call Center)"):
        st.session_state.mostrar_formulario_callcenter = True

    # Mostrar el formulario de creación de interacción si se ha activado
    if st.session_state.get('mostrar_formulario_callcenter', False):
        crear_interaccion(solicitud_id)

def mostrar_historial_interacciones(solicitud_id):
    solicitud_id = int(solicitud_id)
    """Muestra el historial de interacciones para un ID específico y permite crear nuevas interacciones"""
    st.subheader(f"Historial de Interacciones para ID: {solicitud_id}")
    
    solicitud_data = df_mo[df_mo['Solicitud_id'] == solicitud_id]
    if not solicitud_data.empty:
        st.write("Interacciones previas registradas en el sistema:")

        required_columns = [
            'Tipo_Gestión Puerta a Puerta', 'Tipo_Call Center', 
            'Tipo_Agencias Especializadas', 'Resultado_Atendió cliente',
            'Promesa_Sí', 'Promesa_No'
        ]
        
        if all(col in solicitud_data.columns for col in required_columns):
            for _, row in solicitud_data.iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                # Mostrar encabezados y valores con flexbox para una alineación perfecta
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            display: flex; 
                            flex-direction: column; 
                            align-items: center; 
                            justify-content: space-between; 
                            height: 120px; 
                            text-align: center;
                        ">
                            <span style="font-size: 16px; color: #555;">Gestión Puerta a Puerta</span>
                            <span style="font-size: 28px; font-weight: bold; color: #000;">{row['Tipo_Gestión Puerta a Puerta']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div style="
                            display: flex; 
                            flex-direction: column; 
                            align-items: center; 
                            justify-content: space-between; 
                            height: 120px; 
                            text-align: center;
                        ">
                            <span style="font-size: 16px; color: #555;">Gestión Call Center</span>
                            <span style="font-size: 28px; font-weight: bold; color: #000;">{row['Tipo_Call Center']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col3:
                    st.markdown(
                        f"""
                        <div style="
                            display: flex; 
                            flex-direction: column; 
                            align-items: center; 
                            justify-content: space-between; 
                            height: 120px; 
                            text-align: center;
                        ">
                            <span style="font-size: 16px; color: #555;">Gestión Agencias Especializadas</span>
                            <span style="font-size: 28px; font-weight: bold; color: #000;">{row['Tipo_Agencias Especializadas']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col4:
                    st.markdown(
                        f"""
                        <div style="
                            display: flex; 
                            flex-direction: column; 
                            align-items: center; 
                            justify-content: space-between; 
                            height: 120px; 
                            text-align: center;
                        ">
                            <span style="font-size: 16px; color: #555;">Resultado Atendió Cliente</span>
                            <span style="font-size: 28px; font-weight: bold; color: #000;">{row['Resultado_Atendió cliente']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col5:
                    st.markdown(
                        f"""
                        <div style="
                            display: flex; 
                            flex-direction: column; 
                            align-items: center; 
                            justify-content: space-between; 
                            height: 120px; 
                            text-align: center;
                        ">
                            <span style="font-size: 16px; color: #555;">Promesa</span>
                            <span style="font-size: 28px; font-weight: bold; color: #000;">{row['Promesa_Sí']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col6:
                    st.markdown(
                        f"""
                        <div style="
                            display: flex; 
                            flex-direction: column; 
                            align-items: center; 
                            justify-content: space-between; 
                            height: 120px; 
                            text-align: center;
                        ">
                            <span style="font-size: 16px; color: #555;">No hubo promesa</span>
                            <span style="font-size: 28px; font-weight: bold; color: #000;">{row['Promesa_No']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.write("---")
        else:
            st.warning("Las columnas necesarias no están presentes en el archivo CSV.")
    
    # Mostrar interacciones creadas por el usuario
    st.write("Interacciones creadas por el usuario:")
    interacciones_filtradas = [i for i in st.session_state.interacciones if i["Solicitud_id"] == solicitud_id]
    
    if interacciones_filtradas:
        df_interacciones = pd.DataFrame(interacciones_filtradas)
        st.table(df_interacciones)
    else:
        st.info("No hay interacciones creadas para este ID.")

    if st.button("Crear nueva interacción"):
        st.session_state.mostrar_formulario = True

    if st.session_state.get('mostrar_formulario', False):
        crear_interaccion(solicitud_id)




import folium
from selenium import webdriver
from PIL import Image
from io import BytesIO
import os

import os
import folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from PIL import Image
from io import BytesIO

def generar_mapa(solicitud_data):
    """Genera un mapa con la ubicación del cliente y lo guarda como imagen usando Selenium."""
    if 'latitude' in solicitud_data.columns and 'longitude' in solicitud_data.columns:
        coordenadas = solicitud_data[['latitude', 'longitude']].dropna()
        if not coordenadas.empty:
            lat, lon = coordenadas.iloc[0]
            
            # Crear un mapa con folium
            mapa = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker([lat, lon], tooltip="Ubicación del Cliente").add_to(mapa)
            
            # Guardar el mapa como HTML
            mapa_html = "temp_map.html"
            mapa.save(mapa_html)
            
            # Configurar opciones de Selenium para el navegador Chrome en modo headless
            options = Options()
            options.add_argument('--headless')  # Ejecutar en modo sin interfaz gráfica
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')  # Desactivar el uso de la GPU (recomendado para entornos sin GUI)
            
            # Configurar el path al ChromeDriver si es necesario
            # Asegúrate de que chromedriver esté en el PATH o especifica su ubicación aquí
            service = Service(executable_path='/path/to/chromedriver')  # Cambia esta línea si es necesario
            
            # Asegúrate de tener el ChromeDriver compatible con tu versión de Chrome
            driver = webdriver.Chrome(service=service, options=options)
            
            # Abrir el mapa guardado como archivo HTML
            driver.get("file:///" + os.path.abspath(mapa_html))
            
            # Esperar a que la página se cargue completamente
            driver.implicitly_wait(3)  # Puedes ajustar el tiempo de espera si es necesario
            
            # Tomar la captura de pantalla
            screenshot = driver.get_screenshot_as_png()
            driver.quit()
            
            # Convertir la captura de pantalla a una imagen usando Pillow
            imagen = Image.open(BytesIO(screenshot))
            imagen.save("map_image.png")
            
            return "map_image.png"
    return None



from fpdf import FPDF

def generar_pdf(interaccion, mapa_img=None):
    """Generar un PDF con la información de la interacción y el mapa."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Agregar información al PDF
    for key, value in interaccion.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    # Incluir el mapa si está disponible
    if mapa_img and os.path.exists(mapa_img):
        pdf.cell(200, 10, txt="Ubicación del Cliente:", ln=True)
        pdf.image(mapa_img, x=10, y=pdf.get_y(), w=100)
    
    # Guardar el PDF
    pdf_file = "interaccion.pdf"
    pdf.output(pdf_file)
    
    return pdf_file  # Devuelve la ruta del archivo generado



import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import date
import streamlit as st

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import date
import streamlit as st
from fpdf import FPDF

import json
import gspread

from google.auth import exceptions
from google.oauth2 import service_account
import gspread
from google.auth.transport.requests import Request
from google.auth import exceptions
from google.oauth2 import service_account
import streamlit as st
from datetime import datetime

def crear_interaccion(solicitud_id):
    """Formulario para crear una nueva interacción vinculada a un ID y generar un PDF."""
    direccion = None  # Inicializar la variable

    # Inicializar la sesión si no existe
    if "interacciones" not in st.session_state:
        st.session_state.interacciones = []

    # Buscar si ya existe una interacción con el mismo 'Solicitud_id'
    interaccion_existente = next((interaccion for interaccion in st.session_state.interacciones if interaccion["Solicitud_id"] == solicitud_id), None)
    
    if interaccion_existente:
        st.info(f"Interacción previa encontrada para el ID: {solicitud_id}")
        st.dataframe(pd.DataFrame([interaccion_existente]))
        return st.session_state.interacciones

    # Cargar datos de la solicitud
    solicitud_data = df_mo[df_mo['Solicitud_id'] == solicitud_id]

    # Diccionarios para convertir el nivel de atraso a texto y mensualidades
    nivel_atraso_dict = {
        0: 'atraso_120_149',
        1: 'atraso_150_179',
        2: 'atraso_180_más',
        3: 'atraso_1_29',
        4: 'atraso_30_59',
        5: 'atraso_60_89',
        6: 'atraso_90_119'
    }

    mensualidades_dict = {
        'atraso_1_29': 1,
        'atraso_30_59': 2,
        'atraso_60_89': 3,
        'atraso_90_119': 4,
        'atraso_120_149': 5,
        'atraso_150_179': 6,
        'atraso_180_más': 7
    }

    # Extraer información de la solicitud
    if not solicitud_data.empty:
        recomendacion_oferta = solicitud_data['Prediccion Optima'].values[0] if not pd.isna(solicitud_data['Prediccion Optima'].values[0]) else "Sin oferta"
        linea_credito = solicitud_data['Linea credito'].values[0] if 'Linea credito' in solicitud_data.columns else 0
        pago = solicitud_data['Pago'].values[0] if 'Pago' in solicitud_data.columns else 0
        ingreso_mensual = solicitud_data['Ingreso_Bruto'].values[0] if 'Ingreso_Bruto' in solicitud_data.columns else 0
        
        if 'Nivel_Atraso_encoded' in solicitud_data.columns and not pd.isna(solicitud_data['Nivel_Atraso_encoded'].values[0]):
            nivel_atraso_encoded = int(solicitud_data['Nivel_Atraso_encoded'].values[0])
            nivel_atraso = nivel_atraso_dict.get(nivel_atraso_encoded, "Desconocido")
            mensualidades_adeudadas = mensualidades_dict.get(nivel_atraso, 0)
        else:
            nivel_atraso = "Desconocido"
            mensualidades_adeudadas = 0
    else:
        st.warning("No se encontró información para este ID.")
        recomendacion_oferta = "Sin oferta"
        linea_credito = 0
        pago = 0
        ingreso_mensual = 0
        nivel_atraso = "Desconocido"
        mensualidades_adeudadas = 0

    # Opciones de oferta
    opciones_oferta = df_mo['Prediccion Optima'].dropna().unique().tolist()
    if not opciones_oferta:
        opciones_oferta = ["Sin oferta"]

    # Mostrar información en el formulario
    st.subheader("Información de la Cuenta")
    st.text_input("Recomendación Oferta de Cobranza", value=recomendacion_oferta, disabled=True)
    st.number_input("Línea de Crédito", value=float(linea_credito), format="%.0f", disabled=True)
    st.number_input("Pago Mensual", value=float(pago), format="%.0f", disabled=True)
    st.number_input("Ingreso Mensual", value=float(ingreso_mensual), format="%.0f", disabled=True)
    st.text_input("Nivel de Atraso", value=nivel_atraso, disabled=True)
    st.text_input("Mensualidades Adeudadas", value=f"{mensualidades_adeudadas} mensualidad(es)", disabled=True)

    # Definir `resultado_seleccionado` antes de usarlo
    resultado_seleccionado = st.selectbox("Resultado", ["Respondió", "No Respondió"])

    negociacion_oferta = None
    promesa_seleccionada = None
    fecha_pago_estimada = None
    monto_prometido = None
    quien_atendio = None
    especificar_quien = None
    recado = None
    numero_celular = None

    if resultado_seleccionado == "Respondió":
        negociacion_oferta = st.selectbox("Negociación Oferta", opciones_oferta)
        promesa_seleccionada = st.selectbox("Promesa", ["Sí", "No"])
        if promesa_seleccionada == "Sí":
            fecha_pago_estimada = st.date_input("Plazo de Pago Estimado", date.today())
            monto_prometido = st.number_input("Promesa de Monto a Pagar", min_value=0.0, format="%.2f")
        st.subheader("Información de Contacto")
        direccion = st.text_input("Dirección del Cliente", placeholder="Introduce la dirección del cliente")
        numero_celular = st.text_input("Número de Celular", placeholder="Introduce el número de celular del cliente")
    else:
        quien_atendio = st.selectbox("¿Quién atendió?", ["Padre", "Madre", "Hermano", "Hermana", "Esposo", "Esposa", "Hijo", "Hija", "Otro"])
        if quien_atendio == "Otro":
            especificar_quien = st.text_input("Especificar quién atendió")
        recado = st.text_area("Recado dejado a la persona que atendió", "")

    comentarios = st.text_area("Comentarios adicionales", "")

    if st.button("Guardar Interacción"):
        fecha_creacion = datetime.now().strftime("%Y-%m-%d")
        hora_creacion = datetime.now().strftime("%H:%M:%S")

        nueva_interaccion = {
            "Solicitud_id": solicitud_id,
            "Recomendación Oferta": recomendacion_oferta,
            "Línea de Crédito": linea_credito,
            "Pago Mensual": pago,
            "Ingreso Mensual": ingreso_mensual,
            "Nivel de Atraso": nivel_atraso,
            "Mensualidades Adeudadas": mensualidades_adeudadas,
            "Dirección": direccion,
            "Número de Celular": numero_celular,
            "Resultado": resultado_seleccionado,
            "Negociación Oferta": negociacion_oferta,
            "Promesa": promesa_seleccionada,
            "Fecha de Pago Estimada": fecha_pago_estimada,
            "Monto Prometido": monto_prometido,
            "Quién Atendió": quien_atendio,
            "Especificar Quién": especificar_quien,
            "Recado": recado,
            "Comentarios": comentarios,
            "Fecha de Creación": fecha_creacion,
            "Hora de Creación": hora_creacion,
        }

        st.session_state.interacciones.append(nueva_interaccion)
        #guardar_en_google_sheets(nueva_interaccion)
        st.success("Interacción guardada exitosamente en Google Sheets")

        st.subheader("Interacción guardada")
        st.dataframe(pd.DataFrame(st.session_state.interacciones))

        mapa_img = generar_mapa(solicitud_data)
        pdf_file = generar_pdf(nueva_interaccion, mapa_img)

        with open(pdf_file, "rb") as file:
            st.download_button("Descargar PDF", data=file, file_name="interaccion.pdf")

        st.session_state.mostrar_formulario = False

    return st.session_state.interacciones


# Inicializar las claves de sesión necesarias
for key, default_value in {
    "logged_in": False,
    "page": "home",
    "rol": "guest",
    "navegacion": {"solicitud_id": None},
    "solicitud_filtrada": None,
    "solicitud_seleccionada": None,
    "interacciones": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Pantalla de Inicio de Sesión y Navegación
if not st.session_state.get("logged_in", False):
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://fibotech.mx/images/dimex-logo.png" width="300">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("Iniciar Sesión")

    email = st.text_input("Email")
    password = st.text_input("Contraseña", type="password")

    if st.button("Iniciar Sesión"):
        if login(email, password):
            st.success("Inicio de sesión exitoso")
            st.session_state.page = "listado_usuarios"  # Página inicial
            if "navegacion" not in st.session_state:
                st.session_state["navegacion"] = {"solicitud_id": None}
        else:
            st.error("Correo o contraseña incorrectos")
else:
    rol_usuario = st.session_state.get("rol", "guest")
    st.sidebar.title("Navegación")

    # Opciones por rol
    if rol_usuario == "admin":
        opciones_menu = ["Listado de Usuarios", "Información de Solicitud", "Historial de Interacciones"]
    elif rol_usuario == "call_center":
        opciones_menu = ["Listado de Usuarios Call Center", "Información de Solicitud Call Center", "Historial de Interacciones (Call Center)"]
    else:
        opciones_menu = []

    # Menú lateral
    opcion = st.sidebar.selectbox("Selecciona una opción", opciones_menu)

    # Desplegador para seleccionar el ID de la solicitud
    if opcion in ["Información de Solicitud", "Historial de Interacciones"] and rol_usuario == "admin":
        solicitud_id = st.sidebar.selectbox(
            "Selecciona el ID de la solicitud",
            ["Seleccionar"] + list(df_mo["Solicitud_id"].unique())
        )
        if solicitud_id != "Seleccionar":
            st.session_state["solicitud_seleccionada"] = solicitud_id
    elif opcion in ["Información de Solicitud Call Center", "Historial de Interacciones (Call Center)"] and rol_usuario == "call_center":
        solicitud_id = st.sidebar.selectbox(
            "Selecciona el ID de la solicitud",
            ["Seleccionar"] + list(df_mo["Solicitud_id"].unique())
        )
        if solicitud_id != "Seleccionar":
            st.session_state["solicitud_seleccionada"] = solicitud_id

    # Navegación dinámica
    if opcion == "Listado de Usuarios":
        mostrar_lista_usuarios()
    elif opcion == "Información de Solicitud" and st.session_state.get("solicitud_seleccionada"):
        mostrar_informacion_usuario(st.session_state["solicitud_seleccionada"])
    elif opcion == "Historial de Interacciones" and st.session_state.get("solicitud_seleccionada"):
        mostrar_historial_interacciones(st.session_state["solicitud_seleccionada"])
    elif opcion == "Listado de Usuarios Call Center":
        mostrar_lista_usuarios_callcenter()
    elif opcion == "Información de Solicitud Call Center" and st.session_state.get("solicitud_seleccionada"):
        mostrar_informacion_usuario_callcenter(st.session_state["solicitud_seleccionada"])
    elif opcion == "Historial de Interacciones (Call Center)" and st.session_state.get("solicitud_seleccionada"):
        mostrar_historial_interacciones_callcenter(st.session_state["solicitud_seleccionada"])
    else:
        if opcion in ["Información de Solicitud", "Historial de Interacciones",
                      "Información de Solicitud Call Center", "Historial de Interacciones (Call Center)"]:
            st.warning("Selecciona un ID de solicitud para continuar.")

    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.session_state.page = None
        st.session_state.solicitud_seleccionada = None
        st.stop()

