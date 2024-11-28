# DMX-app

DMX-app es una aplicación financiera diseñada para la gestión de clientes, operaciones en centros de llamadas y generación de reportes interactivos. Facilita el manejo de datos, análisis en tiempo real y la creación de reportes personalizados.

---

## Características Principales

- *Gestión de Clientes*: Registra y gestiona información de clientes, como líneas de crédito, historial financiero y probabilidades de deterioro.
- *Centro de Llamadas Integrado*: Optimiza el seguimiento y registro de interacciones telefónicas, incluyendo promesas de pago y resultados.
- *Generación de Reportes*: Exporta datos en formato PDF, incluyendo mapas e informes personalizados.
- *Visualización de Datos*: Gráficos interactivos para analizar distribuciones, probabilidades y métricas clave.
- *Autenticación de Usuarios*: Sistema basado en roles (admin y call_center) para controlar el acceso a funciones específicas.
- *Estilos Personalizados*: La interfaz está diseñada usando estilos definidos en styles.css.

---

## Estructura del Proyecto

- **main.py**: Archivo principal que contiene la lógica central de la aplicación.
- **requirements.txt**: Lista de dependencias necesarias.
- **Dockerfile**: Archivo para construir la imagen Docker.
- **compose.yaml**: Configuración para Docker Compose.
- **styles.css**: Hoja de estilos para personalizar la interfaz.
- **Fotos/**: Carpeta que almacena imágenes de usuarios.
- **df_app_fin (1).csv**: Archivo con datos financieros necesarios.
- **secrets.toml**: Configuración opcional para credenciales y variables de entorno.
- **README.md**: Documentación detallada del proyecto.

---

## Instalación y Configuración

### Requisitos

- Python 3.8 o superior.
- Archivo CSV con datos financieros (df_app_fin (1).csv).
- Imágenes de usuarios almacenadas en la carpeta Fotos/.

### Pasos para la Instalación

1. *Clonar el repositorio*:

   ```bash
   git clone https://github.com/seb400tastian/dimexapp.git
   cd dimexapp
