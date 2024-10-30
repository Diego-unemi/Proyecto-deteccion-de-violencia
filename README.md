# Detección de Violencia con Django y AWS
## Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar un sistema de detección de violencia en tiempo real utilizando tecnologías de visión por computadora. La aplicación, construida con Django para el backend y desplegada en AWS, emplea cámaras para analizar transmisiones de video en vivo y detectar incidentes violentos. El sistema es capaz de identificar comportamientos violentos utilizando modelos de aprendizaje automático entrenados en datasets relevantes, proporcionando alertas en tiempo real a través de una interfaz web accesible para los usuarios.

Este proyecto está diseñado para ser escalable y eficiente, con el potencial de ser utilizado en escenarios de vigilancia, seguridad pública y protección de instalaciones sensibles.
## Características
Detección en Tiempo Real: Analiza transmisiones de video en vivo para detectar eventos violentos de manera inmediata.
Integración con AWS: Utiliza servicios de AWS para la gestión de la base de datos y el despliegue en un entorno de producción escalable.
API RESTful: Implementación de una API en Django que permite la interacción con el frontend y otros sistemas.
Interfaz Web: Proporciona una interfaz web para la visualización en tiempo real de los eventos detectados, accesible desde cualquier dispositivo con conexión a internet.
Optimización y Escalabilidad: El sistema está optimizado para manejar un alto volumen de datos y tráfico, asegurando un rendimiento eficiente en entornos de producción.

## Tecnologías Utilizadas
- Backend: Django

- Base de Datos: AWS RDS (Relational Database Service)

- Frontend: HTML, CSS, JavaScript (integrado con Django)

- Visión por Computadora: OpenCV u otras bibliotecas de detección de violencia

- Despliegue: AWS EC2, AWS S3, AWS RDS

- Control de Versiones: Git, GitHub


## Requisitos Previos

- Python 3.x

- Django 3.x o superior

- AWS CLI configurado

- OpenCV y otras bibliotecas de detección de violencia

- Git

## Pasos para la Instalación

### 1. Clona el repositorio:
```
git clone https://github.com/Diego-unemi/Proyecto-deteccion-de-violencia.git
```
### 2. Navega al directorio del proyecto:
```
cd Proyecto-deteccion-de-violencia
```
### 3. Crea un entorno virtual e instálalo:
```
python -m venv venv
```
```
venv\Scripts\activate
```
### 4. Instala las dependencias:
```
pip install -r requirements.txt
```
### 5. Configura la base de datos en AWS y realiza las migraciones:
```
python manage.py makemigrations
```
```
python manage.py migrate
```
### 6. Ejecuta el servidor de desarrollo:
```
python manage.py runserver
```
## Uso
Inicia el servidor localmente o despliega la aplicación en AWS.
Accede a la interfaz web en tu navegador para monitorear los videos en tiempo real.
El sistema detectará automáticamente eventos de violencia y los mostrará en la interfaz.
## Contribuciones
Las contribuciones son bienvenidas. Si tienes alguna mejora o nueva funcionalidad que te gustaría agregar, por favor, realiza un fork de este repositorio, crea una nueva rama para tus cambios, y envía un pull request para su revisión.

## Uso de Git
### Hacer cambios de tu rama local al repositorio
Para conocer los commits que hay se usa:
```
git load
```
Cambiarse de rama, por ejemplo la rama develop, se debe poner el nombre de la rama a la cual se quiera cambiar:
```
git checkout develop
```
Para agregar todo los cambios:
```
 git add .
```
Para agregar un cambio de un archivo en especifico:
```
git add nombre del archivo
```
Comentar los cambios ya hechos anteriormente:
```
git commit -m "inserte el comentario para los cambios"
```
Subir los cambios:
```
git push origin rama a asignar 
```
Para fusionar la rama remota o repositorio con la hecha en a rama local: 
```
git pull origin rama destinataria
```
### Otros usos de git
Para actualizar tu rama local, esto integra los cambios de la rama remota o repositorio a tu rama local, lo que hace es ordenar los commits de fechas diferentes:
```
git pull --rebase  origin rama destinataria
```
Para verificar si hay novedades o cambios en el repositorios y descargalos unicamente:
```
git fetch origin 
```
Para fusionar las novedades o cambios hechos con fetch con lo hecho en tu rama local se utiliza:
```
git merge origin rama destinataria
```
Para crear una rama se usa:
```
git checkout -b nombre de la rama
```
Para revisar todas las ramas se usa:
```
git branch --all
```
## Licencia

Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo LICENSE .


