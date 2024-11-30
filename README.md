# Grupo 5 - Electrocardiograma con Sensor Cardiaco y Sensor de Temperatura

Este proyecto tiene como objetivo desarrollar un sistema integral para el monitoreo de datos cardiacos en tiempo real utilizando técnicas de Machine Learning supervisado. El sistema es capaz de detectar patrones anómalos en señales cardiacas, generar alertas automáticas y mostrar la información procesada en una plataforma web y un dispositivo LCD. 

El sistema incluye las siguientes funcionalidades:
- **Adquisición de datos** desde sensores cardiacos y de temperatura.
- **Entrenamiento del modelo** en GitHub utilizando runners hospedados en GitHub Actions.
- **Reconocimiento de anomalías** basado en Machine Learning supervisado.
- **Notificaciones automáticas vía WhatsApp**, que incluyen:
  - Fotografía de la serie de tiempo anómala.
  - Nivel de estrés de la persona.
  - Fecha, hora y BPM detectados.

---

## **Arquitectura del Proyecto**

1. **Datos utilizados**:
   - Dataset principal: [ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).
   - Exploración adicional: [Anomagram](https://anomagram.fastforwardlabs.com).

2. **Componentes principales del sistema**:
   - **Modelo LSTM** para la predicción y detección de anomalías.
   - **Integración automática con GitHub** para la ejecución y almacenamiento de datos.
   - **Plataforma web** que muestra:
     - Actividad cardiaca actual.
     - Fecha y hora.
     - Presión atmosférica.
     - Pulso cardiaco.
   - **Dispositivo LCD** para visualización en hardware.

3. **Alertas de anomalías**:
   - Generación de notificaciones en tiempo real.
   - Mensajes automáticos vía WhatsApp con información detallada de la anomalía.

---

## **Archivos del Proyecto**

- `train.py`: Script principal para entrenar el modelo LSTM y realizar predicciones.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
- `data.csv`: Archivo de datos utilizados para las pruebas y validación del sistema.
- `cml.yaml`: Archivo de configuración para la ejecución de Machine Learning continuo (CML) en GitHub Actions.
- `ECG5000_TEST.txt`: Archivo de datos de prueba basado en el dataset ECG5000.

---

## **Cómo Ejecutar el Proyecto**

### **1. Clonar el repositorio**
```bash
git clone https://github.com/Speed-Snake/Machine-Learning
cd Machine-Learning
pip install -r requirements.txt
python train.py
