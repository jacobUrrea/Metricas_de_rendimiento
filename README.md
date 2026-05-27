***Pontificia Universidad Javeriana***
# **Procesamiento de Alto Volumen de Datos**

### Taller: **Métricas y Machine Learning en PySpark**

**Autor**: Andres Jacobo Urrea Ochoa

**Fecha de Inicio**: 28 de Abril del 2026

## **Objetivos**
- Desarrollar un pipeline completo de clasificación binaria sobre el  
  dataset Bank Marketing utilizando Apache Spark en un modo clúster distribuido
- Utilizar técnicas de análisis exploratorio con el fin de descubrir los predictores más relevantes que influyen en la suscripción de depósitos bancarios
- Entrenar y comparar 6 modelos de Machine Learning evaluando mediante su desempeñomedidas tradicionales de clasificación
- Elegir el modelo mejor evaluado en una herramienta para la segmentación de clientes por score de probabilidad
  
## Dataset
El dataset **Bank Marketing** del repositorio UCI contiene 45,211 registros
de clientes de una institución bancaria portuguesa, recolectados durante
campañas de marketing telefónico entre 2008 y 2013. El objetivo es predecir
si el cliente suscribirá un depósito a plazo fijo (`yes`/`no`).


## **Tecnologías Utilizadas**
- **Apache Spark**: procesamiento distribuido en clúster con scheduler FAIR
- **PySpark MLlib**: entrenamiento y evaluación de modelos
- **Pandas / NumPy**: manipulación de datos en memoria
- **Matplotlib / Seaborn**: visualización y análisis exploratorio
- **Scikit-learn**: cálculo de curvas ROC y métricas adicionales

## **Pasos del Laboratorio*

### 1. Configuración de Spark
- Configuración del clúster Spark con scheduler FAIR
- Creación de la sesión Spark en modo distribuido
- Carga del dataset `bank-full.csv`

### 2. Descripción del Dataset
- Inspección inicial con .show() y .describe()
- Identificación de tipos de variables numéricas y categóricas
- Conversión de columnas a tipo entero

### 3. Análisis de Desbalance
- Verificación de la distribución de la variable objetivo 'Suscripcion'
- Resultado: **88.3% no / 11.7% yes**
  
### 4. Análisis Exploratorio de Datos (EDA)
- Histogramas de variables numéricas
- Boxplots de variables numéricas vs suscripción
- Gráficas de pastel para variables binarias
- Pairplot coloreado por clase objetivo
- Barras de distribución de variables categóricas
- Barras agrupadas de variables categóricas vs suscripción
- Tabla cruzada Mes vs Suscripción
- Tasa de conversión real por rango de edad
- Monto y edad promedio por tipo de trabajo
- Boxplots Edad vs Estado Civil y Edad vs Educación

### 5. Limpieza y Preparación
- Verificación de valores nulos por columna
- Análisis del valor -1 en pdays (clientes no contactados)
- Eliminación de outliers en Contactos_Previos 
- Eliminación de la columna Contactos_Previos del dataset final

### 6. Balanceo de Clases
- Separación de clases mayoritaria (no) y minoritaria (yes)
- Aplicación de Oversampling sobre la clase minoritaria
- Resultado: dataset balanceado con proporción ~50/50
- Verificación del balance en conjuntos train y test

### 7. Preprocesamiento con Pipeline
- **StringIndexer** codificación de variables categóricas a índices
- **OneHotEncoder** vectores binarios por cada categoría
- **StringIndexer** sobre variable objetivo → columna `label`
- **VectorAssembler** ensamblado de todas las features en un vector
- Guardado del pipeline en disco en formato Parquet

### 8. División Train / Test
- Split 80/20 con seed=42 para reproducibilidad
- Verificación de balance en ambos conjuntos

### 9. Entrenamiento de Modelos
Seis modelos entrenados sobre el mismo conjunto de entrenamiento
y evaluados sobre el mismo conjunto de prueba:

1. Regresión Logística 
2. Random Forest 
3. GBT
4. Decision Tree
5. Linear SVC 
6. Multilayer Perceptron 

### 10. Evaluación de Modelos
Para cada modelo se calculó:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC
- Matriz de Confusión (heatmap)
- Curva ROC individual
- Curva ROC comparativa de los 6 modelos
- Heatmap comparativo de métricas

### 11. Feature Importance
- Análisis de importancia de variables del Random Forest
- `Duracion` domina con importancia de 0.43
- `poutcome_success` segundo lugar con 0.12

### 12. Segmentación por Score
Usando las probabilidades del modelo GBT se segmentaron los clientes
en tres grupos accionables:
##### Segmento: 
Alta Prioridad 
Media Prioridad 
Baja Prioridad 
- Contactando solo el **13.9%** de la base (Alta Prioridad)
- El banco captura al **93.46%** de los suscriptores con mayor certeza
- Hace que la campaña sea aproximadamente **8 veces más eficiente** que un enfoque sin segmentación.

### 13. Conclusiones y Cierre
- Conclusiones generales del análisis
- Recomendación del modelo GBT para producción
- Cierre de la sesión Spark con `sparkB.stop()`


## Principales Hallazgos del EDA

- **Variable más predictiva:** `Duracion` con importancia de 0.43 en
  el Random Forest. Requiere precaución por riesgo de data leakage.
- **Mejor predictor categórico:** `poutcome = success` con tasa de
  conversión superior al 50%.
- **Desbalance de clases:** abordado mediante oversampling de la clase
  minoritaria, logrando una proporción 50/50 para el entrenamiento.
- **Perfil del cliente ideal:** jubilado o estudiante, sin deudas activas,
  con historial positivo en campañas anteriores, contactado en meses de
  baja actividad (dic, mar, sep) y con llamada de larga duración.


## **Referencias **
- UCI Machine Learning Repository. (2012). Bank Marketing Dataset. https://archive.ics.uci.edu/dataset/222/bank+marketing
- GeeksforGeeks. (2025, 23 julio). Evaluation Metrics For Classification Model in Python. GeeksforGeeks. https://www-geeksforgeeks-org.translate.goog/machine-learning/evaluation-metrics-for-classification-model-in-python/?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc&_x_tr_hist=true
- Apache Software Foundation. (2024). Logistic Regression. https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression
- Apache Software Foundation. (2024). Random Forest. https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier
- Apache Software Foundation. (2024). Gradient Boosted Trees. https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier
- Apache Software Foundation. (2024). Decision Tree. https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier
- Apache Software Foundation. (2024). Linear Support Vector Machine. https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-support-vector-machine
- Apache Software Foundation. (2024). Multilayer Perceptron Classifier. https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier
