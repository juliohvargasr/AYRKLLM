# Informe Técnico: Preparación de Datos para Modelos de IA

**Proyecto:** Sistema de Realidad Aumentada e Inteligencia Artificial para Gestión de Plagas en Arroz.

**Objetivo:** Generar un dataset maestro (*Master Dataset*) de alta calidad, balanceado y formateado para el entrenamiento de modelos YOLOv8/v5 y MobileNetV2.

## 1. Fuentes de Datos (Datasets Base)

Para evitar iniciar desde cero, se recomienda unificar los siguientes datasets *Open Source* validados científicamente, los cuales contienen las clases críticas (*Rice Blast*, *Brown Spot*, *Spodoptera*):

| **Nombre del Dataset**             | **Fuente**                                                                                                                                           | **Contenido Clave**                                                            | **Uso Recomendado**                                   |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------- |
| **Rice Leaf Disease (UCI/Kaggle)** | [UCI Repository](https://archive.ics.uci.edu/dataset/1186/paddy+dataset) / [Kaggle](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset) | ~3,800 imágenes de *Blast*, *Brown Spot* y hoja sana.                          | Base principal para detección de enfermedades.        |
| **Mendeley Rice & Pest**           | [Mendeley Data](https://www.google.com/search?q=https://data.mendeley.com/datasets/vwv3nry3wr/1)                                                     | ~19,000 imágenes (incluye aumentadas) con clases de plagas como *Leaf folder*. | Entrenamiento robusto de plagas específicas.          |
| **Spodoptera Frugiperda (CAB)**    | [CABI / Global Database](https://gd.eppo.int/taxon/laphfr/photos)                                                                                    | Imágenes técnicas de la plaga *Spodoptera* (Gusano Cogollero).                 | Requiere anotación manual cuidadosa (dataset escaso). |

## 2. Técnicas Aplicadas de Preparación de Datos

### A. Limpieza de Datos (*Data Cleaning*)

El objetivo es eliminar "ruido" que confunda al modelo.

1. **Eliminación de duplicados:** Uso de algoritmos de *hashing* (ej. pHash) para detectar y borrar imágenes idénticas provenientes de distintas fuentes.

2. **Filtrado de calidad:** Descartar imágenes con desenfoque extremo (*motion blur*) o resolución inferior a 640x640 píxeles.

3. **Consistencia de clases:** Unificar etiquetas. Por ejemplo, renombrar `leaf_blast`, `RiceBlast` y `blast_disease` a una única etiqueta estándar: **`rice_blast`**.

### B. Normalización y Preprocesamiento

Estandarización necesaria para la arquitectura YOLO/MobileNet.

1. **Redimensionamiento:** Convertir todas las imágenes a **640x640 píxeles** (resolución nativa de entrada de YOLOv8).

2. **Normalización de píxeles:** Escalar valores de píxeles de [0, 255] a [0, 1] para acelerar la convergencia del entrenamiento.

3. **Conversión de color:** Asegurar que todas las imágenes estén en espacio de color **RGB**. Convertir imágenes en escala de grises o CMYK si se detectan.

### C. Estructuración y Anotación

1. **Formato de Etiquetas:**
   
   - Para **Detección (YOLO):** Formato `.txt` donde cada línea es `clase x_centro y_centro ancho alto` (valores normalizados).
   
   - Para **Clasificación (MobileNet):** Estructura de carpetas (`/train/rice_blast/img001.jpg`).

2. **Herramienta recomendada:** Utilizar **Roboflow** o **CVAT** para corregir las *Bounding Boxes* (cajas delimitadoras) de los datasets públicos, que suelen tener errores de precisión.

### D. Aumento de Datos (*Data Augmentation*) - Técnica "Rocking"

Para crear un dataset robusto ("rocked") que soporte condiciones de campo difíciles:

1. **Flip Horizontal/Vertical:** Simula diferentes orientaciones de la hoja.

2. **Cambios de Iluminación:** Variar brillo +/- 25% y contraste (simular sol fuerte o atardecer).

3. **Ruido Gaussiano:** Simular la baja calidad de cámaras de gama baja.

4. **Cutout / Mosaic (Específico de YOLO):** Mezclar 4 imágenes en una para forzar al modelo a detectar objetos pequeños y en contexto parcial.

## 3. Estructura del Dataset Final ("Rocked Dataset")

Una vez procesado, tu dataset debe tener la siguiente estructura de carpetas para ser compatible con scripts de entrenamiento estándar:

Plaintext

```
/Rice_Protection_Master_Dataset_2026
│
├── /train              # 70% de los datos (Imágenes + Etiquetas)
│   ├── /images
│   └── /labels
│
├── /valid              # 20% de los datos (Para ajuste de hiperparámetros)
│   ├── /images
│   └── /labels
│
├── /test               # 10% de los datos (Imágenes puras, NUNCA vistas por el modelo)
│   ├── /images
│   └── /labels
│
└── data.yaml           # Archivo de configuración para YOLO
```

**Ejemplo de contenido de `data.yaml`:**

YAML

```
names:
  0: healthy
  1: rice_blast
  2: brown_spot
  3: spodoptera_frugiperda
  4: leaf_folder
nc: 5  # Número de clases
train: ../train/images
val: ../valid/images
```

### Siguientes Pasos Sugeridos

1. **Descarga:** Obtén los datasets de Kaggle/Mendeley usando las APIs de Python (`kaggle datasets download`).

2. **Fusión:** Ejecuta un script en Python para mover todas las imágenes a una carpeta común y renombrarlas con un prefijo único (ej. `uci_blast_001.jpg`).

3. **Limpieza:** Usa una herramienta como **Roboflow** (versión gratuita académica) para subir las imágenes, aplicar los filtros de limpieza y generar las versiones aumentadas automáticamente.

4. **Exportación:** Exporta el resultado en formato **"YOLOv8 PyTorch"**.
