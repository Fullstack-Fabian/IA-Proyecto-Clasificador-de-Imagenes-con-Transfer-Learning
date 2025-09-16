# 📦 IA-Proyecto-Clasificador-de-Imagenes-con-Transfer-Learning

Este proyecto implementa un **clasificador de imágenes** usando *Transfer Learning* en TensorFlow/Keras.  
Para entrenar el modelo se requiere un **dataset grande**, el cual no está incluido directamente en GitHub por las limitaciones de tamaño (100MB por archivo).  

---

## 🚀 Cómo usar el dataset

### 1. Descargar dataset partido
El dataset está dividido en varios archivos de 200MB y almacenado en Google Drive:  

👉 [Descargar dataset desde Google Drive](https://drive.google.com/drive/folders/1tLNgk7UdlS-awAhWQxX06QT8b4g9X6Tb?usp=drive_link)

---

### 2. Recomponer el dataset
Después de descargar todas las partes (`archive_part_000`, `archive_part_001`, …) mételas en una carpeta llamada `partes/` dentro del proyecto y corre:

```bash
bash recomponer.sh

### 3. Descomprimir dataset
unzip archive.zip -d data/

