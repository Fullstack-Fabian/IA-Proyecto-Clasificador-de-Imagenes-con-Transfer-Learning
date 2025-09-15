
# main.py - Clasificador Cats vs Dogs con Transfer Learning (MobileNetV2)
# Paso a paso y con funciones de predicción / autocorrección

import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

# ------------------- CONFIG -------------------
BASE_DIR = "data"                     # carpeta que contiene train/ y test/
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "transfer_model.h5")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 3     # usa 3 para probar; sube a 10/20 para entrenar en serio
LEARNING_RATE = 1e-4
# ----------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

def make_generators():
    """Crea y retorna los generators (train y val)."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_gen, val_gen

def build_model(num_classes):
    """Construye el modelo con MobileNetV2 (congelado) + capas superiores."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False   # congelamos la base para transfer learning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save():
    """Entrena el modelo desde cero y lo guarda."""
    print("==> Creando generators...")
    train_gen, val_gen = make_generators()
    num_classes = train_gen.num_classes
    print(f"Clases detectadas: {train_gen.class_indices}")

    print("==> Construyendo modelo...")
    model = build_model(num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(os.path.join(MODEL_DIR, "best.h5"), save_best_only=True),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]

    print("==> Entrenando (prueba con pocas épocas)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("==> Evaluando en test:")
    loss, acc = model.evaluate(val_gen)
    print(f"Loss: {loss:.4f} - Accuracy: {acc:.4f}")

    print("==> Guardando modelo final...")
    model.save(MODEL_PATH)
    print("Modelo guardado en:", MODEL_PATH)

    # Guardar mapping de clases
    class_indices = train_gen.class_indices
    inv_map = {v:k for k,v in class_indices.items()}
    np.save(os.path.join(MODEL_DIR, "labels.npy"), inv_map)
    print("Mapa de etiquetas guardado en model/labels.npy")

def predict_image(image_path, model_path=MODEL_PATH):
    """Predice la clase de una imagen usando el modelo guardado."""
    if not os.path.exists(model_path):
        print("No existe el modelo. Entrena primero (train_and_save()).")
        return None

    model = load_model(model_path)
    labels = np.load(os.path.join(MODEL_DIR, "labels.npy"), allow_pickle=True).item()

    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    label = labels[idx]
    confidence = preds[0][idx]
    print(f"Predicción: {label}  (confianza={confidence:.3f})")
    return label, confidence

def interactive_correction(image_path):
    """
    Predice la imagen y pregunta si la predicción fue correcta.
    Si es incorrecta, el usuario escribe la etiqueta correcta y la imagen se mueve a train/<label> para reentrenar.
    """
    res = predict_image(image_path)
    if res is None:
        return
    pred_label, conf = res
    ans = input("¿La predicción fue correcta? (s/n): ").strip().lower()
    if ans == 's' or ans == 'y':
        print("Perfecto. No se mueve el archivo.")
        return
    else:
        correct = input("Escribe la etiqueta correcta (por ejemplo 'cats' o 'dogs'): ").strip()
        dest_dir = os.path.join(TRAIN_DIR, correct)
        os.makedirs(dest_dir, exist_ok=True)
        fname = os.path.basename(image_path)
        new_path = os.path.join(dest_dir, fname)
        shutil.move(image_path, new_path)
        print(f"Imagen movida a {new_path}. Puedes reentrenar para que el modelo aprenda de este ejemplo.")

def retrain_epochs(add_epochs=1):
    """Reentrena el modelo guardado unos cuantos epochs con los datos actuales en train/ y test/."""
    if not os.path.exists(MODEL_PATH):
        print("No hay modelo previo. Ejecuta train_and_save() primero para crear uno.")
        return

    print("==> Recreando generators (para incluir nuevas imágenes)...")
    train_gen, val_gen = make_generators()
    labels = train_gen.class_indices
    print("Clases:", labels)

    print("==> Cargando modelo previo y reentrenando (pocas epochs)...")
    model = load_model(MODEL_PATH)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [ModelCheckpoint(os.path.join(MODEL_DIR, "best_retrain.h5"), save_best_only=True)]
    model.fit(train_gen, validation_data=val_gen, epochs=add_epochs, callbacks=callbacks, verbose=1)

    model.save(MODEL_PATH)
    print("Reentrenamiento terminado. Modelo actualizado en", MODEL_PATH)

# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    print("==== Clasificador Cats vs Dogs - main.py ====")
    print("Opciones:")
    print("1) Entrenar desde cero y guardar (train_and_save())")
    print("2) Predecir una imagen (predict_image(path))")
    print("3) Interactuar y autocorregir (interactive_correction(path))")
    print("4) Reentrenar con nuevas imágenes (retrain_epochs(n))")
    choice = input("Elige 1/2/3/4 (enter para 1): ").strip() or "1"

    if choice == "1":
        train_and_save()
    elif choice == "2":
        p = input("Ruta de la imagen a predecir: ").strip()
        predict_image(p)
    elif choice == "3":
        p = input("Ruta de la imagen para predecir y corregir: ").strip()
        interactive_correction(p)
    elif choice == "4":
        n = input("¿Cuántas epochs adicionales? (ej. 1): ").strip()
        try:
            n = int(n)
        except:
            n = 1
        retrain_epochs(add_epochs=n)
    else:
        print("Opción no válida. Saliendo.")

