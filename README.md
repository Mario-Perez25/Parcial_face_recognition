Reconocimiento Facial con MediaPipe – Proyecto académico

Descripción

Proyecto en Python para reconocimiento facial basado en encodings generados con MediaPipe Tasks. Incluye menú por consola, registro de personas, generación de embeddings y reconocimiento en tiempo real con webcam.

Características

Registro de personas con múltiples fotos

Generación de embeddings precisos

Reconocimiento facial en webcam

Código modular y expandible

Estructura del proyecto

src/
│
├── dataset/          → imágenes usadas para entrenar
├── models/           → modelo del Face Landmarker + embeddings
└── reconocimiento.py → script principal con menú

🧠 Cómo funciona el sistema
📌 1. Generación de embeddings

Cada imagen del dataset es procesada por MediaPipe para extraer un embedding facial (un vector de 256 características).

📌 2. Entrenamiento

Se guardan todos los embeddings y sus etiquetas en models/faces_db2.npz.

📌 3. Reconocimiento

Durante la detección, el embedding capturado se compara con los almacenados usando la distancia euclidiana.

Si la distancia mínima es menor al umbral, se reconoce la persona.

Intalación 

git clone https://github.com/Mario-Perez25/Parcial_face_recognition.git
cd Parcial_face_recognition
pip install -r requirements.txt


Uso

python src/reconocimiento.py

Licencia

MIT License (opcional)
