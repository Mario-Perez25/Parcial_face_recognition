import os
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# CONFIG
# ==========================================
DATASET_DIR = "dataset"
DB_PATH = "faces_db2.npz"
MODEL_PATH = "face_landmarker.task"

mp_face = mp.tasks.vision
from mediapipe.tasks import python
BaseOptions = python.BaseOptions
FaceLandmarker = mp_face.FaceLandmarker
FaceLandmarkerOptions = mp_face.FaceLandmarkerOptions
VisionRunningMode = mp_face.RunningMode


# ==========================================
# LOAD MEDIAPIPE MODEL
# ==========================================
def load_landmarker(mode="IMAGE"):
    if mode == "IMAGE":
        running_mode = VisionRunningMode.IMAGE
    else:
        running_mode = VisionRunningMode.VIDEO

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False
    )

    return FaceLandmarker.create_from_options(options)



# ==========================================
# COMPUTE EMBEDDING
# ==========================================
def get_embedding(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] No pude leer {path}")
        return None

    landmarker = load_landmarker("IMAGE")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convertimos a imagen de MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detección correcta con MediaPipe Tasks
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print(f"[ADVERTENCIA] No detecto cara en {path}")
        return None

    emb = []
    for lm in result.face_landmarks[0]:
        emb.extend([lm.x, lm.y, lm.z])

    return np.array(emb, dtype=np.float32)


# ==========================================
# BUILD DATABASE
# ==========================================
def build_database():
    print("\n📌 Construyendo base de datos de embeddings...\n")

    embeddings = []
    labels = []

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        print(f"➡ Procesando persona: {person}")

        for imgfile in os.listdir(person_dir):
            path = os.path.join(person_dir, imgfile)

            emb = get_embedding(path)
            if emb is not None:
                embeddings.append(emb)
                labels.append(person)

    if len(embeddings) == 0:
        print("❌ No pude generar ningún embedding.")
        return

    np.savez(DB_PATH, embeddings=np.array(embeddings), labels=np.array(labels))
    print("\n✅ Base de datos creada:", DB_PATH)


# ==========================================
# LOAD DATABASE
# ==========================================
def load_database():
    if not os.path.exists(DB_PATH):
        print("❌ No existe la base de datos. Primero crea los embeddings.")
        return None, None

    data = np.load(DB_PATH, allow_pickle=True)
    return data["embeddings"], data["labels"]


# ==========================================
# FACE RECOGNITION (STATIC IMAGE)
# ==========================================
def recognize_image(image_path):
    embeddings, labels = load_database()
    if embeddings is None:
        return

    test_emb = get_embedding(image_path)
    if test_emb is None:
        print("❌ No se pudo obtener embedding de la imagen.")
        return

    # Distancia L2
    distances = np.linalg.norm(embeddings - test_emb, axis=1)
    idx = np.argmin(distances)
    min_dist = distances[idx]

    # Umbral básico
    if min_dist < 0.25:
        print(f"\n✔ Persona reconocida: {labels[idx]} (distancia: {min_dist:.4f})")
    else:
        print(f"\n❌ No coincide con nadie. Distancia mínima: {min_dist:.4f}")


# ==========================================
# FACE RECOGNITION (WEBCAM)
# ==========================================
def recognize_webcam():
    embeddings, labels = load_database()
    if embeddings is None:
        return

    cap = cv2.VideoCapture(0)

    landmarker = load_landmarker("VIDEO")

    print("\n📷 Iniciando webcam (Q para salir)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertimos a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertimos a objeto mp.Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detección correcta para video
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Si hay cara detectada
        if result.face_landmarks:
            lmks = result.face_landmarks[0]

            emb = []
            for lm in lmks:
                emb.extend([lm.x, lm.y, lm.z])
            emb = np.array(emb, dtype=np.float32)

            # Distancias
            distances = np.linalg.norm(embeddings - emb, axis=1)
            idx = np.argmin(distances)
            min_dist = distances[idx]

            name = labels[idx] if min_dist < 0.25 else "Desconocido"

            cv2.putText(frame, f"{name} ({min_dist:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.imshow("Reconocimiento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# MENU
# ==========================================
def menu():
    while True:
        print("\n==============================")
        print("   MENÚ DE RECONOCIMIENTO")
        print("==============================")
        print("1. Crear base de datos de rostros")
        print("2. Reconocer en una imagen")
        print("3. Reconocer con webcam")
        print("4. Salir")
        print("==============================")

        op = input("Elige una opción: ")

        if op == "1":
            build_database()
        elif op == "2":
            img = input("Ruta de la imagen: ")
            recognize_image(img)
        elif op == "3":
            recognize_webcam()
        elif op == "4":
            print("Saliendo...")
            break
        else:
            print("Opción incorrecta.")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    menu()
