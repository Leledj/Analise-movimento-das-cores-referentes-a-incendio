import cv2
from pytube import YouTube
import tensorflow as tf  # Supondo que você está usando o TensorFlow/Keras para a CNN

# Carregue o modelo CNN pré-treinado
model = tf.keras.models.load_model('primeirotreinamento.h5')

def download_youtube_video(url, save_path="."):
    yt = YouTube(url)
    ys = yt.streams.get_highest_resolution()
    ys.download(save_path)
    return ys.default_filename

def get_fire_mask_cnn(frame):
    # Pré-processe a imagem conforme necessário para sua CNN
    processed_frame = preprocess_for_cnn(frame)  # Você precisará definir essa função
    
    # Use a CNN para detectar o fogo
    predictions = model.predict(processed_frame)
    
    # Converta as previsões em uma máscara binária
    mask = predictions > 0.5  # Supondo uma classificação binária
    return mask

def apply_fire_color_filter(frame):
    # Converta a imagem para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define os limites para a cor do fogo
    lower_red1 = (0, 100, 30)
    upper_red1 = (30, 255, 255)

    lower_red2 = (150, 100, 30)
    upper_red2 = (180, 255, 255)

    # Crie máscaras para os intervalos de cores
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine as duas máscaras
    mask = cv2.bitwise_or(mask1, mask2)
    
    return mask

# Insira o link do vídeo do YouTube aqui
url = 'https://www.youtube.com/watch?v=gbM_NPx2GPc'
video_path = download_youtube_video(url)

# Inicializa o subtrator de fundo
fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtenha a máscara de fogo usando a CNN
    fire_mask = get_fire_mask_cnn(frame)

    color_mask = apply_fire_color_filter(frame)

    # Aplica a subtração de fundo
    fgmask = fgbg.apply(frame)

    # Aplica um limiar para segmentar áreas de movimento
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)

    # Remove pequenas áreas de ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

     # Combine as duas máscaras
    combined_mask = cv2.bitwise_and(fire_mask, processed)

    # Mostra o resultado
    cv2.imshow('Original', frame)
    cv2.imshow('Movimento e cor', combined_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
