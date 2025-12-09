"""
FastAPI приложение для определения эмоций по голосу
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import librosa
import soundfile as sf
import json
from train_model import EmotionClassifier
import uvicorn

app = FastAPI(title="Определение эмоций по голосу")

# CORS для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для модели
classifier = None

def load_model():
    """Загружает обученную модель"""
    global classifier
    if classifier is None:
        classifier = EmotionClassifier()
        if os.path.exists('emotion_model.h5') and os.path.exists('label_encoder.json'):
            try:
                classifier.load_model()
                print("✅ Модель загружена успешно")
            except Exception as e:
                print(f"⚠️  Ошибка при загрузке модели: {e}")
                return False
        else:
            print("⚠️  Модель не найдена. Необходимо обучить модель сначала.")
            print("   Запустите: python train_model.py")
            return False
    return True

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Главная страница"""
    html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Определение эмоций по голосу</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .upload-section {
            margin-bottom: 30px;
        }
        
        .section-title {
            font-size: 1.3em;
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .record-section {
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
        }
        
        .record-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 20px 40px;
            font-size: 1.2em;
            border-radius: 50px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 10px;
            font-weight: bold;
        }
        
        .record-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .record-button:active {
            transform: translateY(0);
        }
        
        .record-button.recording {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .file-input {
            width: 100%;
            padding: 15px;
            border: 2px dashed #667eea;
            border-radius: 10px;
            background: #f8f9fa;
            cursor: pointer;
            text-align: center;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        .file-input:hover {
            background: #e9ecef;
            border-color: #764ba2;
        }
        
        input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .result-section {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            display: none;
        }
        
        .result-section.show {
            display: block;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .emotion-result {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .emotion-label {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .confidence {
            font-size: 1.2em;
            color: #666;
        }
        
        .probabilities {
            margin-top: 20px;
        }
        
        .prob-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 8px;
        }
        
        .prob-bar {
            height: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-left: 10px;
            min-width: 50px;
        }
        
        .status {
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            border-radius: 10px;
            display: none;
        }
        
        .status.show {
            display: block;
        }
        
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        
        .audio-player {
            width: 100%;
            margin: 20px 0;
        }
        
        .timer {
            font-size: 1.5em;
            color: #667eea;
            margin: 15px 0;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Определение эмоций по голосу</h1>
        <p class="subtitle">Запишите свой голос или загрузите аудио файл</p>
        
        <div class="status" id="status"></div>
        
        <div class="record-section">
            <div class="section-title">🎙️ Запись голоса</div>
            <div class="timer" id="timer" style="display: none;">00:00</div>
            <button class="record-button" id="recordButton">Начать запись</button>
            <button class="record-button" id="stopButton" style="display: none;">Остановить</button>
            <audio id="audioPlayback" class="audio-player" controls style="display: none;"></audio>
        </div>
        
        <div class="upload-section">
            <div class="section-title">📁 Загрузка файла</div>
            <div class="file-input-wrapper">
                <div class="file-input">
                    <span id="fileLabel">Выберите аудио файл или перетащите его сюда</span>
                    <input type="file" id="fileInput" accept="audio/*">
                </div>
            </div>
            <audio id="uploadedAudio" class="audio-player" controls style="display: none;"></audio>
        </div>
        
        <div class="result-section" id="resultSection">
            <div class="emotion-result">
                <div class="emotion-label" id="emotionLabel"></div>
                <div class="confidence" id="confidence"></div>
            </div>
            <div class="probabilities" id="probabilities"></div>
        </div>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let recording = false;
        let timerInterval;
        let seconds = 0;
        let lastMimeType = 'audio/webm';
        
        const recordButton = document.getElementById('recordButton');
        const stopButton = document.getElementById('stopButton');
        const fileInput = document.getElementById('fileInput');
        const fileLabel = document.getElementById('fileLabel');
        const statusDiv = document.getElementById('status');
        const resultSection = document.getElementById('resultSection');
        const timer = document.getElementById('timer');
        const audioPlayback = document.getElementById('audioPlayback');
        const uploadedAudio = document.getElementById('uploadedAudio');
        
        async function blobToWav(blob) {
            // Преобразует записанный Blob в WAV (PCM 16-bit), чтобы плеер точно смог воспроизвести
            const arrayBuffer = await blob.arrayBuffer();
            const audioCtx = new AudioContext();
            const decoded = await audioCtx.decodeAudioData(arrayBuffer);
            
            const numChannels = decoded.numberOfChannels;
            const sampleRate = decoded.sampleRate;
            const numFrames = decoded.length;
            const bytesPerSample = 2;
            const blockAlign = numChannels * bytesPerSample;
            const wavBuffer = new ArrayBuffer(44 + numFrames * blockAlign);
            const view = new DataView(wavBuffer);
            
            // Заголовок RIFF/WAVE
            const writeString = (offset, str) => {
                for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
            };
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + numFrames * blockAlign, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);          // Subchunk1Size
            view.setUint16(20, 1, true);           // PCM
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * blockAlign, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bytesPerSample * 8, true);
            writeString(36, 'data');
            view.setUint32(40, numFrames * blockAlign, true);
            
            // PCM данные
            let offset = 44;
            const clamp = (v) => Math.max(-1, Math.min(1, v));
            for (let i = 0; i < numFrames; i++) {
                for (let ch = 0; ch < numChannels; ch++) {
                    const sample = clamp(decoded.getChannelData(ch)[i]);
                    const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
                    view.setInt16(offset, int16, true);
                    offset += 2;
                }
            }
            
            return new Blob([wavBuffer], { type: 'audio/wav' });
        }
        
        function showStatus(message, type) {
            statusDiv.textContent = message;
            statusDiv.className = `status show ${type}`;
            setTimeout(() => {
                statusDiv.classList.remove('show');
            }, 5000);
        }
        
        function formatTime(secs) {
            const mins = Math.floor(secs / 60);
            const sec = secs % 60;
            return `${String(mins).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
        }
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const preferredMime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
                mediaRecorder = preferredMime ? new MediaRecorder(stream, { mimeType: preferredMime }) : new MediaRecorder(stream);
                lastMimeType = mediaRecorder.mimeType || preferredMime || 'audio/webm';
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const rawBlob = new Blob(audioChunks, { type: lastMimeType });
                    let audioBlob = rawBlob;
                    try {
                        audioBlob = await blobToWav(rawBlob);
                    } catch (err) {
                        console.warn('Не удалось сконвертировать в WAV, используем исходный формат', err);
                    }
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayback.src = audioUrl;
                    audioPlayback.type = audioBlob.type || lastMimeType;
                    audioPlayback.style.display = 'block';
                    audioPlayback.load();
                    audioPlayback.play().catch(() => {});
                    
                    await sendAudio(audioBlob);
                };
                
                mediaRecorder.start();
                recording = true;
                recordButton.style.display = 'none';
                stopButton.style.display = 'inline-block';
                recordButton.classList.add('recording');
                timer.style.display = 'block';
                seconds = 0;
                
                timerInterval = setInterval(() => {
                    seconds++;
                    timer.textContent = formatTime(seconds);
                }, 1000);
                
                showStatus('Запись начата...', 'info');
            } catch (error) {
                showStatus('Ошибка доступа к микрофону: ' + error.message, 'error');
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && recording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                recording = false;
                recordButton.style.display = 'inline-block';
                stopButton.style.display = 'none';
                recordButton.classList.remove('recording');
                timer.style.display = 'none';
                clearInterval(timerInterval);
                showStatus('Запись остановлена. Обработка...', 'info');
            }
        }
        
        recordButton.addEventListener('click', startRecording);
        stopButton.addEventListener('click', stopRecording);
        
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                fileLabel.textContent = file.name;
                uploadedAudio.src = URL.createObjectURL(file);
                uploadedAudio.style.display = 'block';
                await sendFile(file);
            }
        });
        
        // Drag and drop
        const fileInputWrapper = document.querySelector('.file-input-wrapper');
        
        fileInputWrapper.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileInputWrapper.style.background = '#e9ecef';
        });
        
        fileInputWrapper.addEventListener('dragleave', (e) => {
            e.preventDefault();
            fileInputWrapper.style.background = '#f8f9fa';
        });
        
        fileInputWrapper.addEventListener('drop', async (e) => {
            e.preventDefault();
            fileInputWrapper.style.background = '#f8f9fa';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('audio/')) {
                fileInput.files = e.dataTransfer.files;
                fileLabel.textContent = file.name;
                uploadedAudio.src = URL.createObjectURL(file);
                uploadedAudio.style.display = 'block';
                await sendFile(file);
            }
        });
        
        async function sendFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            showStatus('Загрузка и обработка файла...', 'info');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayResult(result);
                    showStatus('Анализ завершен!', 'success');
                } else {
                    showStatus('Ошибка: ' + result.detail, 'error');
                }
            } catch (error) {
                showStatus('Ошибка при отправке файла: ' + error.message, 'error');
            }
        }
        
        async function sendAudio(audioBlob) {
            const formData = new FormData();
            const mime = audioBlob.type || lastMimeType || 'audio/wav';
            const ext = mime.includes('wav') ? 'wav' : (mime.includes('ogg') ? 'ogg' : 'webm');
            formData.append('file', audioBlob, `recording.${ext}`);
            
            showStatus('Обработка записи...', 'info');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayResult(result);
                    showStatus('Анализ завершен!', 'success');
                } else {
                    showStatus('Ошибка: ' + result.detail, 'error');
                }
            } catch (error) {
                showStatus('Ошибка при отправке записи: ' + error.message, 'error');
            }
        }
        
        function displayResult(result) {
            resultSection.classList.add('show');
            document.getElementById('emotionLabel').textContent = 
                getEmotionEmoji(result.emotion) + ' ' + getEmotionName(result.emotion);
            document.getElementById('confidence').textContent = 
                `Уверенность: ${(result.confidence * 100).toFixed(2)}%`;
            
            const probabilitiesDiv = document.getElementById('probabilities');
            probabilitiesDiv.innerHTML = '<h3>Вероятности всех эмоций:</h3>';
            
            // Сортировка по вероятности
            const sortedProbs = Object.entries(result.probabilities)
                .sort((a, b) => b[1] - a[1]);
            
            sortedProbs.forEach(([emotion, prob]) => {
                const item = document.createElement('div');
                item.className = 'prob-item';
                item.innerHTML = `
                    <span>${getEmotionEmoji(emotion)} ${getEmotionName(emotion)}</span>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span>${(prob * 100).toFixed(1)}%</span>
                        <div class="prob-bar" style="width: ${prob * 200}px;"></div>
                    </div>
                `;
                probabilitiesDiv.appendChild(item);
            });
        }
        
        function getEmotionEmoji(emotion) {
            const emojis = {
                'happiness': '😊',
                'sadness': '😢',
                'anger': '😠',
                'fear': '😨',
                'disgust': '🤢',
                'neutral': '😐',
                'enthusiasm': '🎉'
            };
            return emojis[emotion] || '❓';
        }
        
        function getEmotionName(emotion) {
            const names = {
                'happiness': 'Радость',
                'sadness': 'Грусть',
                'anger': 'Злость',
                'fear': 'Страх',
                'disgust': 'Отвращение',
                'neutral': 'Нейтрально',
                'enthusiasm': 'Энтузиазм'
            };
            return names[emotion] || emotion;
        }
    </script>
</body>
</html>
    """
    return html_content

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """Предсказывает эмоцию по загруженному аудио файлу"""
    if not load_model():
        raise HTTPException(status_code=500, detail="Модель не загружена. Обучите модель сначала.")
    
    # Сохранение временного файла
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Универсальное чтение аудио (wav/webm/ogg) через librosa
        try:
            y, sr = librosa.load(temp_path, sr=classifier.sample_rate, duration=classifier.duration)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Не удалось прочитать аудио: {e}")

        features = classifier._extract_features_from_waveform(y, sr)
        if features is None:
            raise HTTPException(status_code=400, detail="Не удалось извлечь признаки из аудио")

        features = features.reshape(1, -1)
        prediction = classifier.model.predict(features, verbose=0)
        predicted_class_idx = int(np.argmax(prediction[0]))
        predicted_emotion = classifier.label_encoder.classes_[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])
        probabilities = {
            emotion: float(prob) for emotion, prob in zip(classifier.label_encoder.classes_, prediction[0])
        }

        return JSONResponse(
            content={
                "emotion": predicted_emotion,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")
    
    finally:
        # Удаление временного файла
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    """Проверка работоспособности"""
    return {"status": "ok", "model_loaded": classifier is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
