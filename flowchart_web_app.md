# Блок-схема алгоритма веб-приложения

```mermaid
flowchart TD
    Start([Запуск приложения]) --> Init[Инициализация FastAPI]
    Init --> Startup[Событие startup]
    Startup --> LoadModel{Модель загружена?}
    LoadModel -->|Нет| LoadModelFile[Загрузка модели из файла]
    LoadModelFile --> CheckFiles{Файлы существуют?}
    CheckFiles -->|Нет| ErrorModel[Ошибка: модель не найдена]
    CheckFiles -->|Да| LoadSuccess[Модель загружена успешно]
    LoadModel -->|Да| LoadSuccess
    LoadSuccess --> ServerReady[Сервер готов к работе]
    ErrorModel --> ServerReady
    
    ServerReady --> WaitRequest[Ожидание запросов]
    
    WaitRequest --> RequestType{Тип запроса}
    
    RequestType -->|GET /| RootPage[Главная страница]
    RootPage --> ReturnHTML[Возврат HTML с интерфейсом]
    ReturnHTML --> WaitRequest
    
    RequestType -->|POST /predict| PredictRequest[Запрос на предсказание]
    PredictRequest --> CheckModelLoaded{Модель загружена?}
    CheckModelLoaded -->|Нет| Error500[Ошибка 500: модель не загружена]
    Error500 --> WaitRequest
    
    CheckModelLoaded -->|Да| ReceiveFile[Получение аудио файла]
    ReceiveFile --> SaveTemp[Сохранение во временный файл]
    SaveTemp --> ReadAudio[Чтение аудио через librosa]
    ReadAudio --> CheckRead{Успешно прочитано?}
    CheckRead -->|Нет| Error400[Ошибка 400: не удалось прочитать аудио]
    Error400 --> DeleteTemp[Удаление временного файла]
    DeleteTemp --> WaitRequest
    
    CheckRead -->|Да| ExtractFeatures[Извлечение признаков из аудио]
    ExtractFeatures --> CheckFeatures{Признаки извлечены?}
    CheckFeatures -->|Нет| Error400_2[Ошибка 400: не удалось извлечь признаки]
    Error400_2 --> DeleteTemp
    
    CheckFeatures -->|Да| Reshape[Преобразование признаков в формат модели]
    Reshape --> Predict[Предсказание через модель]
    Predict --> GetMax[Получение индекса максимальной вероятности]
    GetMax --> DecodeEmotion[Декодирование эмоции через label_encoder]
    DecodeEmotion --> CalcConfidence[Расчет уверенности]
    CalcConfidence --> GetProbabilities[Получение вероятностей всех эмоций]
    GetProbabilities --> FormatResponse[Формирование JSON ответа]
    FormatResponse --> DeleteTemp
    DeleteTemp --> ReturnJSON[Возврат результата клиенту]
    ReturnJSON --> WaitRequest
    
    RequestType -->|GET /health| HealthCheck[Проверка работоспособности]
    HealthCheck --> ReturnHealth[Возврат статуса]
    ReturnHealth --> WaitRequest
    
    style Start fill:#90EE90
    style ServerReady fill:#87CEEB
    style ErrorModel fill:#FFB6C1
    style Error400 fill:#FFB6C1
    style Error400_2 fill:#FFB6C1
    style Error500 fill:#FFB6C1
    style ReturnJSON fill:#98FB98
    style ReturnHTML fill:#98FB98
```

## Описание основных блоков

### Инициализация
- **Запуск приложения**: FastAPI приложение инициализируется
- **Загрузка модели**: При старте загружается обученная модель из файлов `emotion_model.h5` и `label_encoder.json`

### Обработка запросов
- **GET /** - Возвращает HTML страницу с интерфейсом для записи/загрузки аудио
- **POST /predict** - Обрабатывает загруженный аудио файл и возвращает предсказание эмоции
- **GET /health** - Проверка работоспособности сервера

### Процесс предсказания
1. Получение аудио файла от клиента
2. Сохранение во временный файл
3. Чтение аудио через librosa
4. Извлечение признаков (MFCC, спектральные характеристики)
5. Предсказание через нейронную сеть
6. Декодирование результата и формирование ответа
7. Удаление временного файла

