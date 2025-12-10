# Блок-схема алгоритма телеграм-бота

```mermaid
flowchart TD
    Start([Запуск бота]) --> CheckToken{Токен указан?}
    CheckToken -->|Нет| ErrorToken[Ошибка: токен не указан]
    ErrorToken --> End([Завершение])
    
    CheckToken -->|Да| LoadClassifier[Загрузка классификатора]
    LoadClassifier --> CheckModelFiles{Файлы модели существуют?}
    CheckModelFiles -->|Нет| ErrorModel[Ошибка: модель не найдена]
    ErrorModel --> End
    
    CheckModelFiles -->|Да| InitModel[Инициализация EmotionClassifier]
    InitModel --> LoadModelFiles[Загрузка emotion_model.h5 и label_encoder.json]
    LoadModelFiles --> ModelReady[Модель готова]
    
    ModelReady --> CreateApp[Создание Application]
    CreateApp --> RegisterHandlers[Регистрация обработчиков]
    RegisterHandlers --> StartPolling[Запуск polling]
    StartPolling --> WaitUpdates[Ожидание обновлений от Telegram]
    
    WaitUpdates --> ReceiveUpdate[Получение обновления]
    ReceiveUpdate --> CheckType{Тип обновления}
    
    CheckType -->|Команда /start| StartCommand[Обработка команды start]
    StartCommand --> SendWelcome[Отправка приветственного сообщения]
    SendWelcome --> WaitUpdates
    
    CheckType -->|Голосовое сообщение| VoiceMessage[Обработка голосового сообщения]
    VoiceMessage --> CheckVoice{Голосовое сообщение есть?}
    CheckVoice -->|Нет| WaitUpdates
    
    CheckVoice -->|Да| SendTyping[Отправка действия 'typing']
    SendTyping --> GetFile[Получение файла из Telegram]
    GetFile --> CreateTempDir[Создание временной директории]
    CreateTempDir --> DownloadFile[Скачивание файла в temp/voice.ogg]
    DownloadFile --> CheckDownload{Файл скачан?}
    CheckDownload -->|Нет| ErrorDownload[Ошибка: не удалось скачать файл]
    ErrorDownload --> SendError[Отправка сообщения об ошибке]
    SendError --> WaitUpdates
    
    CheckDownload -->|Да| LoadClassifier2[Загрузка классификатора]
    LoadClassifier2 --> CheckModelLoaded{Модель загружена?}
    CheckModelLoaded -->|Нет| ErrorModelLoad[Ошибка загрузки модели]
    ErrorModelLoad --> SendErrorModel[Отправка сообщения об ошибке модели]
    SendErrorModel --> WaitUpdates
    
    CheckModelLoaded -->|Да| PredictEmotion[Предсказание эмоции через clf.predict]
    PredictEmotion --> CheckResult{Результат получен?}
    CheckResult -->|Нет| ErrorPredict[Ошибка обработки аудио]
    ErrorPredict --> SendErrorPredict[Отправка сообщения об ошибке]
    SendErrorPredict --> WaitUpdates
    
    CheckResult -->|Да| ExtractEmotion[Извлечение эмоции и уверенности]
    ExtractEmotion --> TranslateEmotion[Перевод эмоции на русский]
    TranslateEmotion --> FormatMessage[Формирование сообщения с результатом]
    FormatMessage --> SendResult[Отправка результата пользователю]
    SendResult --> CleanupTemp[Очистка временных файлов]
    CleanupTemp --> WaitUpdates
    
    style Start fill:#90EE90
    style ModelReady fill:#87CEEB
    style ErrorToken fill:#FFB6C1
    style ErrorModel fill:#FFB6C1
    style ErrorDownload fill:#FFB6C1
    style ErrorModelLoad fill:#FFB6C1
    style ErrorPredict fill:#FFB6C1
    style SendResult fill:#98FB98
    style SendWelcome fill:#98FB98
```

## Описание основных блоков

### Инициализация
- **Проверка токена**: Проверяется наличие токена бота (из переменной окружения или кода)
- **Загрузка модели**: При запуске загружается классификатор эмоций из файлов модели
- **Создание приложения**: Инициализация Telegram Application с токеном
- **Регистрация обработчиков**: Регистрация обработчиков для команды `/start` и голосовых сообщений

### Обработка команд
- **Команда /start**: Отправляет приветственное сообщение пользователю

### Обработка голосовых сообщений
1. **Получение сообщения**: Бот получает голосовое сообщение от пользователя
2. **Индикация обработки**: Отправка действия "typing" для уведомления пользователя
3. **Скачивание файла**: Скачивание голосового файла из Telegram во временную директорию
4. **Загрузка модели**: Проверка и загрузка классификатора (если еще не загружен)
5. **Предсказание**: Обработка аудио через метод `predict()` классификатора
6. **Перевод результата**: Перевод названия эмоции с английского на русский
7. **Отправка результата**: Формирование и отправка сообщения с результатом пользователю
8. **Очистка**: Автоматическое удаление временных файлов

### Обработка ошибок
- Ошибки загрузки модели
- Ошибки скачивания файла
- Ошибки обработки аудио
- Все ошибки отправляются пользователю в виде текстовых сообщений

