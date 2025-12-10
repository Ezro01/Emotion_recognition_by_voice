"""
Telegram-бот для определения эмоций по голосовому сообщению.
Запуск:
1) Установить зависимости (см. requirements.txt).
2) Указать токен ниже или через переменную окружения TELEGRAM_BOT_TOKEN.
3) Убедиться, что рядом лежат emotion_model.h5 и label_encoder.json.
4) Запустить: python telegram_bot.py
"""
import os
import tempfile
from typing import Optional

import librosa
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from train_model import EmotionClassifier


classifier: Optional[EmotionClassifier] = None


def load_classifier() -> EmotionClassifier:
    global classifier
    if classifier is None:
        cls = EmotionClassifier()
        cls.load_model()  # ожидает emotion_model.h5 и label_encoder.json в текущей директории
        classifier = cls
    return classifier


async def send_typing(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправьте голосовое сообщение, и я оценю его эмоциональное состояние."
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.voice:
        return

    await send_typing(context, update.effective_chat.id)

    voice = update.message.voice
    tg_file = await voice.get_file()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "voice.ogg")
        await tg_file.download_to_drive(custom_path=local_path)

        try:
            clf = load_classifier()
        except Exception as e:
            await update.message.reply_text(f"Не удалось загрузить модель: {e}")
            return

        try:
            result = clf.predict(local_path)
        except Exception as e:
            await update.message.reply_text(f"Не удалось обработать аудио: {e}")
            return

        if not result:
            await update.message.reply_text("Не удалось распознать эмоцию, попробуйте еще раз.")
            return

        emotion_en = result["emotion"]
        confidence = result["confidence"]
        # Перевод эмоций на русский
        emotion_map = {
            "anger": "злость",
            "disgust": "грусть",
            "enthusiasm": "энтузиазм",
            "fear": "страх",
            "happiness": "радость",
            "neutral": "нейтрально",
            "sadness": "грусть",
        }
        emotion_ru = emotion_map.get(emotion_en, emotion_en)

        await update.message.reply_text(
            f"Эмоция: {emotion_ru}\nУверенность: {confidence:.2%}"
        )


def main():
    # Вставьте сюда токен бота, либо оставьте как есть и задайте TELEGRAM_BOT_TOKEN в окружении.
    token = "8288752658:AAEZb9fVPF3AfbgFaK6-7gWiMHg3TNXEhrI"  # пример: "123456:ABC..."
    if not token or token == "8288752658:AAEZb9fVPF3AfbgFaK6-7gWiMHg3TNXEhrI":
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Укажите токен в переменной окружения TELEGRAM_BOT_TOKEN или вставьте его в переменную token"
        )

    # Раннее создание модели, чтобы упасть сразу, если файлов нет
    load_classifier()

    app = (
        ApplicationBuilder()
        .token(token)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("Бот запущен. Отправьте голосовое сообщение боту в Telegram.")
    app.run_polling()


if __name__ == "__main__":
    main()
