"""
Скрипт для обучения модели классификации эмоций по голосу
"""
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from tqdm import tqdm
import io
import matplotlib

matplotlib.use("Agg")  # безопасный backend для генерации графиков
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionClassifier:
    def __init__(
        self,
        sample_rate=22050,
        duration=3,
        n_mfcc=13,
        augmentation_factor=1,
        random_state=42,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.augmentation_factor = augmentation_factor
        self.model = None
        self.label_encoder = LabelEncoder()
        self._rng = np.random.default_rng(random_state)

    def _get_model_summary(self):
        """Возвращает текстовое представление архитектуры модели."""
        if self.model is None:
            return ""

        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + "\n"))
        return buffer.getvalue()

    def _plot_confusion_matrix(self, cm, labels, output_path="confusion_matrix.png"):
        """Сохраняет матрицу ошибок в виде тепловой карты."""
        plt.figure(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.7)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Предсказано")
        plt.ylabel("Истинно")
        plt.title("Матрица ошибок (heatmap)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _plot_training_curves(self, history_dict, output_path="training_curves.png"):
        """Строит и сохраняет графики точности и функции потерь."""
        epochs = range(1, len(history_dict["accuracy"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(epochs, history_dict["accuracy"], label="Train Accuracy")
        axes[0].plot(epochs, history_dict["val_accuracy"], label="Val Accuracy")
        axes[0].set_title("Динамика точности")
        axes[0].set_xlabel("Эпоха")
        axes[0].set_ylabel("Точность")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, history_dict["loss"], label="Train Loss")
        axes[1].plot(epochs, history_dict["val_loss"], label="Val Loss")
        axes[1].set_title("Динамика функции потерь")
        axes[1].set_xlabel("Эпоха")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.suptitle("Графики обучения", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def _build_analysis_text(self, report, final_train_acc, final_val_acc, best_val_acc, cm, labels):
        """Формирует краткий анализ результатов."""
        analysis_lines = []

        analysis_lines.append(
            f"Финальная точность: train {final_train_acc:.2%}, val {final_val_acc:.2%}, лучшая val {best_val_acc:.2%}."
        )

        per_class_f1 = [
            (cls, metrics["f1-score"])
            for cls, metrics in report.items()
            if isinstance(metrics, dict) and "f1-score" in metrics
        ]
        if per_class_f1:
            best_cls, best_f1 = max(per_class_f1, key=lambda x: x[1])
            worst_cls, worst_f1 = min(per_class_f1, key=lambda x: x[1])
            analysis_lines.append(f"Лучший класс по F1: {best_cls} ({best_f1:.3f}).")
            analysis_lines.append(f"Слабейший класс по F1: {worst_cls} ({worst_f1:.3f}).")

        if cm.size > 1:
            off_diag = cm.copy()
            np.fill_diagonal(off_diag, 0)
            max_pos = np.unravel_index(np.argmax(off_diag), off_diag.shape)
            if off_diag[max_pos] > 0:
                analysis_lines.append(
                    f"Наибольшая путаница: истинный '{labels[max_pos[0]]}' ошибочно классифицируется как '{labels[max_pos[1]]}' ({off_diag[max_pos]} раз)."
                )

        return "\n".join(f"- {line}" for line in analysis_lines)

    def _save_markdown_report(
        self,
        model_summary,
        final_train_acc,
        final_val_acc,
        best_val_acc,
        best_train_acc,
        history_dict,
        classification_report_dict,
        confusion_matrix_path,
        curves_path,
        analysis_text,
        encoder_classes,
        report_path="TRAINING_REPORT.md",
    ):
        """Сохраняет полный отчёт по обучению в Markdown."""
        lines = []
        lines.append("# Отчёт об обучении модели классификации эмоций\n")

        lines.append("## Архитектура модели")
        lines.append("```\n" + model_summary.strip() + "\n```\n")

        lines.append("## Извлечение признаков")
        lines.append(
            "- MFCC (13 коэффициентов) + усреднение по времени\n"
            "- Chroma STFT\n"
            "- Mel-спектрограмма\n"
            "- Спектральный контраст\n"
            "- Tonnetz\n"
            "- Все признаки усредняются по временной оси и конкатенируются"
        )
        lines.append(
            "### Аугментации\n"
            "- Добавление шума\n"
            "- Pitch-shift\n"
            "- Time-stretch\n"
            "- Сдвиг по времени\n"
            "- Изменение усиления (gain)"
        )

        lines.append("## Итоги обучения")
        lines.append(f"- Финальная точность (train): {final_train_acc:.4f}")
        lines.append(f"- Лучшая точность (train): {best_train_acc:.4f}")
        lines.append(f"- Финальная точность (val): {final_val_acc:.4f}")
        lines.append(f"- Лучшая точность (val): {best_val_acc:.4f}")
        lines.append(f"- Финальная loss (train): {history_dict['loss'][-1]:.4f}")
        lines.append(f"- Финальная loss (val): {history_dict['val_loss'][-1]:.4f}")
        lines.append(f"- Минимальная loss (val): {min(history_dict['val_loss']):.4f}\n")

        lines.append("## Classification report")
        lines.append("| Класс | Precision | Recall | F1-score | Support |")
        lines.append("| --- | --- | --- | --- | --- |")
        for cls in encoder_classes:
            metrics = classification_report_dict.get(cls, {})
            lines.append(
                f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | "
                f"{metrics.get('f1-score', 0):.4f} | {metrics.get('support', 0)} |"
            )
        lines.append(
            f"| Macro avg | {classification_report_dict['macro avg']['precision']:.4f} | "
            f"{classification_report_dict['macro avg']['recall']:.4f} | "
            f"{classification_report_dict['macro avg']['f1-score']:.4f} | "
            f"{classification_report_dict['macro avg']['support']} |"
        )
        lines.append(
            f"| Weighted avg | {classification_report_dict['weighted avg']['precision']:.4f} | "
            f"{classification_report_dict['weighted avg']['recall']:.4f} | "
            f"{classification_report_dict['weighted avg']['f1-score']:.4f} | "
            f"{classification_report_dict['weighted avg']['support']} |"
        )
        lines.append("")

        lines.append("## Матрица ошибок")
        lines.append(f"![Матрица ошибок]({confusion_matrix_path})\n")

        lines.append("## Графики обучения")
        lines.append(f"![Графики точности и loss]({curves_path})\n")

        lines.append("## Анализ результатов")
        lines.append(analysis_text + "\n")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _pad_or_trim(self, y):
        """Приводит аудио к фиксированной длине."""
        target_len = self.sample_rate * self.duration
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        elif len(y) > target_len:
            y = y[:target_len]
        return y

    def _extract_features_from_waveform(self, y, sr):
        """Извлекает признаки из уже загруженного аудио."""
        y = self._pad_or_trim(y)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        return np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ])

    def _augment_waveforms(self, y, sr):
        """Генерирует аугментированные версии аудио."""
        if self.augmentation_factor <= 1:
            return []

        def add_noise(signal, scale=0.02):
            noise = self._rng.normal(0, 1, len(signal))
            return signal + scale * np.std(signal) * noise

        def pitch_shift(signal, steps):
            return librosa.effects.pitch_shift(signal, sr=sr, n_steps=steps)

        def time_stretch(signal, rate):
            return librosa.effects.time_stretch(signal, rate=rate)

        def shift(signal, max_fraction=0.2):
            shift_amt = int(self._rng.uniform(-max_fraction, max_fraction) * len(signal))
            return np.roll(signal, shift_amt)

        def gain(signal, db):
            factor = np.power(10, db / 20)
            return signal * factor

        augmenters = [
            lambda sig: add_noise(sig, scale=self._rng.uniform(0.01, 0.05)),
            lambda sig: pitch_shift(sig, steps=self._rng.uniform(-3, 3)),
            lambda sig: time_stretch(sig, rate=self._rng.uniform(0.8, 1.25)),
            lambda sig: shift(sig, max_fraction=0.2),
            lambda sig: gain(sig, db=self._rng.uniform(-6, 6)),
        ]

        augmented = []
        target_len = self.sample_rate * self.duration
        needed = self.augmentation_factor - 1

        for i in range(needed):
            aug_fn = augmenters[i % len(augmenters)]
            try:
                aug = aug_fn(y)
                aug = self._pad_or_trim(aug)
                augmented.append(aug)
            except Exception:
                continue

        normalized = []
        for a in augmented:
            if len(a) != target_len:
                a = self._pad_or_trim(a)
            normalized.append(a)

        return normalized

    def extract_features(self, audio_path):
        """Извлекает признаки из аудио файла."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            return self._extract_features_from_waveform(y, sr)
        except Exception as e:
            print(f"Ошибка при обработке {audio_path}: {e}")
            return None

    def prepare_data(self, csv_path, audio_dir):
        """Подготавливает данные для обучения"""
        print("\n" + "=" * 60)
        print("ПОДГОТОВКА ДАННЫХ")
        print("=" * 60)
        
        df = pd.read_csv(csv_path)
        print(f"Загружено {len(df)} записей из CSV")
        
        X = []
        y = []
        failed = 0
        
        print("\nИзвлечение признаков из аудио файлов...")
        if self.augmentation_factor > 1:
            print(f"⚙️  Аугментация: {self.augmentation_factor}x на каждый файл")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка"):
            audio_path = os.path.join(audio_dir, row['path'])
            if os.path.exists(audio_path):
                try:
                    y_wave, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
                    base_features = self._extract_features_from_waveform(y_wave, sr)
                    if base_features is not None:
                        X.append(base_features)
                        y.append(row['emotion'])

                        for aug_wave in self._augment_waveforms(y_wave, sr):
                            aug_features = self._extract_features_from_waveform(aug_wave, sr)
                            if aug_features is not None:
                                X.append(aug_features)
                                y.append(row['emotion'])
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Ошибка при обработке {audio_path}: {e}")
                    failed += 1
            else:
                failed += 1

        print(f"\n✅ Успешно обработано: {len(X)} файлов")
        print(f"❌ Пропущено: {failed} файлов")
        
        X = np.array(X)
        y = np.array(y)
        
        # Кодирование меток
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nКлассы эмоций: {self.label_encoder.classes_}")
        print(f"Размерность признаков: {X.shape[1]}")
        
        return X, y_encoded, y
    
    def build_model(self, input_dim, num_classes):
        """Создает нейронную сеть"""
        print("\n" + "=" * 60)
        print("СОЗДАНИЕ МОДЕЛИ")
        print("=" * 60)
        
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nАрхитектура модели:")
        model.summary()
        
        return model
    
    def train(self, X, y, epochs=1000, batch_size=32, validation_split=0.2):
        """Обучает модель"""
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)
        
        # Разделение на train и validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Обучающая выборка: {len(X_train)}")
        print(f"Валидационная выборка: {len(X_val)}")
        
        # Создание модели
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_model(X.shape[1], num_classes)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=50,
            min_lr=0.0001
        )
        
        # Обучение
        print("\n🚀 Начало обучения...\n")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Вывод статистик
        print("\n" + "=" * 80)
        print(" " * 25 + "СТАТИСТИКИ ОБУЧЕНИЯ")
        print("=" * 80)
        
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        best_train_acc = max(history.history['accuracy'])
        
        print(f"\n{'='*80}")
        print(f"{'📊 ОСНОВНЫЕ МЕТРИКИ':^80}")
        print(f"{'='*80}")
        print(f"{'Метрика':<40} {'Значение':>20} {'Процент':>20}")
        print(f"{'-'*80}")
        print(f"{'Точность на обучении (финальная)':<40} {final_train_acc:>20.4f} {final_train_acc*100:>19.2f}%")
        print(f"{'Точность на обучении (лучшая)':<40} {best_train_acc:>20.4f} {best_train_acc*100:>19.2f}%")
        print(f"{'Точность на валидации (финальная)':<40} {final_val_acc:>20.4f} {final_val_acc*100:>19.2f}%")
        print(f"{'Точность на валидации (лучшая)':<40} {best_val_acc:>20.4f} {best_val_acc*100:>19.2f}%")
        print(f"{'Лучшая эпоха':<40} {best_epoch:>20}")
        
        print(f"\n{'='*80}")
        print(f"{'📈 ИСТОРИЯ ОБУЧЕНИЯ':^80}")
        print(f"{'='*80}")
        print(f"{'Параметр':<40} {'Значение':>40}")
        print(f"{'-'*80}")
        print(f"{'Всего эпох обучено':<40} {len(history.history['loss']):>40}")
        print(f"{'Финальная loss (обучение)':<40} {history.history['loss'][-1]:>40.4f}")
        print(f"{'Финальная loss (валидация)':<40} {history.history['val_loss'][-1]:>40.4f}")
        print(f"{'Минимальная loss (валидация)':<40} {min(history.history['val_loss']):>40.4f}")
        print(f"{'Эпоха с минимальной loss':<40} {history.history['val_loss'].index(min(history.history['val_loss'])) + 1:>40}")
        
        # Динамика обучения
        print(f"\n{'='*80}")
        print(f"{'📉 ДИНАМИКА ОБУЧЕНИЯ (первые 5 и последние 5 эпох)':^80}")
        print(f"{'='*80}")
        print(f"{'Эпоха':<8} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<12} {'Val Loss':<12}")
        print(f"{'-'*80}")
        
        epochs_list = list(range(1, len(history.history['loss']) + 1))
        for i in [0, 1, 2, 3, 4]:
            if i < len(epochs_list):
                ep = epochs_list[i]
                print(f"{ep:<8} {history.history['accuracy'][i]:<12.4f} {history.history['val_accuracy'][i]:<12.4f} "
                      f"{history.history['loss'][i]:<12.4f} {history.history['val_loss'][i]:<12.4f}")
        
        if len(epochs_list) > 5:
            print(f"{'...':^80}")
            for i in range(-5, 0):
                ep = epochs_list[i]
                idx = len(epochs_list) + i
                print(f"{ep:<8} {history.history['accuracy'][idx]:<12.4f} {history.history['val_accuracy'][idx]:<12.4f} "
                      f"{history.history['loss'][idx]:<12.4f} {history.history['val_loss'][idx]:<12.4f}")
        
        # Предсказания на валидации
        print(f"\n{'='*80}")
        print(f"{'📋 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ':^80}")
        print(f"{'='*80}")
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        report = classification_report(
            y_val, 
            y_pred_classes, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        print(f"\n{'Класс':<20} {'Precision':<15} {'Recall':<15} {'F1-score':<15} {'Support':<15}")
        print(f"{'-'*80}")
        for emotion in self.label_encoder.classes_:
            if emotion in report:
                print(f"{emotion:<20} {report[emotion]['precision']:<15.4f} {report[emotion]['recall']:<15.4f} "
                      f"{report[emotion]['f1-score']:<15.4f} {report[emotion]['support']:<15}")
        
        print(f"{'-'*80}")
        print(f"{'Среднее (macro)':<20} {report['macro avg']['precision']:<15.4f} {report['macro avg']['recall']:<15.4f} "
              f"{report['macro avg']['f1-score']:<15.4f} {report['macro avg']['support']:<15}")
        print(f"{'Среднее (weighted)':<20} {report['weighted avg']['precision']:<15.4f} {report['weighted avg']['recall']:<15.4f} "
              f"{report['weighted avg']['f1-score']:<15.4f} {report['weighted avg']['support']:<15}")
        
        # Матрица ошибок
        print(f"\n{'='*80}")
        print(f"{'🔢 МАТРИЦА ОШИБОК (CONFUSION MATRIX)':^80}")
        print(f"{'='*80}")
        cm = confusion_matrix(y_val, y_pred_classes)
        print(f"\n{'Предсказано →':<15}", end="")
        for emotion in self.label_encoder.classes_:
            print(f"{emotion[:8]:<10}", end="")
        print()
        print("-" * (15 + 10 * len(self.label_encoder.classes_)))
        for i, emotion in enumerate(self.label_encoder.classes_):
            print(f"{'Реально ↓':<5} {emotion[:8]:<6}", end="")
            for j in range(len(self.label_encoder.classes_)):
                print(f"{cm[i][j]:<10}", end="")
            print()

        # Визуализация метрик
        cm_path = "confusion_matrix.png"
        curves_path = "training_curves.png"
        self._plot_confusion_matrix(cm, self.label_encoder.classes_, cm_path)
        self._plot_training_curves(history.history, curves_path)

        # Анализ и отчёт
        analysis_text = self._build_analysis_text(
            report,
            final_train_acc,
            final_val_acc,
            best_val_acc,
            cm,
            self.label_encoder.classes_,
        )
        model_summary = self._get_model_summary()
        self._save_markdown_report(
            model_summary=model_summary,
            final_train_acc=final_train_acc,
            final_val_acc=final_val_acc,
            best_val_acc=best_val_acc,
            best_train_acc=best_train_acc,
            history_dict=history.history,
            classification_report_dict=report,
            confusion_matrix_path=cm_path,
            curves_path=curves_path,
            analysis_text=analysis_text,
            encoder_classes=self.label_encoder.classes_,
        )
        
        print(f"\n{'='*80}")
        print(f"{'✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО':^80}")
        print(f"{'='*80}\n")
        
        return history
    
    def save_model(self, model_path='emotion_model.h5', encoder_path='label_encoder.json'):
        """Сохраняет модель и энкодер"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"\n✅ Модель сохранена: {model_path}")
            
            # Сохранение энкодера
            encoder_dict = {
                'classes': self.label_encoder.classes_.tolist()
            }
            with open(encoder_path, 'w', encoding='utf-8') as f:
                json.dump(encoder_dict, f, ensure_ascii=False, indent=2)
            print(f"✅ Энкодер сохранен: {encoder_path}")
    
    def load_model(self, model_path='emotion_model.h5', encoder_path='label_encoder.json'):
        """Загружает модель и энкодер"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Файл энкодера не найден: {encoder_path}")
        
        self.model = keras.models.load_model(model_path)
        
        with open(encoder_path, 'r', encoding='utf-8') as f:
            encoder_dict = json.load(f)
        self.label_encoder.classes_ = np.array(encoder_dict['classes'])
        
        print(f"✅ Модель загружена: {model_path}")
        print(f"✅ Энкодер загружен: {encoder_path}")
        print(f"✅ Классы: {', '.join(self.label_encoder.classes_)}")
    
    def predict(self, audio_path):
        """Предсказывает эмоцию для аудио файла"""
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        features = features.reshape(1, -1)
        prediction = self.model.predict(features, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_emotion = self.label_encoder.classes_[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        # Все вероятности
        probabilities = {
            emotion: float(prob) 
            for emotion, prob in zip(self.label_encoder.classes_, prediction[0])
        }
        
        return {
            'emotion': predicted_emotion,
            'confidence': float(confidence),
            'probabilities': probabilities
        }


def main():
    """Основная функция для обучения"""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ ЭМОЦИЙ")
    print("=" * 60)
    
    classifier = EmotionClassifier(augmentation_factor=10)
    
    # Подготовка данных
    X, y_encoded, y_labels = classifier.prepare_data('train.csv', 'train')
    
    # Обучение
    history = classifier.train(X, y_encoded, epochs=1000, batch_size=32)
    
    # Сохранение модели
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()
