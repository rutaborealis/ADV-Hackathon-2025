# Система валидации видеорекламы

Автоматизированная система для проверки соответствия видеорекламных материалов техническим и юридическим требованиям.

## Назначение

Система выполняет комплексный анализ видеофайлов:
- **Техническая валидация** — проверка параметров видео/аудио
- **Контент-анализ** — извлечение и анализ содержимого
- **Юридическая проверка** — AI-анализ соответствия требованиям

## Архитектура

### Модульная структура
```
┌─────────────────────────────────────────────────────────┐
│                    MAIN APPLICATION                     │
│  main.py — координация всех процессов                   │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│                CORE VALIDATION LAYER                    │
│  • validate_video — технические параметры               │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│               CONTENT ANALYSIS LAYER                    │
│  • extract_audio — извлечение аудио                     │
│  • extract_frames — извлечение кадров                   │
│  • transcribe_audio — распознавание речи                │
│  • ocr_text_from_frames — распознавание текста          │
│  • detect_qr_codes — детекция QR-кодов                  │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│                  AI ANALYSIS LAYER                      │
│  • legal_ai_check — юридический анализ                  │
│  • save_technical_validation_report — отчёты            │
└─────────────────────────────────────────────────────────┘
```

### Поток обработки
1. **Валидация** → Проверка технических параметров
2. **Извлечение** → Аудио, кадры, текст, QR-коды
3. **AI-анализ** → Юридическая проверка через GPT-4
4. **Отчетность** → Генерация файлов с результатами

## Технологический стек

- **Python 3.x** — основной язык разработки
- **PyMediaInfo** — анализ метаданных медиафайлов
- **MoviePy** — обработка видео/аудио
- **OpenCV** — компьютерное зрение
- **Whisper** — распознавание речи
- **EasyOCR** — OCR распознавание
- **PyZBar** — детекция QR-кодов
- **OpenAI GPT-4.1** — AI-анализ

### Зависимости

См. `requirements.txt` для полного списка. Основные:
- openai
- whisper
- moviepy
- opencv-python
- easyocr
- pyzbar
- Pillow
- pymediainfo
- numpy

## Технические требования

### Входные данные
- **Формат:** MXF (Material Exchange Format)
- **Видео:** Full HD (1920x1080), 25 fps, чересстрочная развертка
- **Аудио:** PCM, 2 канала, 48 кГц, 24 бит
- **Битрейт:** 50 Мбит/с

### Проверяемые параметры
- Контейнер (MXF)
- Видеокодек (MPEG Video)
- Разрешение (1920x1080)
- Частота кадров (25 fps)
- Соотношение сторон (16:9)
- Тип развертки (чересстрочная)
- Глубина цвета (8 бит)
- Цветовое пространство (YUV)
- Хроматическая субдискретизация (4:2:2)
- Битрейт (50 Мбит/с)
- Аудиоформат (PCM, 2 канала, 48 кГц, 24 бит)

## Выходные данные

### Структура выходных данных

После анализа для каждого видеофайла создаются отдельные папки (имя зависит от исходного файла):

- `<basename>_to_openai_api/` — текстовые данные для анализа:
  - `validation_report.txt` — техническая валидация
  - `transcript.txt` — транскрипция аудио
  - `ocr_results.txt` — текст с экрана (OCR)
  - `qr_codes.json` — найденные QR-коды
  - `prompt.txt` — промпт для AI/ChatGPT

- `<basename>_extracted_media/` — извлечённые медиа:
  - `frames/` — все извлечённые кадры (jpg)
  - `audio.wav` — извлечённая аудиодорожка
  - `prompt.txt` — промпт для AI/ChatGPT

- `<basename>_all_reports/` — все отчёты:
  - `technical_validation_YYYYMMDD_HHMMSS.txt` — технический отчёт
  - `openai_analysis_YYYYMMDD_HHMMSS.txt` — AI-анализ (юридическая проверка)

### Пример структуры после запуска:
```
sovkombank_10sec_to_openai_api/
  ├── validation_report.txt
  ├── transcript.txt
  ├── ocr_results.txt
  ├── qr_codes.json
  └── prompt.txt
sovkombank_10sec_extracted_media/
  ├── frames/
  ├── audio.wav
  └── prompt.txt
sovkombank_10sec_all_reports/
  ├── technical_validation_YYYYMMDD_HHMMSS.txt
  └── openai_analysis_YYYYMMDD_HHMMSS.txt
```

## Использование

### Установка
```bash
pip install -r requirements.txt
```

### Подготовка
- Поместите видеофайл (формат MXF) в корневую папку проекта.
- Проверьте наличие файлов `legal_requirements.txt` и `technical_requirements.txt` в корне.
- Убедитесь, что все зависимости установлены.
- Укажите свой OpenAI API ключ в файле `open_ai_api_key.txt` (или измените в коде).

### Запуск
```bash
python main.py
```

### Результаты
- Все результаты сохраняются в папки, связанные с именем видеофайла (см. выше).
- В консоли отображается ход выполнения и краткий отчёт.
- Для анализа в ChatGPT используйте содержимое папок `<basename>_to_openai_api/` и `<basename>_extracted_media/`.


