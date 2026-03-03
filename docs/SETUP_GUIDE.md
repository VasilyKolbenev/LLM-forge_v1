# llm-forge: Setup Guide

Пошаговая инструкция по установке и использованию платформы llm-forge для файнтюнинга LLM.

## Требования

### Hardware
- **GPU**: NVIDIA с поддержкой CUDA (минимум 6 GB VRAM для моделей 0.8-2B)
- **RAM**: 16+ GB (рекомендуется 32 GB)
- **Диск**: 50+ GB свободного места (модели, датасеты, адаптеры)

### Software
- **Python** 3.10+ (рекомендуется 3.12-3.14)
- **Node.js** 18+ (для Web UI)
- **Git**
- **CUDA Toolkit** (совместимый с вашей версией PyTorch)
- Драйверы NVIDIA последней версии

---

## 1. Клонирование репозитория

```bash
git clone https://github.com/<your-org>/llm-forge.git
cd llm-forge
```

---

## 2. Установка Python-зависимостей

### Рекомендуемый вариант (с UI и eval):

```bash
# Создаём виртуальное окружение
python -m venv .venv

# Активируем (Linux/Mac)
source .venv/bin/activate

# Активируем (Windows)
.venv\Scripts\activate

# Устанавливаем пакет с зависимостями
pip install -e ".[ui,eval]"
```

### Варианты установки:

| Команда | Что включает |
|---------|-------------|
| `pip install -e .` | Базовый CLI (train, eval, export) |
| `pip install -e ".[ui]"` | + Web UI (FastAPI, uvicorn) |
| `pip install -e ".[eval]"` | + Графики eval (seaborn, matplotlib) |
| `pip install -e ".[unsloth]"` | + Unsloth (2-5x ускорение, Linux only) |
| `pip install -e ".[all]"` | Все зависимости |

### Дополнительно (для UI):

```bash
pip install python-dotenv openai slowapi
```

---

## 3. Установка UI (Web Dashboard)

```bash
cd ui
npm install
cd ..
```

---

## 4. Настройка окружения

Создайте файл `.env` в корне проекта:

```bash
# .env

# (Опционально) OpenAI API ключ для Co-pilot чата в UI
OPENAI_API_KEY=sk-your-key-here

# (Опционально) Кастомные CORS origins
# FORGE_CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# (Опционально) Включить аутентификацию API
# FORGE_AUTH_ENABLED=true
```

---

## 5. Запуск платформы

### 5.1 Backend (API сервер)

```bash
python -c "from llm_forge.ui.app import create_app; import uvicorn; uvicorn.run(create_app(), host='0.0.0.0', port=8888)"
```

Или короче:
```bash
python -m llm_forge.ui.app
```

Backend доступен на `http://localhost:8888`.

### 5.2 Frontend (Web UI)

В отдельном терминале:

```bash
cd ui
npm run dev
```

UI доступен на `http://localhost:5173`.

### 5.3 Проверка

Откройте в браузере `http://localhost:5173` — должен загрузиться дашборд.
Проверьте API: `curl http://localhost:8888/api/v1/health` — ответ `{"status": "ok"}`.

---

## 6. Подготовка данных

### Формат датасета

llm-forge поддерживает CSV, JSONL, JSON, Parquet. Минимальный CSV:

```csv
phrase,domain,skill
"Оплатить коммуналку",HOUSE,utility_bill
"Когда придёт посылка",DELIVERY,tracking
"Привет!",BOLTALKA,greeting
```

Поместите файл в директорию `data/`:

```bash
cp your_dataset.csv data/my_intents.csv
```

### System Prompt (опционально)

Если нужен system prompt для модели, создайте текстовый файл:

```bash
# prompts/my_system_prompt.txt
You are an intent classifier. Given a user message, respond with JSON:
{"domain": "<DOMAIN>", "skill": "<SKILL>"}
```

---

## 7. Скачивание модели

Модели скачиваются автоматически с HuggingFace при первом запуске.
Убедитесь, что есть доступ к интернету.

### Предустановленные конфиги моделей:

| Конфиг | Модель | VRAM (QLoRA) |
|--------|--------|-------------|
| `models/qwen3.5-0.8b` | Qwen/Qwen3.5-0.8B | ~2-3 GB |
| `models/qwen3.5-2b` | Qwen/Qwen3.5-2B | ~4-5 GB |
| `models/qwen3.5-4b` | Qwen/Qwen3.5-4B | ~6-7 GB |
| `models/llama3.2-1b` | meta-llama/Llama-3.2-1B-Instruct | ~3-4 GB |
| `models/qwen2.5-3b` | Qwen/Qwen2.5-3B-Instruct | ~4-5 GB |
| `models/mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 | ~8-10 GB |

### Использование своей модели

Создайте конфиг `configs/models/my-model.yaml`:

```yaml
model:
  name: your-org/your-model-name  # HuggingFace model ID
  family: llama  # или qwen3, mistral
  max_seq_length: 4096
  chat_template: chatml  # или llama3, mistral

load_in_4bit: true
gradient_checkpointing: true

lora:
  r: 16
  lora_alpha: 16
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  lora_dropout: 0
  bias: none
```

---

## 8. Создание конфига эксперимента

Создайте YAML файл в `configs/examples/`:

```yaml
# configs/examples/my-experiment.yaml

inherit:
  - base                   # Базовые настройки (optimizer, seed, etc.)
  - models/qwen3.5-0.8b   # Модель (можно заменить на свою)

task: sft  # sft или dpo

dataset:
  path: data/my_intents.csv
  format: csv
  text_column: phrase              # Колонка с текстом
  label_columns:                   # Колонки с метками
    - domain
    - skill
  system_prompt_file: prompts/my_system_prompt.txt
  test_size: 0.15                  # 15% данных на тест

training:
  epochs: 3          # Кол-во эпох
  learning_rate: 2e-4  # Learning rate
  batch_size: 1        # Размер батча (увеличить если хватает VRAM)
  gradient_accumulation: 16  # Эффективный batch = batch_size * grad_accum

output:
  dir: ./outputs/my-experiment
  save_adapter: true
```

### Ключевые параметры:

| Параметр | Описание | Рекомендация |
|----------|----------|-------------|
| `epochs` | Количество эпох | 3-5 для маленьких датасетов |
| `learning_rate` | Скорость обучения | 2e-4 (0.8B), 1e-4 (2-4B), 5e-5 (7B+) |
| `batch_size` | Размер батча | 1-2 (8GB), 2-4 (16GB), 4-8 (24GB+) |
| `gradient_accumulation` | Шаги накопления | 16 при batch_size=1 |
| `lora.r` | LoRA rank | 8 (быстро), 16 (баланс), 32 (качество) |

---

## 9. Запуск обучения

### Через CLI:

```bash
forge train configs/examples/my-experiment.yaml
```

С переопределением параметров:
```bash
forge train configs/examples/my-experiment.yaml epochs=5 learning_rate=1e-4
```

### Через Web UI:

1. Откройте `http://localhost:5173`
2. Перейдите в **New Experiment**
3. Выберите модель, загрузите датасет, настройте параметры
4. Нажмите **Start Training**

### Через API:

```python
import requests
from llm_forge.config import load_config

config = load_config('configs/examples/my-experiment.yaml')
resp = requests.post('http://localhost:8888/api/v1/training/start', json={
    'name': 'my-experiment',
    'config': config,
    'task': 'sft'
})
print(resp.json())  # {"job_id": "...", "experiment_id": "...", "status": "running"}
```

Тренировка идёт в фоне. Прогресс виден в UI на странице **Experiments**.

---

## 10. Оценка модели (Eval)

После обучения запустите eval:

```bash
python scripts/run_eval.py \
  --model Qwen/Qwen3.5-0.8B \
  --adapter outputs/my-experiment/lora \
  --test-data data/my_intents_test.csv \
  --experiment-id <ID из UI>
```

Результаты:
- **Accuracy** — общая точность
- **JSON Parse Rate** — % корректных JSON ответов
- **F1 (weighted)** — взвешенный F1
- **Confusion Matrix** — матрица ошибок по классам

Результаты автоматически сохраняются в experiment store и видны в UI.

---

## 11. DPO (опционально)

DPO (Direct Preference Optimization) — второй этап после SFT.
Нужны пары "предпочтительный/непредпочтительный" ответы.

### Формат DPO пар (JSONL):

```jsonl
{"prompt": "Оплатить газ", "chosen": "{\"domain\": \"HOUSE\", \"skill\": \"utility_bill\"}", "rejected": "{\"domain\": \"PAYMENTS\", \"skill\": \"payment_status\"}"}
```

### Конфиг DPO:

```yaml
# configs/examples/my-dpo.yaml
inherit:
  - base
  - models/qwen3.5-0.8b
  - tasks/dpo

sft_adapter_path: ./outputs/my-sft/lora  # Путь к SFT адаптеру

dpo:
  beta: 0.1
  max_length: 512
  pairs_path: ./data/my_dpo_pairs.jsonl

output:
  dir: ./outputs/my-dpo
```

```bash
forge train configs/examples/my-dpo.yaml
```

---

## 12. Экспорт модели

### GGUF (для llama.cpp, Ollama):

```bash
forge export --model ./outputs/my-experiment/lora --format gguf --quant q4_k_m
```

### Merge LoRA + Base:

```bash
forge export --model ./outputs/my-experiment/lora --format merged
```

---

## 13. Serving (запуск модели как API)

```bash
forge serve --model ./outputs/model-q4_k_m.gguf --port 8080
```

---

## Структура проекта

```
llm-forge/
  configs/
    base.yaml              # Дефолтные настройки
    models/                # Конфиги моделей (qwen, llama, mistral)
    tasks/                 # Конфиги задач (sft, dpo, eval)
    examples/              # Готовые эксперименты
  data/                    # Датасеты (.csv, .jsonl)
  prompts/                 # System prompts
  outputs/                 # Результаты обучения (адаптеры, модели)
  src/llm_forge/           # Python backend
  ui/                      # React frontend
  scripts/                 # Утилиты (run_eval.py и др.)
  .env                     # Переменные окружения (не в git!)
```

## Troubleshooting

### CUDA Out of Memory
- Уменьшите `batch_size` до 1
- Увеличьте `gradient_accumulation`
- Используйте модель поменьше
- Убедитесь что `load_in_4bit: true`

### Модель не скачивается
- Проверьте интернет-соединение
- Для gated моделей (Llama) нужен `HF_TOKEN`:
  ```bash
  export HF_TOKEN=hf_your_token_here
  ```

### UI не подключается к backend
- Убедитесь что backend запущен на порту 8888
- Проверьте CORS: `FORGE_CORS_ORIGINS=http://localhost:5173`

### Тренировка зависает
- Проверьте `nvidia-smi` — GPU должен быть загружен
- Проверьте логи backend: `tail -f backend.log`
