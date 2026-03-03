# CLI Reference

Все команды llm-forge доступны через точку входа `forge`. CLI построен на [Click](https://click.palletsprojects.com/) с подсветкой через [Rich](https://rich.readthedocs.io/).

## Глобальные опции

```
forge [--verbose/-v] [--version] <command>
```

| Опция | Тип | Описание |
|-------|-----|----------|
| `--verbose`, `-v` | flag | Включить отладочный вывод (DEBUG-уровень логирования) |
| `--version` | flag | Показать версию пакета и выйти |

---

## forge train

Запуск обучения модели (SFT или DPO).

```bash
forge train <config.yaml> [overrides...] [--task sft|dpo|auto] [--base-model PATH] [--resume PATH]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `config.yaml` | PATH (аргумент) | -- | Путь к YAML-конфигу эксперимента |
| `overrides` | key=value... | -- | CLI-переопределения параметров конфига |
| `--task` | `sft` / `dpo` / `auto` | `auto` | Тип задачи обучения. `auto` определяет по конфигу |
| `--base-model` | PATH | `None` | Путь к SFT-адаптеру (для DPO-обучения) |
| `--resume` | PATH | `None` | Возобновить обучение из директории чекпоинта |

!!! tip "CLI-переопределения"
    Любой параметр конфига можно переопределить через `key=value` аргументы.
    Они имеют наивысший приоритет и применяются поверх всех `inherit`-конфигов.

**Примеры:**

=== "SFT-обучение"

    ```bash
    forge train experiments/cam-sft.yaml
    ```

=== "DPO-обучение"

    ```bash
    forge train experiments/cam-dpo.yaml \
      --task dpo \
      --base-model ./outputs/cam-sft
    ```

=== "С переопределениями"

    ```bash
    forge train experiments/cam-sft.yaml \
      learning_rate=1e-4 \
      epochs=5 \
      batch_size=4
    ```

---

## forge eval

Оценка обученной модели на тестовых данных.

```bash
forge eval --model PATH --test-data PATH [--config PATH] [--batch-size N] [--output PATH]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `--model` | PATH | **обязательный** | Путь к модели или директории с адаптером |
| `--test-data` | PATH | **обязательный** | Путь к тестовому датасету |
| `--config` | PATH | `None` | Конфиг с настройками оценки |
| `--batch-size` | int | `8` | Размер батча для инференса |
| `--output` | PATH | `None` | Директория для отчёта об оценке |

**Примеры:**

=== "Базовая оценка"

    ```bash
    forge eval \
      --model ./outputs/cam-sft/lora \
      --test-data data/test.csv
    ```

=== "С отчётом"

    ```bash
    forge eval \
      --model ./outputs/cam-sft/lora \
      --test-data data/test.csv \
      --output reports/ \
      --batch-size 16
    ```

---

## forge export

Экспорт модели в продакшен-формат.

```bash
forge export --model PATH [--format gguf|merged|hub] [--quant q4_k_m|q8_0|f16] [--output PATH] [--config PATH]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `--model` | PATH | **обязательный** | Путь к модели или директории с адаптером |
| `--format` | `gguf` / `merged` / `hub` | `gguf` | Формат экспорта |
| `--quant` | `q4_k_m` / `q8_0` / `f16` | `q4_k_m` | Уровень квантизации для GGUF |
| `--output` | PATH | `None` | Путь для экспортированной модели |
| `--config` | PATH | `None` | Конфиг с настройками экспорта |

!!! info "Форматы экспорта"
    - **gguf** -- квантизированный формат для llama.cpp и Ollama
    - **merged** -- полная модель с вмёрженным LoRA-адаптером
    - **hub** -- публикация на HuggingFace Hub (требует `HF_TOKEN`)

**Примеры:**

=== "GGUF q4_k_m"

    ```bash
    forge export \
      --model ./outputs/cam-sft/lora \
      --format gguf \
      --quant q4_k_m
    ```

=== "Merged модель"

    ```bash
    forge export \
      --model ./outputs/cam-sft/lora \
      --format merged \
      --output ./merged/
    ```

=== "Push to Hub"

    ```bash
    forge export \
      --model ./outputs/cam-sft/lora \
      --format hub
    ```

---

## forge serve

Запуск сервера для инференса модели.

```bash
forge serve --model PATH [--port 8080] [--backend llamacpp|vllm] [--host 0.0.0.0]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `--model` | PATH | **обязательный** | Путь к файлу модели (GGUF) или директории |
| `--port` | int | `8080` | Порт сервера |
| `--backend` | `llamacpp` / `vllm` | `llamacpp` | Бэкенд для сервинга |
| `--host` | string | `0.0.0.0` | Хост сервера |

**Примеры:**

=== "llama.cpp"

    ```bash
    forge serve \
      --model ./outputs/model.gguf \
      --port 8080
    ```

=== "vLLM"

    ```bash
    forge serve \
      --model ./outputs/cam-sft \
      --backend vllm \
      --port 8000
    ```

---

## forge init

Создание нового конфига эксперимента.

```bash
forge init <name> [--task sft|dpo] [--model qwen2.5-3b|llama3.2-1b|mistral-7b] [--output-dir PATH]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `name` | string (аргумент) | -- | Имя эксперимента |
| `--task` | `sft` / `dpo` | `sft` | Тип задачи обучения |
| `--model` | `qwen2.5-3b` / `llama3.2-1b` / `mistral-7b` | `qwen2.5-3b` | Базовая модель |
| `--output-dir` | PATH | `./outputs/<name>` | Директория для результатов |

!!! note "Генерируемые файлы"
    Конфиг создаётся в `configs/experiments/<name>.yaml` и наследует
    базовые настройки через механизм `inherit`.

**Примеры:**

=== "SFT-классификатор"

    ```bash
    forge init my-classifier
    ```

=== "DPO-чатбот"

    ```bash
    forge init my-chatbot \
      --task dpo \
      --model llama3.2-1b
    ```

---

## forge info

Показать информацию об обнаруженном оборудовании и рекомендуемую стратегию обучения.

```bash
forge info
```

Команда не принимает аргументов. Выводит таблицу с данными GPU: имя, VRAM, compute capability, поддержка BF16, рекомендуемая стратегия, batch size и gradient accumulation.

**Пример:**

```bash
forge info
```

```
┌───────────────────────────────────┐
│         Hardware Info             │
├─────────────────────┬─────────────┤
│ GPUs                │ 1           │
│ GPU Name            │ RTX 4090    │
│ VRAM per GPU        │ 24.0 GB     │
│ Compute Capability  │ 8.9         │
│ BF16 Supported      │ True        │
│ Recommended Strategy│ unsloth     │
│ Recommended Batch   │ 4           │
│ Recommended Grad Ac │ 4           │
└─────────────────────┴─────────────┘
```

---

## forge ui

Запуск веб-интерфейса (Dashboard).

```bash
forge ui [--host 0.0.0.0] [--port 8888]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `--host` | string | `0.0.0.0` | Хост сервера |
| `--port` | int | `8888` | Порт сервера |

**Примеры:**

=== "По умолчанию"

    ```bash
    forge ui
    ```

=== "Кастомный порт"

    ```bash
    forge ui --port 9000
    ```

После запуска: Dashboard -- `http://localhost:8888`, API docs -- `http://localhost:8888/docs`.

---

## forge sweep

Запуск оптимизации гиперпараметров (HPO) через Optuna.

```bash
forge sweep <config.yaml> <sweep_config.yaml> [--n-trials N] [--name NAME]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `config.yaml` | PATH (аргумент) | -- | Базовый конфиг эксперимента |
| `sweep_config.yaml` | PATH (аргумент) | -- | Конфиг поиска гиперпараметров |
| `--n-trials` | int | из конфига | Количество триалов (переопределяет конфиг) |
| `--name` | string | `None` | Имя Optuna-study |

**Примеры:**

=== "Базовый sweep"

    ```bash
    forge sweep \
      configs/experiments/sft.yaml \
      configs/sweeps/lr-search.yaml
    ```

=== "30 триалов"

    ```bash
    forge sweep \
      configs/experiments/sft.yaml \
      configs/sweeps/full.yaml \
      --n-trials 30 \
      --name lr-and-epochs
    ```

---

## forge agent

Подсистема агентов: создание, тестирование и деплой AI-агентов с инструментами.

### forge agent init

Создать конфиг нового агента.

```bash
forge agent init <name> [--model-url URL] [--model-name NAME]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `name` | string (аргумент) | -- | Имя агента |
| `--model-url` | string | `http://localhost:8080/v1` | URL сервера модели |
| `--model-name` | string | `default` | Имя модели на сервере |

**Примеры:**

=== "Базовый агент"

    ```bash
    forge agent init my-assistant
    ```

=== "С Ollama"

    ```bash
    forge agent init code-helper \
      --model-url http://localhost:11434/v1
    ```

### forge agent test

Интерактивный REPL для тестирования агента.

```bash
forge agent test <config.yaml> [--native-tools]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `config.yaml` | PATH (аргумент) | -- | Путь к конфигу агента |
| `--native-tools` | flag | `False` | Использовать native tool calling вместо ReAct |

**Примеры:**

=== "ReAct-режим"

    ```bash
    forge agent test configs/agents/my-assistant.yaml
    ```

=== "Native tools"

    ```bash
    forge agent test configs/agents/my-assistant.yaml \
      --native-tools
    ```

### forge agent serve

Запуск REST API-сервера агента.

```bash
forge agent serve <config.yaml> [--host 0.0.0.0] [--port 8081]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `config.yaml` | PATH (аргумент) | -- | Путь к конфигу агента |
| `--host` | string | `0.0.0.0` | Хост сервера |
| `--port` | int | `8081` | Порт сервера |

**Примеры:**

=== "По умолчанию"

    ```bash
    forge agent serve configs/agents/my-assistant.yaml
    ```

=== "Кастомный порт"

    ```bash
    forge agent serve configs/agents/my-assistant.yaml \
      --port 9000
    ```

!!! info "Эндпоинты агента"
    После запуска доступны:

    - `POST /v1/agent/chat` -- отправка сообщения
    - `GET /v1/agent/health` -- проверка состояния

---

## forge pipeline

Оркестратор многоэтапных пайплайнов.

### forge pipeline run

Запуск пайплайна из YAML-конфига.

```bash
forge pipeline run <config.yaml>
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `config.yaml` | PATH (аргумент) | -- | Путь к YAML-конфигу пайплайна |

**Примеры:**

```bash
forge pipeline run configs/pipelines/example.yaml
```

### forge pipeline list

Список прошлых запусков пайплайнов.

```bash
forge pipeline list [--name NAME]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `--name` | string | `None` | Фильтр по имени пайплайна |

**Примеры:**

=== "Все запуски"

    ```bash
    forge pipeline list
    ```

=== "По имени"

    ```bash
    forge pipeline list --name full-pipeline
    ```

---

## forge runs

Управление записями экспериментов.

### forge runs list

Список записей экспериментов.

```bash
forge runs list [--project X] [--status Y] [--limit N]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `--project` | string | `None` | Фильтр по имени проекта |
| `--status` | string | `None` | Фильтр по статусу (`completed`, `failed`, `running`) |
| `--limit` | int | `20` | Максимальное количество записей |

**Примеры:**

=== "Последние 10 завершённых"

    ```bash
    forge runs list --status completed --limit 10
    ```

=== "По проекту"

    ```bash
    forge runs list --project customer-intent
    ```

### forge runs show

Детали конкретного запуска.

```bash
forge runs show <run_id>
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `run_id` | string (аргумент) | -- | Идентификатор запуска |

**Пример:**

```bash
forge runs show abc123def456
```

### forge runs compare

Сравнение нескольких запусков.

```bash
forge runs compare <run_id1> <run_id2> ...
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `run_ids` | string... (аргументы) | -- | Два или более ID запусков для сравнения |

**Примеры:**

=== "Два запуска"

    ```bash
    forge runs compare abc123 def456
    ```

=== "Три запуска"

    ```bash
    forge runs compare run1 run2 run3
    ```

---

## forge registry

Реестр моделей: регистрация, продвижение по стадиям, листинг.

### forge registry list

Список зарегистрированных моделей.

```bash
forge registry list [--name X] [--status Y]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `--name` | string | `None` | Фильтр по имени модели |
| `--status` | string | `None` | Фильтр по статусу (`staging`, `production`, `archived`) |

**Примеры:**

=== "Все модели"

    ```bash
    forge registry list
    ```

=== "Продакшен-модели"

    ```bash
    forge registry list \
      --name customer-intent \
      --status production
    ```

### forge registry register

Регистрация модели в реестре.

```bash
forge registry register <name> --model-path PATH [--task sft] [--base-model NAME] [--tag TAG...]
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `name` | string (аргумент) | -- | Имя модели |
| `--model-path` | PATH | **обязательный** | Путь к модели/адаптеру |
| `--task` | string | `sft` | Тип задачи обучения |
| `--base-model` | string | `""` | Имя базовой модели |
| `--tag` | string (повторяемая) | -- | Теги (можно указывать несколько раз) |

**Пример:**

```bash
forge registry register customer-intent \
  --model-path ./outputs/sft/lora \
  --base-model qwen2.5-3b \
  --tag v1 \
  --tag production-ready
```

### forge registry promote

Продвижение модели по стадиям жизненного цикла.

```bash
forge registry promote <model_id> staging|production|archived
```

| Опция | Тип | По умолчанию | Описание |
|-------|-----|-------------|----------|
| `model_id` | string (аргумент) | -- | ID модели в реестре |
| `status` | `staging` / `production` / `archived` | -- | Целевой статус |

!!! warning "Необратимое действие"
    Перевод в `archived` означает, что модель больше не используется в продакшене.
    Перед этим убедитесь, что другая модель уже в `production`.

**Примеры:**

=== "В staging"

    ```bash
    forge registry promote customer-intent-v2 staging
    ```

=== "В production"

    ```bash
    forge registry promote customer-intent-v2 production
    ```
