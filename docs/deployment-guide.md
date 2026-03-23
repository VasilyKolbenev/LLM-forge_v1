# Pulsar AI -- Руководство по развертыванию

Полное руководство по установке и развертыванию платформы Pulsar AI для разработки и production-окружений.

---

## Содержание

1. [Обзор архитектуры](#1-обзор-архитектуры)
2. [Системные требования](#2-системные-требования)
3. [Локальная разработка](#3-локальная-разработка)
4. [Docker deployment (development)](#4-docker-deployment-development)
5. [Docker deployment (production)](#5-docker-deployment-production)
6. [Kubernetes deployment](#6-kubernetes-deployment)
7. [Настройка базы данных](#7-настройка-базы-данных)
8. [Настройка аутентификации](#8-настройка-аутентификации)
9. [Настройка хранилища](#9-настройка-хранилища)
10. [SSL/TLS](#10-ssltls)
11. [Мониторинг и логирование](#11-мониторинг-и-логирование)
12. [Бэкап и восстановление](#12-бэкап-и-восстановление)
13. [Безопасность](#13-безопасность)
14. [Обновление](#14-обновление)
15. [Устранение неполадок](#15-устранение-неполадок)
16. [Переменные окружения](#16-переменные-окружения)

---

## 1. Обзор архитектуры

Pulsar AI состоит из следующих компонентов:

| Компонент | Технология | Порт | Описание |
|-----------|------------|------|----------|
| Backend API | FastAPI + Uvicorn | 8888 | REST API, SSE-стриминг, WebSocket |
| Frontend | React (Vite) | 3000 (dev) / встроен в backend (prod) | SPA-интерфейс |
| База данных | SQLite (dev) / PostgreSQL 16 (prod) | 5432 | Хранение экспериментов, пользователей, метаданных |
| Очередь задач | In-process (dev) / Redis 7 (prod) | 6379 | Асинхронные задачи обучения |
| Reverse proxy | Nginx 1.27 | 80, 443 | TLS-терминация, сжатие, проксирование |
| Метрики | Prometheus endpoint | /metrics | Системные и application-метрики |

### Схема взаимодействия

```
Клиент (браузер)
    |
    v
[Nginx :80/:443] --- TLS-терминация, gzip, security headers
    |
    v
[FastAPI :8888] --- REST API + SPA (статика)
    |           \
    v            v
[PostgreSQL]   [Redis]
    :5432       :6379
```

В production-режиме React frontend собирается в статические файлы и встраивается в Docker-образ backend. Nginx проксирует все запросы к FastAPI.

---

## 2. Системные требования

### Минимальные (разработка)

| Ресурс | Требование |
|--------|------------|
| CPU | 4 ядра |
| RAM | 16 ГБ |
| Диск | 50 ГБ SSD |
| GPU | Не требуется (CPU-only inference) |
| Python | 3.10+ (рекомендуется 3.12) |
| Node.js | 18+ (рекомендуется 22) |
| ОС | Linux (Ubuntu 22.04+), macOS 13+, Windows 10/11 (WSL2) |

### Рекомендуемые (production)

| Ресурс | Требование |
|--------|------------|
| CPU | 8+ ядер |
| RAM | 32+ ГБ |
| GPU | NVIDIA A100/H100 (для обучения), NVIDIA T4/L4 (для inference) |
| Диск | 200+ ГБ NVMe SSD |
| Python | 3.12 |
| Node.js | 22 LTS |
| ОС | Ubuntu 22.04 LTS / 24.04 LTS |
| Docker | 24+ с Docker Compose v2 |
| Kubernetes | 1.28+ (для k8s-деплоя) |

### Драйверы GPU

Для обучения и inference на GPU требуются:
- NVIDIA Driver 535+
- CUDA Toolkit 12.1+
- cuDNN 8.9+

---

## 3. Локальная разработка

### 3.1. Клонирование репозитория

```bash
git clone https://github.com/your-org/pulsar-ai.git
cd pulsar-ai
```

### 3.2. Python-окружение

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3.3. Установка зависимостей

Базовая установка (обучение + CLI):

```bash
pip install -e .
```

Установка с дополнительными модулями:

```bash
# Web UI (FastAPI + React)
pip install -e ".[ui]"

# PostgreSQL backend
pip install -e ".[postgres]"

# Redis очередь
pip install -e ".[redis]"

# S3 хранилище артефактов
pip install -e ".[s3]"

# Unsloth (ускоренное обучение)
pip install -e ".[unsloth]"

# vLLM inference
pip install -e ".[vllm]"

# DeepSpeed (распределенное обучение)
pip install -e ".[deepspeed]"

# Evaluation (графики, метрики)
pip install -e ".[eval]"

# Агентный сервер
pip install -e ".[agent-serve]"

# Все модули сразу
pip install -e ".[all]"
```

### 3.4. Конфигурация окружения

```bash
cp .env.example .env
# Отредактируйте .env при необходимости
```

По умолчанию в dev-режиме используются:
- **SQLite** (`./data/pulsar.db`) -- база данных
- **In-process thread pool** -- очередь задач
- **Локальная файловая система** (`./data/artifacts`) -- хранилище артефактов

### 3.5. Запуск backend

```bash
pulsar ui
# или напрямую через uvicorn:
python -m uvicorn pulsar_ai.ui.app:create_app --factory --host 0.0.0.0 --port 8888 --reload
```

### 3.6. Запуск frontend (hot-reload для разработки)

```bash
cd ui
npm install
npm run dev
```

Frontend dev-сервер запускается на `http://localhost:3000` и проксирует API-запросы к backend на порт 8888.

### 3.7. Доступ

- **Backend API**: `http://localhost:8888/api/v1/health`
- **Web UI (dev)**: `http://localhost:3000`
- **Web UI (встроенный)**: `http://localhost:8888`

---

## 4. Docker deployment (development)

### 4.1. Запуск через Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Это запустит:
- **app** -- FastAPI backend со встроенным React frontend (порт 8888)
- **nginx** -- Reverse proxy (порты 80/443)
- **postgres** -- PostgreSQL 16 (порт 5432)
- **redis** -- Redis 7 с AOF-персистентностью (порт 6379)

### 4.2. Переменные окружения (.env)

Основные переменные для dev-режима:

```bash
PULSAR_ENV=development
PULSAR_PORT=8888
PULSAR_CORS_ORIGINS=http://localhost:8888
PULSAR_AUTH_ENABLED=false
PULSAR_LOG_LEVEL=INFO
PULSAR_LOG_FORMAT=console
PULSAR_PG_PASSWORD=changeme
```

### 4.3. Persistent volumes

Docker Compose создает именованные тома:

| Том | Назначение | Путь в контейнере |
|-----|------------|-------------------|
| `pulsar-data` | Данные приложения, артефакты | `/app/data` |
| `pg-data` | Данные PostgreSQL | `/var/lib/postgresql/data` |
| `redis-data` | AOF-персистентность Redis | `/data` |

### 4.4. Доступ

- **Через Nginx (HTTPS)**: `https://localhost` (self-signed сертификат)
- **Напрямую к backend**: `http://localhost:8888`

### 4.5. Остановка

```bash
docker compose down          # остановить контейнеры
docker compose down -v       # остановить + удалить тома (ОСТОРОЖНО: удалит данные)
```

---

## 5. Docker deployment (production)

### 5.1. Подготовка конфигурации

```bash
cp .env.production .env
```

### 5.2. Обязательные переменные окружения

```bash
# ── Среда ──────────────────────────────────────
PULSAR_ENV=production

# ── Аутентификация (ОБЯЗАТЕЛЬНО) ──────────────
PULSAR_AUTH_ENABLED=true
PULSAR_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")

# ── PostgreSQL (ОБЯЗАТЕЛЬНО) ──────────────────
PULSAR_DB_URL=postgresql://pulsar:<PASSWORD>@postgres:5432/pulsar_ai
PULSAR_PG_PASSWORD=<STRONG_PASSWORD>

# ── Redis ─────────────────────────────────────
PULSAR_REDIS_URL=redis://redis:6379/0

# ── CORS (ОБЯЗАТЕЛЬНО) ────────────────────────
PULSAR_CORS_ORIGINS=https://your-domain.com

# ── Логирование ───────────────────────────────
PULSAR_LOG_FORMAT=json
PULSAR_LOG_LEVEL=INFO
```

> **ВНИМАНИЕ**: В production-режиме приложение не запустится без `PULSAR_DB_URL` и `PULSAR_JWT_SECRET`. Это жесткое требование, реализованное в `_validate_production_config()`.

### 5.3. Генерация JWT-секрета

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
# или через openssl:
openssl rand -base64 48
```

Запишите результат в `PULSAR_JWT_SECRET` в файле `.env`.

### 5.4. Настройка PostgreSQL

Пароль PostgreSQL задается через `PULSAR_PG_PASSWORD` и автоматически используется контейнером postgres из docker-compose. Строка подключения в `PULSAR_DB_URL` должна содержать тот же пароль:

```bash
PULSAR_PG_PASSWORD=MyStr0ngP@ssw0rd
PULSAR_DB_URL=postgresql://pulsar:MyStr0ngP@ssw0rd@postgres:5432/pulsar_ai
```

### 5.5. Настройка Redis

```bash
PULSAR_REDIS_URL=redis://redis:6379/0
```

Redis запускается с AOF-персистентностью и лимитом памяти 256 МБ с политикой вытеснения `allkeys-lru`.

### 5.6. SSL/TLS через Nginx

Замените self-signed сертификаты в `nginx/certs/`:

```bash
# Сгенерировать self-signed для тестирования:
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout nginx/certs/selfsigned.key \
  -out nginx/certs/selfsigned.crt \
  -subj "/CN=your-domain.com"
```

Для production используйте сертификаты от Let's Encrypt или вашего CA (см. раздел [10. SSL/TLS](#10-ssltls)).

### 5.7. Resource limits

Docker Compose задает следующие лимиты:

| Сервис | CPU limit | Memory limit | Memory reservation |
|--------|-----------|-------------|-------------------|
| app | 2.0 | 4 ГБ | 512 МБ |
| nginx | 0.5 | 256 МБ | -- |
| postgres | 1.0 | 1 ГБ | 256 МБ |
| redis | 0.5 | 512 МБ | 64 МБ |

### 5.8. Health checks

Все сервисы имеют встроенные health checks:

- **app**: `curl -f http://localhost:8888/api/v1/health` (интервал 30s, таймаут 5s)
- **postgres**: `pg_isready -U pulsar -d pulsar_ai` (интервал 10s)
- **redis**: `redis-cli ping` (интервал 10s)

### 5.9. Запуск

```bash
docker compose up -d
docker compose logs -f  # следить за логами
```

---

## 6. Kubernetes deployment

### 6.1. Пререквизиты

- `kubectl` настроенный на целевой кластер
- Kubernetes 1.28+
- **cert-manager** (для автоматических TLS-сертификатов)
- **ingress-nginx** (ingress controller)
- Доступ к container registry (например, `ghcr.io`)

### 6.2. Сборка и публикация Docker-образа

```bash
docker build -t ghcr.io/your-org/pulsar-ai:latest .
docker push ghcr.io/your-org/pulsar-ai:latest
```

### 6.3. Пошаговый деплой

Все манифесты находятся в директории `k8s/`.

#### Шаг 1: Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

Создает namespace `pulsar-ai` для изоляции всех ресурсов.

#### Шаг 2: Secrets

Сначала сгенерируйте и закодируйте секреты в base64:

```bash
# Генерация JWT-секрета
JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
echo -n "$JWT_SECRET" | base64

# Генерация пароля PostgreSQL
PG_PASSWORD=$(openssl rand -base64 24)
echo -n "$PG_PASSWORD" | base64

# Формирование строки подключения
echo -n "postgresql://pulsar:${PG_PASSWORD}@pulsar-ai-postgresql:5432/pulsar_ai" | base64

# Redis URL
echo -n "redis://pulsar-ai-redis:6379" | base64
```

Подставьте полученные значения в `k8s/secret.yaml` и `k8s/postgresql.yaml` (секция Secret), затем примените:

```bash
kubectl apply -f k8s/secret.yaml
```

> **Рекомендация**: В production используйте secrets manager (HashiCorp Vault, AWS Secrets Manager, sealed-secrets или external-secrets-operator) вместо plain Secret-объектов.

#### Шаг 3: ConfigMap

Отредактируйте `k8s/configmap.yaml` -- замените `PULSAR_CORS_ORIGINS` на ваш домен:

```yaml
data:
  PULSAR_ENV: "production"
  PULSAR_LOG_FORMAT: "json"
  PULSAR_LOG_LEVEL: "info"
  PULSAR_CORS_ORIGINS: "https://pulsar-ai.your-domain.com"
  PULSAR_AUTH_ENABLED: "true"
```

```bash
kubectl apply -f k8s/configmap.yaml
```

#### Шаг 4: PostgreSQL StatefulSet

```bash
kubectl apply -f k8s/postgresql.yaml
```

Создает:
- Secret с паролем PostgreSQL
- Headless Service (`pulsar-ai-postgresql:5432`)
- StatefulSet с PersistentVolumeClaim (10 ГБ)
- Liveness/Readiness/Startup probes

> Для production рассмотрите управляемую БД (AWS RDS, GCP Cloud SQL) или оператор (CloudNativePG, Zalando Postgres Operator).

#### Шаг 5: Redis Deployment

```bash
kubectl apply -f k8s/redis.yaml
```

Создает:
- Service (`pulsar-ai-redis:6379`)
- Deployment с AOF-персистентностью
- PersistentVolumeClaim (5 ГБ)

#### Шаг 6: Application Deployment

Отредактируйте `k8s/deployment.yaml` -- укажите ваш Docker-образ:

```yaml
image: ghcr.io/your-org/pulsar-ai:latest
```

```bash
kubectl apply -f k8s/deployment.yaml
```

Deployment включает:
- 2 реплики с `topologySpreadConstraints` (распределение по нодам)
- Rolling update (maxUnavailable: 1, maxSurge: 1)
- Security context: `runAsNonRoot`, `readOnlyRootFilesystem`, drop all capabilities
- Resource requests/limits: 500m--2 CPU, 1--4 ГБ RAM
- Startup/Liveness/Readiness probes на `/api/v1/health`
- Volumes: PVC для `/app/data`, emptyDir для `/tmp`

#### Шаг 7: Service

```bash
kubectl apply -f k8s/service.yaml
```

Создает ClusterIP-сервис `pulsar-ai:8888`.

#### Шаг 8: Ingress (TLS)

Отредактируйте `k8s/ingress.yaml` -- замените `pulsar-ai.example.com` на ваш домен:

```yaml
spec:
  tls:
    - hosts:
        - pulsar-ai.your-domain.com
      secretName: pulsar-ai-tls
  rules:
    - host: pulsar-ai.your-domain.com
```

```bash
kubectl apply -f k8s/ingress.yaml
```

Ingress использует:
- cert-manager для автоматического получения TLS-сертификатов (ClusterIssuer `letsencrypt-prod`)
- Увеличенные таймауты для длительных запросов обучения (300s)
- Поддержку WebSocket для SSE/streaming

#### Шаг 9: HorizontalPodAutoscaler (HPA)

```bash
kubectl apply -f k8s/hpa.yaml
```

Параметры автомасштабирования:
- Минимум: 2 реплики
- Максимум: 10 реплик
- Scale-up при CPU > 70% или Memory > 80%
- Консервативный scale-down (стабилизация 300s, максимум 1 pod за 60s)

#### Шаг 10: PodDisruptionBudget (PDB)

```bash
kubectl apply -f k8s/pdb.yaml
```

Гарантирует, что минимум 1 pod остается доступным при voluntary disruptions (drain ноды, обновление кластера).

#### Шаг 11: NetworkPolicy

```bash
kubectl apply -f k8s/networkpolicy.yaml
```

Ограничения трафика:
- **Ingress**: только от ingress-nginx namespace на порт 8888
- **Egress**: DNS (53/UDP+TCP), PostgreSQL (5432), Redis (6379), HTTPS (443) для S3/API
- Блокировка доступа к metadata endpoint (169.254.169.254)

#### Применение всех манифестов одной командой

```bash
kubectl apply -f k8s/
```

### 6.4. Проверка статуса

```bash
# Статус всех ресурсов в namespace
kubectl get all -n pulsar-ai

# Проверка подов
kubectl get pods -n pulsar-ai

# Логи приложения
kubectl logs -n pulsar-ai -l app.kubernetes.io/component=api -f

# Health check
kubectl exec -n pulsar-ai deploy/pulsar-ai -- curl -s http://localhost:8888/api/v1/health

# Проверка Ingress
kubectl get ingress -n pulsar-ai
```

### 6.5. Масштабирование

```bash
# Ручное масштабирование
kubectl scale deployment pulsar-ai -n pulsar-ai --replicas=5

# Проверка HPA
kubectl get hpa -n pulsar-ai
```

---

## 7. Настройка базы данных

### 7.1. SQLite (только для разработки)

По умолчанию Pulsar AI использует SQLite:
- Файл БД: `./data/pulsar.db`
- Режим WAL для конкурентного чтения
- Автоматическое создание схемы при первом запуске

Конфигурация не требуется. Используется, если `PULSAR_DB_URL` не задан.

> **ВНИМАНИЕ**: SQLite не подходит для production -- нет конкурентной записи, нет репликации, ограниченная масштабируемость.

### 7.2. PostgreSQL

#### Установка (Ubuntu)

```bash
sudo apt update
sudo apt install -y postgresql-16 postgresql-client-16
sudo systemctl enable postgresql
sudo systemctl start postgresql
```

#### Создание базы данных и пользователя

```bash
sudo -u postgres psql <<EOF
CREATE USER pulsar WITH PASSWORD 'your_strong_password';
CREATE DATABASE pulsar_ai OWNER pulsar;
GRANT ALL PRIVILEGES ON DATABASE pulsar_ai TO pulsar;
EOF
```

#### Формат строки подключения

```
postgresql://<user>:<password>@<host>:<port>/<database>
```

Примеры:

```bash
# Локальная БД
PULSAR_DB_URL=postgresql://pulsar:secret@localhost:5432/pulsar_ai

# Docker Compose (внутренняя сеть)
PULSAR_DB_URL=postgresql://pulsar:secret@postgres:5432/pulsar_ai

# Kubernetes
PULSAR_DB_URL=postgresql://pulsar:secret@pulsar-ai-postgresql:5432/pulsar_ai

# AWS RDS
PULSAR_DB_URL=postgresql://pulsar:secret@mydb.abc123.us-east-1.rds.amazonaws.com:5432/pulsar_ai
```

#### Миграции

Схема автоматически создается при первом запуске приложения. PostgreSQL backend (`pulsar_ai.storage.postgres.PostgresDatabase`) автоматически конвертирует SQLite DDL в PostgreSQL-совместимый синтаксис и применяет его.

Версия схемы хранится в таблице `_schema_meta` и проверяется при каждом запуске.

#### Рекомендуемые настройки PostgreSQL (production)

```ini
# postgresql.conf
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 128MB
wal_level = replica
max_wal_size = 1GB
```

---

## 8. Настройка аутентификации

### 8.1. Включение JWT-аутентификации

```bash
PULSAR_AUTH_ENABLED=true
PULSAR_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
```

При включенной аутентификации:
- Все API-запросы требуют JWT-токен в заголовке `Authorization: Bearer <token>`
- Поддерживается fallback на API-ключи
- При первом запуске автоматически генерируется дефолтный API-ключ (выводится в лог)

### 8.2. Генерация JWT-секрета

```bash
# Python
python -c "import secrets; print(secrets.token_urlsafe(48))"

# OpenSSL
openssl rand -base64 48
```

> **ВНИМАНИЕ**: Если `PULSAR_JWT_SECRET` не задан при включенной аутентификации, используется случайный секрет. Все токены будут недействительны после перезапуска.

### 8.3. Создание первого пользователя (admin)

После запуска с `PULSAR_AUTH_ENABLED=true` зарегистрируйте первого пользователя через API:

```bash
curl -X POST http://localhost:8888/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@your-domain.com",
    "password": "YourStr0ngP@ssword!",
    "name": "Admin"
  }'
```

Затем назначьте роль `admin` напрямую через БД:

```sql
-- PostgreSQL
UPDATE users SET role = 'admin' WHERE email = 'admin@your-domain.com';

-- SQLite
sqlite3 data/pulsar.db "UPDATE users SET role = 'admin' WHERE email = 'admin@your-domain.com';"
```

После этого admin может управлять другими пользователями через `/api/v1/admin/users`.

### 8.4. Защита от brute-force

Встроенная защита:
- **5** неудачных попыток на email -- блокировка на 15 минут
- **20** неудачных попыток с одного IP -- блокировка на 15 минут
- Rate limiting: 60 запросов в минуту на IP

### 8.5. Настройка OIDC/SSO

Pulsar AI поддерживает любой OpenID Connect провайдер. Конфигурация через переменные окружения:

```bash
PULSAR_OIDC_PROVIDER_URL=<URL провайдера>
PULSAR_OIDC_CLIENT_ID=<Client ID>
PULSAR_OIDC_CLIENT_SECRET=<Client Secret>
PULSAR_OIDC_REDIRECT_URI=https://your-domain.com/api/v1/auth/oidc/callback
```

#### Azure AD

```bash
PULSAR_OIDC_PROVIDER_URL=https://login.microsoftonline.com/<TENANT_ID>/v2.0
PULSAR_OIDC_CLIENT_ID=<Application (client) ID>
PULSAR_OIDC_CLIENT_SECRET=<Client secret value>
PULSAR_OIDC_REDIRECT_URI=https://pulsar-ai.your-domain.com/api/v1/auth/oidc/callback
```

В Azure Portal:
1. Зарегистрируйте приложение в Azure AD
2. Добавьте redirect URI: `https://pulsar-ai.your-domain.com/api/v1/auth/oidc/callback`
3. Создайте client secret
4. Настройте API permissions: `openid`, `profile`, `email`

#### Google

```bash
PULSAR_OIDC_PROVIDER_URL=https://accounts.google.com
PULSAR_OIDC_CLIENT_ID=<Client ID>.apps.googleusercontent.com
PULSAR_OIDC_CLIENT_SECRET=<Client Secret>
PULSAR_OIDC_REDIRECT_URI=https://pulsar-ai.your-domain.com/api/v1/auth/oidc/callback
```

В Google Cloud Console:
1. Создайте OAuth 2.0 credentials
2. Добавьте authorized redirect URI
3. Включите OpenID Connect scopes

#### Okta

```bash
PULSAR_OIDC_PROVIDER_URL=https://<your-org>.okta.com
PULSAR_OIDC_CLIENT_ID=<Client ID>
PULSAR_OIDC_CLIENT_SECRET=<Client Secret>
PULSAR_OIDC_REDIRECT_URI=https://pulsar-ai.your-domain.com/api/v1/auth/oidc/callback
```

#### Keycloak

```bash
PULSAR_OIDC_PROVIDER_URL=https://keycloak.your-domain.com/realms/<realm>
PULSAR_OIDC_CLIENT_ID=pulsar-ai
PULSAR_OIDC_CLIENT_SECRET=<Client Secret>
PULSAR_OIDC_REDIRECT_URI=https://pulsar-ai.your-domain.com/api/v1/auth/oidc/callback
```

Провайдер автоматически определяется по URL (Microsoft, Google, Okta, Auth0) для отображения названия кнопки в UI.

### 8.6. Настройка MFA/TOTP

Pulsar AI поддерживает TOTP-based двухфакторную аутентификацию (RFC 6238):

1. Пользователь включает MFA через `POST /api/v1/auth/mfa/setup`
2. Получает QR-код (otpauth:// URI) для сканирования в приложении-аутентификаторе
3. Подтверждает активацию кодом из приложения
4. При следующем логине после ввода пароля потребуется 6-значный TOTP-код

Параметры:
- 6-значные коды
- Период: 30 секунд
- Допуск: +/- 1 временной шаг
- Резервные коды генерируются при активации

Администратор может отключить MFA пользователю через `DELETE /api/v1/admin/users/{user_id}/mfa`.

---

## 9. Настройка хранилища

### 9.1. Локальная файловая система (по умолчанию)

По умолчанию артефакты модели сохраняются в `./data/artifacts`. Конфигурация не требуется.

### 9.2. S3 / MinIO

```bash
# AWS S3
PULSAR_S3_BUCKET=pulsar-artifacts
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# MinIO (self-hosted)
PULSAR_S3_BUCKET=pulsar-artifacts
PULSAR_S3_ENDPOINT=http://minio:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

Требуется установка:

```bash
pip install "pulsar-ai[s3]"
```

### 9.3. Миграция с локального хранилища на S3

```bash
# Установка AWS CLI
pip install awscli

# Загрузка существующих артефактов
aws s3 sync ./data/artifacts s3://pulsar-artifacts/ \
  --endpoint-url http://minio:9000  # только для MinIO

# Обновление конфигурации
export PULSAR_S3_BUCKET=pulsar-artifacts
# перезапустите приложение
```

---

## 10. SSL/TLS

### 10.1. Self-signed сертификаты (разработка)

```bash
mkdir -p nginx/certs

openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout nginx/certs/selfsigned.key \
  -out nginx/certs/selfsigned.crt \
  -subj "/CN=localhost"
```

### 10.2. Let's Encrypt (production с Docker)

Используйте certbot:

```bash
# Установка certbot
apt install -y certbot

# Получение сертификата
certbot certonly --standalone -d pulsar-ai.your-domain.com

# Копирование сертификатов
cp /etc/letsencrypt/live/pulsar-ai.your-domain.com/fullchain.pem nginx/certs/selfsigned.crt
cp /etc/letsencrypt/live/pulsar-ai.your-domain.com/privkey.pem nginx/certs/selfsigned.key

# Автоматическое обновление (cron)
echo "0 3 * * * certbot renew --quiet && docker compose restart nginx" | crontab -
```

### 10.3. Конфигурация Nginx

Nginx (`nginx/nginx.conf`) уже настроен для TLS:
- TLS 1.2 и 1.3
- Набор шифров HIGH (без aNULL и MD5)
- HSTS с `max-age=31536000` и `includeSubDomains`
- Автоматический редирект HTTP -> HTTPS
- Максимальный размер тела запроса: 500 МБ (для загрузки датасетов)
- Gzip-сжатие для текстовых форматов
- Увеличенные таймауты для SSE/WebSocket (3600s)

### 10.4. cert-manager для Kubernetes

cert-manager автоматически управляет сертификатами в кластере. Создайте ClusterIssuer:

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@your-domain.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
      - http01:
          ingress:
            class: nginx
```

```bash
kubectl apply -f cluster-issuer.yaml
```

Ingress из `k8s/ingress.yaml` уже содержит аннотацию `cert-manager.io/cluster-issuer: letsencrypt-prod`, поэтому сертификат будет получен автоматически.

---

## 11. Мониторинг и логирование

### 11.1. Structured logging

Pulsar AI использует structlog для структурированного логирования:

| Формат | Переменная | Использование |
|--------|------------|---------------|
| `console` | `PULSAR_LOG_FORMAT=console` | Разработка (цветной вывод) |
| `json` | `PULSAR_LOG_FORMAT=json` | Production (парсинг, агрегация) |

Уровни: `DEBUG`, `INFO`, `WARNING`, `ERROR` (через `PULSAR_LOG_LEVEL`).

### 11.2. Prometheus метрики

Endpoint: `GET /metrics` (стандартный Prometheus exposition format).

Доступные метрики:
- `pulsar_requests_total` -- общее количество запросов
- `pulsar_training_jobs_total` -- количество запусков обучения
- `pulsar_errors_total` -- количество ошибок
- `pulsar_cpu_percent` -- загрузка CPU
- `pulsar_memory_percent` -- использование RAM
- `pulsar_memory_used_bytes` -- использование RAM (байты)
- `pulsar_gpu_*` -- метрики GPU (при наличии NVIDIA GPU)

### 11.3. Рекомендуемый стек мониторинга

**Grafana + Prometheus**:

```yaml
# prometheus.yml (добавьте в scrape_configs)
scrape_configs:
  - job_name: 'pulsar-ai'
    scrape_interval: 15s
    static_configs:
      - targets: ['pulsar-ai:8888']
    metrics_path: /metrics
```

### 11.4. Агрегация логов

Рекомендуемые решения:
- **ELK Stack** (Elasticsearch + Logstash + Kibana) -- для JSON-логов
- **Grafana Loki + Promtail** -- легковесная альтернатива

Для Kubernetes используйте Fluentd или Promtail как DaemonSet.

### 11.5. Health endpoint

```bash
curl http://localhost:8888/api/v1/health
# {"status": "ok"}
```

В Kubernetes health endpoint используется для:
- **startupProbe**: `/api/v1/health` (до 5 минут на старт)
- **livenessProbe**: `/api/v1/health` (каждые 30s)
- **readinessProbe**: `/api/v1/health` (каждые 10s)

---

## 12. Бэкап и восстановление

### 12.1. SQLite backup

Используйте встроенный скрипт:

```bash
./scripts/backup_db.sh [db_path] [backup_dir]

# По умолчанию:
./scripts/backup_db.sh
# db_path:    data/pulsar.db
# backup_dir: backups/
```

Скрипт:
- Выполняет hot backup через SQLite `.backup` (безопасно для WAL-режима)
- Сжимает бэкап в gzip
- Хранит последние 7 дней (автоочистка)

### 12.2. PostgreSQL pg_dump

```bash
# Бэкап
pg_dump -h localhost -U pulsar -d pulsar_ai -Fc -f backup_$(date +%Y%m%d_%H%M%S).dump

# Для Docker Compose
docker compose exec postgres pg_dump -U pulsar -d pulsar_ai -Fc > backup_$(date +%Y%m%d_%H%M%S).dump

# Для Kubernetes
kubectl exec -n pulsar-ai pulsar-ai-postgresql-0 -- \
  pg_dump -U pulsar -d pulsar_ai -Fc > backup_$(date +%Y%m%d_%H%M%S).dump
```

### 12.3. Автоматические бэкапы (cron)

```bash
# Ежедневный бэкап SQLite в 2:00
0 2 * * * /path/to/pulsar-ai/scripts/backup_db.sh

# Ежедневный бэкап PostgreSQL в 2:00
0 2 * * * pg_dump -h localhost -U pulsar -d pulsar_ai -Fc -f /backups/pulsar_$(date +\%Y\%m\%d).dump && find /backups -name "pulsar_*.dump" -mtime +7 -delete
```

### 12.4. Восстановление из бэкапа

**SQLite:**

```bash
# Остановите приложение
cp backups/pulsar_YYYYMMDD_HHMMSS.db.gz data/
cd data
gunzip pulsar_YYYYMMDD_HHMMSS.db.gz
mv pulsar.db pulsar.db.old
mv pulsar_YYYYMMDD_HHMMSS.db pulsar.db
# Запустите приложение
```

**PostgreSQL:**

```bash
# Остановите приложение
pg_restore -h localhost -U pulsar -d pulsar_ai --clean --if-exists backup.dump
# Запустите приложение
```

### 12.5. RPO/RTO рекомендации

| Параметр | SQLite | PostgreSQL |
|----------|--------|------------|
| RPO (Recovery Point Objective) | 24 часа (daily backup) | 1 час (WAL archiving) |
| RTO (Recovery Time Objective) | 5 минут | 15 минут |
| Рекомендация | Только dev | Production |

Для минимального RPO с PostgreSQL настройте WAL archiving и continuous backup (pgBackRest, WAL-G).

---

## 13. Безопасность

### 13.1. Security headers

FastAPI автоматически добавляет заголовки через `SecurityHeadersMiddleware`:

| Заголовок | Значение |
|-----------|----------|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `X-XSS-Protection` | `1; mode=block` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=()` |
| `Content-Security-Policy` | `default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; ...` |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` (только HTTPS) |

Nginx дублирует ключевые заголовки для дополнительной защиты.

### 13.2. Rate limiting

- **Глобальный лимит**: 60 запросов в минуту на IP (через slowapi)
- **Request size limit**: 100 МБ (предотвращение DoS через большие upload-ы)
- **Request timeout**: 5 минут (SSE/streaming endpoints исключены)

### 13.3. Защита от brute-force

- Блокировка email после 5 неудачных попыток (15 минут)
- Блокировка IP после 20 неудачных попыток (15 минут)
- Валидация сложности пароля при регистрации

### 13.4. CORS

В production CORS строго ограничен:

```bash
# Только указанные домены
PULSAR_CORS_ORIGINS=https://pulsar-ai.your-domain.com

# Несколько доменов (через запятую)
PULSAR_CORS_ORIGINS=https://pulsar-ai.your-domain.com,https://admin.your-domain.com
```

> Если `PULSAR_CORS_ORIGINS` не задан в production, все cross-origin запросы будут отклонены.

### 13.5. Network policies (Kubernetes)

NetworkPolicy (`k8s/networkpolicy.yaml`) ограничивает:
- **Входящий трафик**: только от ingress-nginx на порт 8888
- **Исходящий трафик**: только DNS, PostgreSQL, Redis, HTTPS (443)
- **Блокировка**: доступ к cloud metadata endpoint (169.254.169.254)

### 13.6. Docker security

- **Non-root user**: контейнер запускается от пользователя `pulsar` (UID 1000)
- **Read-only filesystem**: в Kubernetes `readOnlyRootFilesystem: true`
- **Capability drop**: все Linux capabilities отключены (`drop: ALL`)
- **Запрет privilege escalation**: `allowPrivilegeEscalation: false`

---

## 14. Обновление

### 14.1. Docker

```bash
# Получение нового образа
docker compose pull
# или пересборка из исходников
docker compose build

# Перезапуск с zero-downtime (новый контейнер стартует до остановки старого)
docker compose up -d
```

### 14.2. Kubernetes (rolling update)

```bash
# Обновление образа
kubectl set image deployment/pulsar-ai -n pulsar-ai \
  pulsar-ai=ghcr.io/your-org/pulsar-ai:v1.2.0

# Или обновление манифеста и применение
kubectl apply -f k8s/deployment.yaml

# Отслеживание прогресса
kubectl rollout status deployment/pulsar-ai -n pulsar-ai
```

Rolling update гарантирует: `maxUnavailable: 1`, `maxSurge: 1`.

### 14.3. Миграции базы данных

Миграции применяются автоматически при старте приложения. При обновлении:

1. Новый pod стартует и выполняет `_bootstrap()`, которая применяет DDL идемпотентно (`CREATE TABLE IF NOT EXISTS`)
2. Версия схемы обновляется в `_schema_meta`
3. Старый pod продолжает обслуживать трафик до готовности нового

### 14.4. Rollback

**Docker:**

```bash
docker compose down
docker compose up -d --force-recreate
```

**Kubernetes:**

```bash
# Откат к предыдущей ревизии
kubectl rollout undo deployment/pulsar-ai -n pulsar-ai

# Откат к конкретной ревизии
kubectl rollout history deployment/pulsar-ai -n pulsar-ai
kubectl rollout undo deployment/pulsar-ai -n pulsar-ai --to-revision=3
```

---

## 15. Устранение неполадок

### 15.1. Приложение не запускается в production

**Ошибка**: `RuntimeError: Production mode requires the following environment variables: PULSAR_DB_URL, PULSAR_JWT_SECRET`

**Решение**: Установите обе переменные:

```bash
export PULSAR_DB_URL=postgresql://pulsar:secret@localhost:5432/pulsar_ai
export PULSAR_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
```

### 15.2. Проверка логов

```bash
# Docker
docker compose logs app -f --tail=100

# Kubernetes
kubectl logs -n pulsar-ai -l app.kubernetes.io/component=api -f --tail=100
```

### 15.3. Проверка health endpoint

```bash
curl -v http://localhost:8888/api/v1/health
# Ожидаемый ответ: {"status": "ok"}
```

### 15.4. Подключение к базе данных

```bash
# PostgreSQL (Docker)
docker compose exec postgres psql -U pulsar -d pulsar_ai

# PostgreSQL (Kubernetes)
kubectl exec -it -n pulsar-ai pulsar-ai-postgresql-0 -- psql -U pulsar -d pulsar_ai

# Проверка таблиц
\dt
SELECT * FROM _schema_meta;
```

### 15.5. Redis не отвечает

```bash
# Docker
docker compose exec redis redis-cli ping
# Ожидаемый ответ: PONG

# Kubernetes
kubectl exec -n pulsar-ai deploy/pulsar-ai-redis -- redis-cli ping
```

### 15.6. GPU не обнаружен

```bash
# Проверка NVIDIA драйвера
nvidia-smi

# Проверка CUDA в Python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Для Docker: убедитесь, что используете nvidia runtime
docker run --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### 15.7. Nginx возвращает 502 Bad Gateway

Причина: backend не готов или не доступен.

```bash
# Проверьте статус backend
docker compose ps app
docker compose logs app --tail=50

# Проверьте, что backend слушает на порту 8888
docker compose exec app curl -s http://localhost:8888/api/v1/health
```

### 15.8. CORS ошибки в браузере

Проверьте `PULSAR_CORS_ORIGINS`:

```bash
# Должен совпадать с доменом, с которого открывается UI
PULSAR_CORS_ORIGINS=https://pulsar-ai.your-domain.com
```

### 15.9. JWT-токены невалидны после перезапуска

Убедитесь, что `PULSAR_JWT_SECRET` задан и сохранен. Без явного секрета используется случайный, который теряется при перезапуске.

---

## 16. Переменные окружения

Полная таблица всех поддерживаемых переменных:

| Переменная | Описание | Значение по умолчанию | Обязательно (prod) |
|------------|----------|----------------------|-------------------|
| `PULSAR_ENV` | Режим работы: `development` / `production` | `development` | Да |
| `PULSAR_PORT` | Порт backend-сервера | `8888` | Нет |
| `PULSAR_CORS_ORIGINS` | Разрешенные CORS-домены (через запятую) | `http://localhost:3000,http://localhost:8888` | Да |
| `PULSAR_STAND_MODE` | Режим стенда: `dev` / `demo` (read-only) | `dev` | Нет |
| `PULSAR_AUTH_ENABLED` | Включить JWT-аутентификацию | `false` | Да (`true`) |
| `PULSAR_JWT_SECRET` | Секрет для подписи JWT-токенов | Случайный (при каждом запуске) | Да |
| `PULSAR_DB_URL` | Строка подключения к БД (`postgresql://...`) | SQLite (`./data/pulsar.db`) | Да |
| `PULSAR_PG_PASSWORD` | Пароль PostgreSQL (для docker-compose) | `changeme` | Да |
| `PULSAR_REDIS_URL` | URL Redis (`redis://host:port/db`) | Не задан (in-process pool) | Нет |
| `PULSAR_S3_BUCKET` | Имя S3-бакета для артефактов | Не задан (локальная FS) | Нет |
| `PULSAR_S3_ENDPOINT` | Custom S3 endpoint (MinIO) | Не задан (AWS S3) | Нет |
| `AWS_ACCESS_KEY_ID` | AWS/MinIO access key | Не задан | При использовании S3 |
| `AWS_SECRET_ACCESS_KEY` | AWS/MinIO secret key | Не задан | При использовании S3 |
| `PULSAR_LOG_LEVEL` | Уровень логирования: `DEBUG`/`INFO`/`WARNING`/`ERROR` | `INFO` | Нет |
| `PULSAR_LOG_FORMAT` | Формат логов: `console` / `json` | `console` | Нет (`json`) |
| `PULSAR_ENV_FILE` | Путь к файлу переменных окружения | `.env` | Нет |
| `OPENAI_API_KEY` | API-ключ OpenAI (для AI-ассистента) | Не задан | Нет |
| `PULSAR_OIDC_PROVIDER_URL` | URL OIDC-провайдера | Не задан | Нет |
| `PULSAR_OIDC_CLIENT_ID` | OIDC Client ID | Не задан | При использовании SSO |
| `PULSAR_OIDC_CLIENT_SECRET` | OIDC Client Secret | Не задан | При использовании SSO |
| `PULSAR_OIDC_REDIRECT_URI` | OIDC Redirect URI | Не задан | При использовании SSO |

---

> **Документ обновлен**: 2026-03-23
