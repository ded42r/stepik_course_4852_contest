# путь до файлов с данными
DATA_DIR = "../data"
# каталог со сгенерированными датасетами для обучения
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
# каталог хранения отчетов по прогнозам
REPORTS_DIR = "../reports"
# каталог хранения бинарников модели
BIN_MODELS_DIR = "../bin_models"

# колво дней по которым доступны данные для прогноза
DATA_PERIOD_DAYS = 2
ACTION_CATEGORIES = ('discovered', 'viewed', 'started_attempt', 'passed')
SUBMISSION_STATUSES = ('wrong', 'correct')
