# Titanic Survival Prediction

# Задача
  
Классическая задача бинарной классификации: по данным о пассажирах Титаника (пол, возраст, класс каюты, стоимость билета и др.) предсказать, выжил ли пассажир.

  

- **Датасет:** 891 тренировочный пример, 418 тестовых

- **Метрика:** Accuracy

- **Источник:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

  

##  Лучшие результаты

  

| Модель | CV Accuracy | CV Std | Public LB | Дата |

|--------|-------------|--------|-----------|------|

| **DNN (PyTorch)** | **0.8435** | - | - | 25.09.2026 |

| **LightGBM** | **0.8406** | 0.0174 | *- | 25.09.2026 |

| **CatBoost** | 0.8372 | 0.0107 | - | 25.09.2026 |

| **XGBoost** | 0.8350 | 0.0078 |- | 25.09.2026 |

| **Random Forest** | 0.8350 | 0.0110 | -| 25.09.2026 |

| **Lasso** | 0.8350 | 0.0092 | 0.77511 | 24.09.2026 |

| **Logistic Regression** | 0.8317 | 0.0105 | 0.77751 | 24.09.2026 |

| **Ridge** | 0.8193 | 0.0130 | **0.78229** | 24.09.2026 |

| Decision Tree | 0.8182 | 0.0204 | 0.76076 | 24.09.2026 |

| KNN | 0.7980 | 0.0216 | 0.70574 | 24.09.2026 |

| Gender Baseline | - | - | 0.76555 | 23.09.2026 |

  

> CV = 5-Fold Stratified Cross-Validation, LB = Kaggle Public Leaderboard.  


  

##  Как  запустить

1)  В bash:
  
- pip install -r requirements.txt


2) Скачать train.csv и test.csv с [Kaggle](https://www.kaggle.com/competitions/titanic/data)

3) Открой `config.py` и выбери модель:
`'general': {`
`'active': 'lightgbm',      # xgb, catboost, dnn, ridge, ensemble_avg и др.`
`'experiment_id': 23,       # уникальный номер эксперимента`
 `'seed': 42,``},`
4) Запустить main.py
5) Забрать результаты в папке results



# Структура

MLDLPr1-Titanic/
├── main.py                  # Точка входа: оркестрация всего пайплайна
├── config.py                # Единый центр настроек (модели, гиперпараметры, пути)
├── requirements.txt         # Зависимости проекта
├── .gitignore               # Исключения для Git
├── notebooks/
│   └── EDA Titanic.ipynb    # Исследовательский анализ данных
── data/                    # Сырые данные (не в Git)
├── results/                 # Артефакты экспериментов (не в Git)
└── src/                     # Исходный код
    ├── __init__.py
    ├── data.py              # Загрузка CSV и подготовка признаков
    ├── preprocessing.py     # Трансформации: титулы, биннинг, OHE
    ├── models.py            # Фабрика моделей (Sklearn, Boosting, PyTorch DNN)
    └── utils.py             # Статистики, сохранение submission и метрик
