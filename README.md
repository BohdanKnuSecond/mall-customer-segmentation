# Customer Segmentation — Mall Customers (K-means)

Лабораторна робота з курсу **Fundamentals of Data Mining 2026** (Basic Part).
Автор: Богдан Герасименко, ТТП-41. Звіт: [`report/`](report/).

Сегментація 200 клієнтів торгового центру за річним доходом і Spending Score методом K-means (k = 5),
з перевіркою якості, стабільності та порівнянням з ієрархічною кластеризацією і DBSCAN.

## Структура

| Шлях | Вміст |
|---|---|
| `mall_segmentation.ipynb` | основний ноутбук з усіма обчисленнями та графіками (виконаний) |
| `data/Mall_Customers.csv` | датасет Mall Customer Segmentation Data (Kaggle) |
| `figures/` | усі графіки, які зберігає ноутбук |
| `results/summary.json` | усі числові результати, з яких формується звіт |
| `results/customers_with_segments.csv` | клієнти з призначеним сегментом |
| `results/segment_profile.csv` | профілі сегментів |
| `report/Звіт_Customer_Segmentation_Герасименко.pdf` | звіт |
| `tools/build_notebook.py` | генерує і виконує ноутбук |
| `tools/build_report.py` | збирає звіт (HTML → PDF через Microsoft Edge) |

## Основні результати

| Сегмент | Клієнтів | Дохід, k$ | Spending |
|---|---|---|---|
| Преміум-покупці | 39 | 86,5 | 82,1 |
| Стримані заможні | 35 | 88,2 | 17,1 |
| Масовий сегмент | 81 | 55,3 | 49,5 |
| Активні з малим доходом | 22 | 25,7 | 79,4 |
| Ощадливі | 23 | 26,3 | 20,9 |

Silhouette 0,555 · Davies–Bouldin 0,572 · Calinski–Harabasz 248,6 · bootstrap ARI 0,971 ± 0,024.

## Запуск

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tools\build_notebook.py
.venv\Scripts\python tools\build_report.py
```

Ноутбук можна також відкрити й виконати в Jupyter / VS Code (робоча директорія — корінь проєкту).
