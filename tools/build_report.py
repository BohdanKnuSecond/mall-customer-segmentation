import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
HTML_PATH = REPORT_DIR / "report.html"
PDF_PATH = REPORT_DIR / "Звіт_Customer_Segmentation_Герасименко.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
REPO_URL = "https://github.com/BohdanKnuSecond/mall-customer-segmentation"

S = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))


def num(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def pct(x, nd=1):
    return num(100 * x, nd) + "%"


def table(headers, rows, cls=""):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def fig(name, caption, width=100):
    return (f'<figure><img src="../figures/{name}.png" style="width:{width}%">'
            f"<figcaption>{caption}</figcaption></figure>")


FIG_N = iter(range(1, 100))
TAB_N = iter(range(1, 100))


def F(name, text, width=100):
    return fig(name, f"Рис. {next(FIG_N)}. {text}", width)


def T(text):
    return f'<p class="tcap">Таблиця {next(TAB_N)}. {text}</p>'


desc = S["describe"]
out = {o["feature"]: o for o in S["outliers"]}
gt = {g["feature"]: g for g in S["gender_tests"]}
hop = S["hopkins"]
own = S["own_kmeans"]
votes = S["votes"]
scan2 = S["scans"]["F2"]
scan3 = S["scans"]["F3"]
init = S["init"]
prof = S["profile"]
q = S["quality"]
st = S["stability"]
alt = S["alternatives"]
f3 = S["f3"]
pca = S["pca"]
ag = S["age_groups"]
pear = S["corr_pearson"]
NAMES = [p["segment"] for p in prof]
P = {p["cluster"]: p for p in prof}

versions = {"python": sys.version.split()[0], "pandas": version("pandas"), "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"), "scipy": version("scipy"),
            "matplotlib": version("matplotlib"), "seaborn": version("seaborn")}

feat_ua = {"age": "Вік", "income": "Дохід, k$", "spending": "Spending Score"}

repo_line = (f'<p><b>Репозиторій:</b> <a href="{REPO_URL}">{REPO_URL}</a></p>' if REPO_URL
             else "")

f3_prof = {r["cluster_f3"]: r for r in f3["profile"]}
kruskal_str = f"{S['age_kruskal_p']:.1e}".replace(".", ",").replace("e-0", "·10<sup>−") + "</sup>"
mass_split = f3["cross"]["Масовий сегмент"]
mass_old = max(mass_split, key=lambda k: f3_prof[int(k)]["age"] if mass_split[k] > 10 else -1)
mass_young = min(mass_split, key=lambda k: f3_prof[int(k)]["age"] if mass_split[k] > 10 else 999)

body = f"""
<header>
<div class="kicker">Fundamentals of Data Mining 2026 · Basic Part (Laboratory Work)</div>
<h1>Звіт з лабораторної роботи<br><span>Customer Segmentation: сегментація відвідувачів торгового центру методом K-means</span></h1>
<p class="meta"><b>Виконав:</b> Богдан Герасименко, група ТТП-41<br>
<b>Код:</b> Jupyter-ноутбук <code>mall_segmentation.ipynb</code>, дані — <code>data/Mall_Customers.csv</code>,
усі числа звіту зберігаються ноутбуком у <code>results/summary.json</code>.</p>
{repo_line}
</header>

<h2>1. Постановка задачі</h2>
<p>Мета роботи — розбити клієнтів торгового центру на однорідні групи за доходом і купівельною активністю,
описати кожну групу та запропонувати для неї маркетингову стратегію. Основний алгоритм — K-means,
результат порівнюється з ієрархічною кластеризацією та DBSCAN.</p>
<p>Порядок роботи: перевірка якості даних і викидів → розвідувальний аналіз (у тому числі статистичні тести
і оцінка кластерної тенденції) → власна реалізація K-means та звірка з бібліотечною → вибір набору ознак і
кількості кластерів за чотирма критеріями → дослідження ініціалізації → фінальна модель і профілі сегментів →
оцінка якості та стабільності → візуалізація і бізнес-висновки.</p>
<p><b>Інструменти:</b> Python {versions['python']}, pandas {versions['pandas']}, NumPy {versions['numpy']},
scikit-learn {versions['scikit-learn']}, SciPy {versions['scipy']}, matplotlib {versions['matplotlib']},
seaborn {versions['seaborn']}. Для відтворюваності всюди зафіксовано <code>random_state = 42</code>.</p>

<h2>2. Дані та їх якість</h2>
<p>Набір <b>Mall Customer Segmentation Data</b> (Kaggle) містить 200 записів і 5 стовпців:
<code>CustomerID</code>, <code>Gender</code>, <code>Age</code>, <code>Annual Income (k$)</code>,
<code>Spending Score (1-100)</code>. Spending Score — внутрішній індекс торгового центру, що відображає
поведінку і обсяг покупок клієнта. Для зручності стовпці перейменовано на <code>id, gender, age, income, spending</code>.</p>
{T("Описова статистика числових ознак")}
{table(["Ознака", "Середнє", "σ", "Мін", "Медіана", "Макс", "Асиметрія", "Ексцес"],
       [[feat_ua[c], num(desc[c]['mean']), num(desc[c]['std']), num(desc[c]['min'], 0), num(desc[c]['50%'], 1),
         num(desc[c]['max'], 0), num(desc[c]['skew']), num(desc[c]['kurtosis'])] for c in ["age", "income", "spending"]])}
<p><b>Перевірки якості.</b> Пропусків — {S['checks']['missing']}, повних дублікатів — {S['checks']['duplicate_rows']},
дублікатів без урахування ID — {S['checks']['duplicates_without_id']}. Перевірено допустимі діапазони
(стать ∈ {{Male, Female}}, вік 18–100, дохід &gt; 0, Spending 1–100): некоректних значень немає.
Жінок — {S['gender_counts']['Female']}, чоловіків — {S['gender_counts']['Male']}.</p>
<p><b>Викиди.</b> Використано два правила. За IQR (межі Q1 − 1,5·IQR, Q3 + 1,5·IQR) знайдено
{out['income']['iqr_outliers']} значення доходу вище {num(out['income']['high'])} k$ — це два клієнти з доходом 137 k$.
За правилом |z| &gt; 3 викидів немає взагалі (максимум |z| для доходу — {num(out['income']['z_max'])}).
Значення реалістичні (високий дохід, а не помилка введення), розподіл доходу лише помірно скошений
(асиметрія {num(desc['income']['skew'])}), тому записи залишено.</p>
{F("01_outliers", "Діаграми розмаху з усіма точками; пунктир — верхня межа IQR для доходу")}

<p><b>Ознаки та масштабування.</b> <code>CustomerID</code> не несе інформації. Стать закодовано як
<code>is_female</code> і використано лише для опису сегментів: це бінарна ознака, евклідова відстань до неї
некоректна, а розділ 3 показує, що стать не пов'язана з іншими ознаками. Розглянуто два набори:
<b>F2</b> = {{дохід, Spending}} та <b>F3</b> = {{вік, дохід, Spending}}. Ознаки стандартизовано
(<code>StandardScaler</code>, z = (x − μ)/σ), бо K-means оперує відстанями і ознака з більшим розкидом
домінувала б. Перевірка: без стандартизації розбиття F2 не змінюється (ARI = {num(S['ari_unscaled_vs_scaled']['F2'])} —
дохід і Spending мають близькі масштаби), а для F3 змінюється (ARI = {num(S['ari_unscaled_vs_scaled']['F3'], 3)}).</p>

<h2>3. Розвідувальний аналіз</h2>
<h3>3.1 Чоловіки і жінки</h3>
<p>Розподіли не є нормальними, тому для порівняння груп застосовано непараметричний U-тест Манна–Уітні
(H₀: розподіли ознаки у чоловіків і жінок однакові).</p>
{T("Порівняння ознак за статтю")}
{table(["Ознака", "Жінки: середнє / медіана", "Чоловіки: середнє / медіана", "U", "p-value"],
       [[feat_ua[c], f"{num(gt[c]['female_mean'], 1)} / {num(gt[c]['female_median'], 1)}",
         f"{num(gt[c]['male_mean'], 1)} / {num(gt[c]['male_median'], 1)}", num(gt[c]['U'], 1), num(gt[c]['p_value'], 3)]
        for c in ["age", "income", "spending"]])}
<p>Усі p-value більші за 0,4 — статистично значущих відмінностей за статтю немає. Це ще один аргумент
не включати стать до ознак кластеризації. Помітна лише форма розподілу Spending: у чоловіків він ширший
(більше як дуже низьких, так і дуже високих значень).</p>
{F("02_distributions", "Гістограми з KDE та скрипкові діаграми за статтю", 84)}

<h3>3.2 Вік і витрати</h3>
{F("03_age_groups", "Середній Spending Score (стовпці, 95% ДІ) і частка клієнтів зі Spending &gt; 60 (лінія) за віковими групами", 64)}
<p>Найактивніші покупці — у групі 26–35 років (середній Spending {num(ag[1]['spending'], 1)}, у
{pct(ag[1]['high_spenders'], 0)} клієнтів Spending &gt; 60). Після 45 років <b>жодного</b> клієнта зі Spending &gt; 60 немає,
хоча середній дохід у цих групах ({num(ag[3]['income'], 1)} та {num(ag[4]['income'], 1)} k$) не нижчий, ніж у
наймолодшої ({num(ag[0]['income'], 1)} k$). Отже, високі витрати — характеристика молодших клієнтів, але вік сам
по собі не задає групи: всередині кожної вікової групи є і економні, і середні покупці.</p>

<h3>3.3 Кореляції і попарні залежності</h3>
{F("04_correlations", "Матриці кореляцій Пірсона і Спірмена", 80)}
<p>Єдиний помітний зв'язок — від'ємна кореляція віку і Spending (r = {num(pear['spending']['age'])}).
Дохід майже не корелює з витратами (r = {num(pear['spending']['income'])}): лінійно Spending з доходу не випливає,
проте на діаграмі «дохід — Spending» (нижній рядок рис. 5) чітко видно п'ять скупчень, розташованих «хрестом».
Кореляція описує лише лінійну залежність і не бачить такої групової структури — саме для неї потрібна кластеризація.</p>
{F("05_pairplot", "Попарні діаграми розсіювання (кольори — стать)", 56)}

<h3>3.4 Кластерна тенденція: статистика Гопкінса</h3>
<p>Перш ніж кластеризувати, варто перевірити, чи є у даних структура взагалі: K-means розіб'є на k груп навіть
рівномірний шум. Статистика Гопкінса H = Σu / (Σu + Σw) порівнює відстані до найближчого сусіда для
реальних точок (w) та для рівномірно згенерованих у тих самих межах (u). H ≈ 0,5 — структури немає, H → 1 —
дані згруповані. Обчислено 200 разів на випадкових підвибірках по 10% точок.</p>
{T("Статистика Гопкінса (середнє ± σ за 200 повторами)")}
{table(["Дані", "H"],
       [["F2 (дохід, Spending), стандартизовані", f"{num(hop['F2']['mean'], 3)} ± {num(hop['F2']['std'], 3)}"],
        ["F3 (вік, дохід, Spending), стандартизовані", f"{num(hop['F3']['mean'], 3)} ± {num(hop['F3']['std'], 3)}"],
        ["Контроль: рівномірний шум 200 × 2", f"{num(hop['uniform']['mean'], 3)} ± {num(hop['uniform']['std'], 3)}"]])}
<p>Для реальних даних H ≈ 0,7 і суттєво відрізняється від контрольного значення ≈ 0,5, тобто кластерна
структура існує, і для F2 вона трохи виразніша, ніж для F3.</p>

<h2>4. Алгоритм K-means</h2>
<p>K-means шукає розбиття на k кластерів, що мінімізує суму квадратів відстаней до центроїдів
(inertia, WCSS) = Σᵢ ‖xᵢ − μ<sub>c(i)</sub>‖². Алгоритм Ллойда повторює два кроки: (1) кожна точка
призначається найближчому центроїду; (2) кожен центроїд замінюється середнім своїх точок. Жоден крок
не збільшує WCSS, тому процес збігається, але лише до <i>локального</i> мінімуму, що залежить від стартових
центрів. Ініціалізація <b>k-means++</b> обирає перший центр випадково, а кожен наступний — з імовірністю,
пропорційною D(x)², квадрату відстані до найближчого вже обраного центру; так стартові центри рознесені
по даних. Додатково виконується <code>n_init</code> запусків і береться найкращий.</p>
<p>Щоб переконатися в розумінні алгоритму, його реалізовано самостійно на NumPy (k-means++ + ітерації
Ллойда, 10 запусків) і звірено з <code>sklearn.cluster.KMeans</code> на F2 при k = 5:</p>
{T("Власна реалізація проти scikit-learn")}
{table(["Показник", "Значення"],
       [["Inertia: власна / scikit-learn", f"{num(own['own_inertia'], 4)} / {num(own['sklearn_inertia'], 4)}"],
        ["ARI між розбиттями", num(own["ari"], 3)],
        ["Макс. відстань між відповідними центроїдами", f"{own['max_center_diff']:.1e}".replace(".", ",")],
        ["Ітерацій до збіжності (найкращий запуск)", str(own["own_n_iter"])]])}
<p>Розбиття повністю збігаються, центроїди — з точністю до похибки обчислень з плаваючою комою.</p>

<h2>5. Вибір ознак і кількості кластерів</h2>
<p>Для обох наборів ознак і k = 2…10 обчислено чотири критерії (кожна модель — 20 запусків k-means++):</p>
<ul>
<li><b>Метод ліктя</b> — inertia завжди спадає зі зростанням k, шукаємо точку, після якої виграш різко зменшується.
Щоб не вибирати «на око», лікоть визначено алгоритмом <b>Kneedle</b>: криву нормалізують до [0, 1] і беруть точку
з найбільшою відстанню до хорди між крайніми точками;</li>
<li><b>Silhouette</b> s = (b − a)/max(a, b), де a — середня відстань до своїх, b — до найближчого чужого кластера; більше — краще;</li>
<li><b>Calinski–Harabasz</b> — відношення міжкластерної дисперсії до внутрішньокластерної з поправкою на k; більше — краще;</li>
<li><b>Davies–Bouldin</b> — середня по кластерах максимальна «схожість» із сусідом (розкид / відстань між центрами); менше — краще.</li>
</ul>
{T("Критерії для набору F2 (дохід, Spending)")}
{table(["k"] + [str(r["k"]) for r in scan2],
       [["Inertia"] + [num(r["inertia"], 1) for r in scan2],
        ["Silhouette"] + [("<b>%s</b>" if r["k"] == 5 else "%s") % num(r["silhouette"], 3) for r in scan2],
        ["Calinski–Harabasz"] + [num(r["calinski_harabasz"], 1) for r in scan2],
        ["Davies–Bouldin"] + [("<b>%s</b>" if r["k"] == 5 else "%s") % num(r["davies_bouldin"], 3) for r in scan2]], "compact")}
{T("Найкраще k за кожним критерієм")}
{table(["Набір ознак", "Лікоть (Kneedle)", "Silhouette", "Calinski–Harabasz", "Davies–Bouldin", "Найкращий silhouette"],
       [["F2: дохід, Spending"] + [str(votes["F2"][m]) for m in ["elbow", "silhouette", "calinski_harabasz", "davies_bouldin"]]
        + [num(max(r["silhouette"] for r in scan2), 3)],
        ["F3: вік, дохід, Spending"] + [str(votes["F3"][m]) for m in ["elbow", "silhouette", "calinski_harabasz", "davies_bouldin"]]
        + [num(max(r["silhouette"] for r in scan3), 3)]])}
{F("06_k_selection", "Чотири критерії вибору k для наборів F2 і F3; кружком позначено найкраще значення")}
<p><b>Рішення.</b> Для F2 три критерії з чотирьох одностайно дають <b>k = 5</b>, і при цьому досягаються найкращі
значення silhouette ({num(q['silhouette'], 3)}) і Davies–Bouldin ({num(q['davies_bouldin'], 3)}). Calinski–Harabasz
після різкого стрибка при k = 5 далі майже не змінюється (243–271) і формально максимальний при k = 9 — цей критерій відомий
тим, що завищує k, коли кластери щільні й мають різний розмір. Для F3 критерії розходяться (4, 6, 6, 7), лікоть нечіткий,
а найкращий silhouette — лише {num(max(r['silhouette'] for r in scan3), 3)}. Тож основна модель — <b>F2, k = 5</b>.
Вік не відкидається: він використовується для опису сегментів (розд. 7) і в окремому експерименті (розд. 8.4).
Зауважимо, що silhouette у просторах різної розмірності порівнюється обережно, тому рішення спирається не лише на
нього, а й на узгодженість критеріїв та інтерпретовність.</p>

<h2>6. Вплив ініціалізації</h2>
<p>Для k = 5 виконано по 100 одиночних запусків (<code>n_init = 1</code>) з ініціалізацією k-means++ і з випадковим
вибором стартових центрів.</p>
{T("Одиночні запуски K-means (k = 5, F2)")}
{table(["Ініціалізація", "Середня inertia", "Найгірша inertia", "Частка запусків із глобальним мінімумом", "Середня кількість ітерацій"],
       [[name, num(init[name]["mean_inertia"], 2), num(init[name]["worst_inertia"], 2), pct(init[name]["share_global"], 0),
         num(init[name]["mean_iter"], 2)] for name in ["k-means++", "random"]])}
{F("07_init", "Inertia окремих запусків; пунктир — найкращий знайдений мінімум", 62)}
<p>Значення inertia утворюють кілька дискретних рівнів — це різні локальні мінімуми. Наприклад, у розв'язку з inertia ≈ 98,7
«Преміум-покупці» розділені на дві групи, а «Ощадливі» поглинуті масовим сегментом (кластер зі 100 клієнтів). k-means++ потрапляє в глобальний мінімум
{pct(init['k-means++']['share_global'], 0)} проти {pct(init['random']['share_global'], 0)} при випадковому старті й збігається
швидше. Тому фінальна модель використовує k-means++ з <code>n_init = 50</code>: імовірність пропустити глобальний мінімум
за 50 запусків ≈ (1 − {num(init['k-means++']['share_global'], 2)})⁵⁰ — практично нуль.</p>

<h2>7. Результат: п'ять сегментів</h2>
<p>Номери кластерів K-means довільні, тому сегменти перенумеровано за фіксованим правилом
(рівень доходу та Spending центроїда: низький / середній / високий) і названо.</p>
{T("Профілі сегментів (вік і стать у кластеризації не брали участі)")}
{table(["№", "Сегмент", "Клієнтів", "Дохід, k$", "Spending", "Вік: середній (min–max)", "Жінки", "Silhouette"],
       [[str(p["cluster"]), p["segment"], f"{p['n']} ({pct(p['share'], 1)})", num(p["income"], 1), num(p["spending"], 1),
         f"{num(p['age'], 1)} ({p['age_min']}–{p['age_max']})", pct(p["female_share"], 0), num(p["silhouette"], 3)] for p in prof])}
{F("11_regions", "Сегменти на площині «дохід — Spending» і області тяжіння центроїдів", 74)}
<p>Межі областей на рис. 8 — це діаграма Вороного: кожна нова точка належить сегменту з найближчим
центроїдом (у стандартизованому просторі). Модель можна застосовувати до нових клієнтів через
<code>scaler.transform</code> + <code>kmeans.predict</code>.</p>
{F("13_profiles", "Профілі сегментів: z-оцінки середніх, вік і склад за статтю")}
<p><b>Вік</b> не використовувався при кластеризації, але суттєво відрізняється між сегментами (тест Краскела–Уолліса,
p = {kruskal_str}): «Преміум-покупці» — {P[0]['age_min']}–{P[0]['age_max']} років,
«Активні з малим доходом» — {P[3]['age_min']}–{P[3]['age_max']}, тоді як «Масовий сегмент» охоплює всі вікові групи.
<b>Стать</b> між сегментами розподілена однаково (χ², p = {num(S['gender_chi2_p'], 2)}), частка жінок — від
{pct(min(p['female_share'] for p in prof), 0)} до {pct(max(p['female_share'] for p in prof), 0)}.</p>

<h2>8. Оцінка якості</h2>
<h3>8.1 Внутрішні метрики</h3>
{T("Метрики фінальної моделі (F2, k = 5)")}
{table(["Метрика", "Значення", "Як читати"],
       [["Silhouette (середній)", num(q["silhouette"], 3), "&gt; 0,5 — розумна, чітка структура"],
        ["Davies–Bouldin", num(q["davies_bouldin"], 3), "чим менше, тим краще розділені кластери; &lt; 1 — добре"],
        ["Calinski–Harabasz", num(q["calinski_harabasz"], 1), "чим більше, тим краще (порівнюється між k)"],
        ["Inertia (WCSS)", num(q["inertia"], 2), "у стандартизованих одиницях"],
        ["Точок із silhouette &lt; 0 / &lt; 0,25", f"{q['negative_silhouette']} / {q['low_silhouette']}", "граничні клієнти між сусідніми сегментами"]])}
{F("09_silhouette", "Silhouette-діаграма: кожна смуга — один клієнт", 56)}
<p>Усі п'ять сегментів мають середній silhouette понад 0,5, жоден не «провалюється» нижче середнього цілком.
Лише {q['negative_silhouette']} клієнти мають від'ємний silhouette (близький до нуля): доходи 39 і 77 k$ при Spending 36 —
вони лежать на межі «Масового сегмента» з «Ощадливими» та «Стриманими заможними».</p>

<h3>8.2 Стабільність (bootstrap)</h3>
<p>200 разів формувалася bootstrap-вибірка (200 клієнтів з поверненням), на ній навчався K-means, і нова модель
розмічала всіх клієнтів. Порівнювалося з еталонним розбиттям: ARI для розбиття в цілому і коефіцієнт Жаккара
|A ∩ B| / |A ∪ B| для кожного сегмента (з найкращим відповідником). За Hennig, Jaccard &gt; 0,85 — кластер дуже стабільний.</p>
{T("Bootstrap-стабільність (200 повторів)")}
{table(["Сегмент", "Jaccard: середній", "Jaccard: 5-й перцентиль"],
       [[n, num(st["jaccard_mean"][n], 3), num(st["jaccard_p05"][n], 3)] for n in NAMES]
       + [["<b>Усе розбиття (ARI)</b>", f"<b>{num(st['ari_mean'], 3)} ± {num(st['ari_std'], 3)}</b>", f"мін. {num(st['ari_min'], 3)}"]])}
<p>Навіть у 5% найгірших випадків кожен сегмент зберігає понад 90% складу — розбиття стійке до зміни вибірки,
а не є артефактом конкретних 200 клієнтів.</p>

<h3>8.3 Порівняння з іншими алгоритмами</h3>
{T("Інші алгоритми на тих самих ознаках F2")}
{table(["Алгоритм", "Кластерів", "Шум", "Silhouette", "ARI з K-means"],
       [["K-means (k-means++)", "5", "—", num(alt["K-means"]["silhouette"], 3), "1,000"],
        ["Агломеративна, метод Ворда", "5", "—", num(alt["Agglomerative (Ward)"]["silhouette"], 3), num(alt["Agglomerative (Ward)"]["ari"], 3)],
        [f"DBSCAN (eps = {num(alt['DBSCAN']['eps'], 2)}, min_samples = {alt['DBSCAN']['min_samples']})", str(alt["DBSCAN"]["clusters"]),
         pct(alt["DBSCAN"]["noise"], 0), num(alt["DBSCAN"]["silhouette"], 3) + "*", num(alt["DBSCAN"]["ari"], 3) + "*"]])}
<p class="note">* без точок-шуму.</p>
{F("10_alternatives", "Дендрограма Ворда з розрізом на 5 кластерів; графік k-distance для вибору eps; результат DBSCAN")}
<p>Ієрархічна кластеризація Ворда, яка працює зовсім інакше (послідовне злиття), знаходить майже те саме розбиття
(ARI = {num(alt['Agglomerative (Ward)']['ari'], 3)}), а на дендрограмі добре видно великий «стрибок» відстані злиття
після 5 кластерів. DBSCAN знаходить щільне ядро «Масового сегмента» і чотири кутові групи, але розріджені
групи ділить на частини, а найбагатших клієнтів позначає як шум: алгоритм, що спирається на щільність, погано
підходить для груп різної щільності. Загалом три різні підходи підтверджують ту саму п'ятигрупову структуру.</p>

<h3>8.4 Що змінює вік: модель на трьох ознаках</h3>
{F("08_f3_crosstab", "Таблиця спряженості: сегменти F2 (рядки) проти кластерів моделі F3 з k = 6 (стовпці)", 74)}
<p>Модель F3 (k = 6, silhouette {num(f3['silhouette'], 3)}) майже без змін відтворює чотири «кутові» сегменти,
а єдина суттєва відмінність — «Масовий сегмент» ділиться за віком на дві групи:
{mass_split[mass_old]} клієнтів із середнім віком {num(f3_prof[int(mass_old)]['age'], 0)} років і {mass_split[mass_young]} — з
{num(f3_prof[int(mass_young)]['age'], 0)} років (ARI між моделями — {num(f3['ari_vs_f2'], 3)}, низький саме через цей поділ).
Тобто вік не створює нових поведінкових груп, а лише уточнює найбільшу з них, і цей поділ доцільно використовувати
як підсегментацію для комунікації, не ускладнюючи основну модель.</p>

<h2>9. Візуалізація методом PCA</h2>
<p>Фінальні сегменти побудовано у двовимірному просторі, тож вони відображаються напряму (рис. 8). Щоб подивитися,
як сегменти лягають у повному просторі ознак {{вік, дохід, Spending}}, застосовано метод головних компонент до
стандартизованого F3 і побудовано біплот — точки разом із векторами навантажень вихідних ознак.</p>
{T("Навантаження головних компонент (F3)")}
{table(["Ознака", "PC1", "PC2", "PC3"],
       [[feat_ua[c], num(pca['loadings']['PC1'][c], 3), num(pca['loadings']['PC2'][c], 3), num(pca['loadings']['PC3'][c], 3)]
        for c in ["age", "income", "spending"]]
       + [["<b>Частка дисперсії</b>"] + [f"<b>{pct(v, 1)}</b>" for v in pca["explained"]]])}
{F("12_pca_biplot", "PCA-біплот: точки — клієнти (кольори — сегменти), стрілки — напрями вихідних ознак", 68)}
<p>PC1 ({pct(pca['explained'][0], 1)}) — це протиставлення «вік ↔ Spending» (навантаження ±0,71): праворуч старші й економніші,
ліворуч молодші й активніші. PC2 ({pct(pca['explained'][1], 1)}) — фактично дохід (0,999). Дві компоненти зберігають
{pct(pca['explained'][0] + pca['explained'][1], 1)} дисперсії. Сегменти добре розділені й у цій проєкції:
«Преміум-покупці» та «Активні з малим доходом» — ліворуч (високий Spending), «Стримані заможні» та «Ощадливі» — праворуч,
розведені по вертикалі доходом; «Масовий сегмент» витягнутий уздовж PC1, бо охоплює всі вікові групи.</p>

<h2>10. Бізнес-інтерпретація</h2>
{T("Сегменти і рекомендовані дії")}
{table(["Сегмент", "Портрет", "Що робити"],
       [[f"<b>{P[0]['segment']}</b><br>{P[0]['n']} кл., {pct(P[0]['share'], 1)}",
         f"дохід ≈ {num(P[0]['income'], 0)} k$, Spending ≈ {num(P[0]['spending'], 0)}; вік {P[0]['age_min']}–{P[0]['age_max']}",
         "найцінніший сегмент — утримання: програма лояльності з кешбеком, ранній доступ до колекцій, персональний сервіс; відстежувати відтік"],
        [f"<b>{P[1]['segment']}</b><br>{P[1]['n']} кл., {pct(P[1]['share'], 1)}",
         f"дохід ≈ {num(P[1]['income'], 0)} k$, Spending ≈ {num(P[1]['spending'], 0)}; половина — 34–48 років, чоловіків трохи більше",
         "головний резерв зростання: з'ясувати причини низьких витрат (опитування), робити акцент на якості, сервісі й зручності, а не на знижках"],
        [f"<b>{P[2]['segment']}</b><br>{P[2]['n']} кл., {pct(P[2]['share'], 1)}",
         f"дохід ≈ {num(P[2]['income'], 0)} k$, Spending ≈ {num(P[2]['spending'], 0)}; усі вікові групи",
         "ядро трафіку: регулярні акції, бонусні бали, сезонні пропозиції; комунікацію ділити на молодших (≈ 27 р.) і старших (≈ 56 р.)"],
        [f"<b>{P[3]['segment']}</b><br>{P[3]['n']} кл., {pct(P[3]['share'], 1)}",
         f"дохід ≈ {num(P[3]['income'], 0)} k$, Spending ≈ {num(P[3]['spending'], 0)}; наймолодші ({P[3]['age_min']}–{P[3]['age_max']})",
         "мас-маркет і тренди, соцмережі, студентські знижки; розстрочку пропонувати обережно (високі витрати при низькому доході)"],
        [f"<b>{P[4]['segment']}</b><br>{P[4]['n']} кл., {pct(P[4]['share'], 1)}",
         f"дохід ≈ {num(P[4]['income'], 0)} k$, Spending ≈ {num(P[4]['spending'], 0)}; медіана віку 46, половина — 36–54 роки",
         "розпродажі, дисконт, товари першої потреби; мінімальний маркетинговий бюджет"]], "wide")}
<p>Головний висновок для бізнесу: дохід не визначає витрати. Дві групи з однаковим високим доходом
(≈ 87–88 k$) — «Преміум-покупці» і «Стримані заможні» — поводяться протилежно, і саме друга з них є найбільшою
можливістю для зростання виручки. Водночас серед клієнтів з низьким доходом є дуже активні покупці.</p>

<h2>11. Обмеження</h2>
<ul>
<li>Мала навчальна вибірка (200 клієнтів), лише три числові ознаки; Spending Score — непрозорий індекс, невідомо, як саме його розраховано.</li>
<li>Немає транзакційних і часових даних: справжня сегментація зазвичай будується на RFM-ознаках (давність, частота, сума покупок).</li>
<li>K-means припускає опуклі кластери приблизно однакового розміру й розкиду та чутливий до масштабу ознак і викидів.</li>
<li>Silhouette для F2 і F3 обчислено у просторах різної розмірності, тому їх пряме порівняння лише орієнтовне.</li>
<li>Можливі покращення: K-prototypes для змішаних (числових і категоріальних) ознак, GMM для еліпсоїдних кластерів, перевірка моделі на нових клієнтах.</li>
</ul>

<h2>12. Висновки</h2>
<p>Дані чисті (пропусків, дублікатів і помилкових значень немає, два «викиди» за IQR — реальні клієнти) і мають
виражену кластерну структуру (статистика Гопкінса ≈ 0,7 проти 0,5 для шуму). За узгодженням трьох із чотирьох
критеріїв обрано K-means на ознаках «дохід» і «Spending Score» з <b>k = 5</b>. Власна реалізація алгоритму на NumPy
дала результат, ідентичний scikit-learn. Отримане розбиття якісне (silhouette {num(q['silhouette'], 3)},
Davies–Bouldin {num(q['davies_bouldin'], 3)}), стабільне (bootstrap ARI {num(st['ari_mean'], 3)}, Jaccard кожного
сегмента ≥ {num(min(st['jaccard_mean'].values()), 2)}) і підтверджується ієрархічною кластеризацією (ARI {num(alt['Agglomerative (Ward)']['ari'], 2)}).
Виділено п'ять сегментів зі зрозумілою інтерпретацією; вік і стать у моделі не використовувались, але вік суттєво
відрізняє сегменти і корисний для підсегментації масової групи, а стать на поведінку не впливає.</p>
"""

CSS = """
@page { size: A4; margin: 15mm 15mm 15mm 17mm; }
body { font-family: Cambria, 'Times New Roman', serif; font-size: 10.8pt; line-height: 1.38; color: #1b1b1b; }
header { border-bottom: 2px solid #1d3557; padding-bottom: 6px; margin-bottom: 10px; }
.kicker { font-family: Calibri, sans-serif; font-size: 9pt; letter-spacing: .06em; text-transform: uppercase; color: #6b7a8f; }
h1 { font-size: 19pt; color: #1d3557; margin: 4px 0 8px; line-height: 1.2; }
h1 span { display: block; font-size: 13pt; font-weight: normal; color: #333; margin-top: 4px; }
.meta { margin: 4px 0; }
h2 { font-size: 13.5pt; color: #1d3557; margin: 16px 0 6px; padding-bottom: 2px; border-bottom: 1px solid #c9d3df; break-after: avoid; }
h3 { font-size: 11.5pt; color: #2b4a6f; margin: 11px 0 4px; break-after: avoid; }
p { margin: 4px 0 7px; text-align: justify; }
ul { margin: 3px 0 8px; padding-left: 20px; }
li { margin: 2px 0; }
code { font-family: Consolas, monospace; font-size: 9.3pt; background: #f1f3f6; padding: 0 3px; border-radius: 2px; }
a { color: #1d3557; }
table { border-collapse: collapse; margin: 3px auto 9px; font-size: 9.6pt; break-inside: avoid; width: 100%; }
th { background: #1d3557; color: #fff; font-weight: 600; padding: 4px 6px; text-align: left; font-family: Calibri, sans-serif; }
td { padding: 3px 6px; border-bottom: 1px solid #dde3ea; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f6f8fa; }
table.compact th, table.compact td { text-align: center; padding: 3px 4px; }
table.compact td:first-child, table.compact th:first-child { text-align: left; }
table.wide td:first-child { width: 22%; }
.tcap { font-family: Calibri, sans-serif; font-size: 9.5pt; color: #444; margin: 8px 0 2px; text-align: left; break-after: avoid; }
.note { font-size: 9pt; color: #555; margin-top: -6px; }
figure { margin: 6px 0 10px; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-family: Calibri, sans-serif; font-size: 9.3pt; color: #444; margin-top: 2px; }
"""


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    html = (f'<!doctype html><html lang="uk"><head><meta charset="utf-8">'
            f"<title>Customer Segmentation — Герасименко</title><style>{CSS}</style></head><body>{body}</body></html>")
    HTML_PATH.write_text(html, encoding="utf-8")
    PDF_PATH.unlink(missing_ok=True)
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF_PATH}", HTML_PATH.as_uri()], check=True, capture_output=True, timeout=120)
    last, stable = -1, 0
    for _ in range(120):
        size = PDF_PATH.stat().st_size if PDF_PATH.exists() else -1
        stable = stable + 1 if size == last and size > 0 else 0
        if stable >= 4:
            break
        last = size
        time.sleep(0.25)
    else:
        raise RuntimeError("PDF was not written")
    print(f"written {PDF_PATH}")


if __name__ == "__main__":
    main()
