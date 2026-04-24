"""Streamlit приложение за засичане и разпознаване на хора по походка.

Стартиране:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from gait.classifier import GaitClassifier
from gait.dataset import build_gei_dataset, load_dataset
from gait.gei import compute_gei, detect_gait_cycle
from gait.silhouette import extract_silhouettes
from gait.visualize import plot_confusion_matrix, plot_gei_grid

# ---------------------------------------------------------------------------
# Конфигурация и константи
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data" / "datasets"
MODEL_DIR = APP_ROOT / "models"
REPORT_DIR = APP_ROOT / "reports"
RUNS_FILE = MODEL_DIR / "runs.json"

for p in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

st.set_page_config(
    page_title="Разпознаване по походка",
    page_icon="🚶",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Помощни функции
# ---------------------------------------------------------------------------


def _save_run(meta: dict) -> None:
    runs = _load_runs()
    runs.append(meta)
    RUNS_FILE.write_text(json.dumps(runs, indent=2, ensure_ascii=False))


def _load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    return json.loads(RUNS_FILE.read_text())


def _extract_zip_to(zip_bytes: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest)


def _flatten_dataset_root(root: Path) -> Path:
    """Ако zip-ът е с единична главна папка, влез в нея."""
    children = [p for p in root.iterdir() if not p.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        inner_children = [p for p in children[0].iterdir() if not p.name.startswith(".")]
        if all(p.is_dir() for p in inner_children):
            return children[0]
    return root


def _list_datasets() -> list[Path]:
    return sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])


def _dataset_summary(path: Path) -> dict:
    data = load_dataset(path)
    return {
        "subjects": len(data),
        "videos": sum(len(v) for v in data.values()),
        "breakdown": {k: len(v) for k, v in data.items()},
    }


def _list_models() -> list[Path]:
    return sorted(MODEL_DIR.glob("*.joblib"))


def _gei_to_png_bytes(gei: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", gei)
    if not ok:
        return b""
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Странични елементи
# ---------------------------------------------------------------------------

st.sidebar.title("🚶 Разпознаване по походка")
page = st.sidebar.radio(
    "Раздели",
    ["Общ преглед", "Алгоритъм и формули", "Обучение", "Разпознаване", "История на експериментите"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Набори данни: **{len(_list_datasets())}**")
st.sidebar.caption(f"Обучени модели: **{len(_list_models())}**")


# ---------------------------------------------------------------------------
# Раздел: Общ преглед
# ---------------------------------------------------------------------------

if page == "Общ преглед":
    st.title("Засичане и разпознаване на хора по походка")
    st.markdown(
        """
Това приложение демонстрира пълен пайплайн за **биометрична идентификация**
по походка. Работният процес се състои от четири основни етапа:

1. **Извличане на силуети** от видеопоток (MOG2 + морфологична обработка)
2. **Нормализация** – изрязване, мащабиране и центриране на силуетите
3. **Gait Energy Image (GEI)** – осредняване на силуетите в един цикъл
4. **Класификация** – k-NN или SVM върху векторизираното GEI

### Как да използвате приложението

- Разделът **„Алгоритъм и формули“** показва математическите основи.
- В **„Обучение“** качвате ZIP файл с набора си от видеа и пускате обучението.
- В **„Разпознаване“** качвате единично видео и получавате топ-k предсказания.
- **„История на експериментите“** пази всички минали обучения.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Налични набори", len(_list_datasets()))
    col2.metric("Обучени модели", len(_list_models()))
    runs = _load_runs()
    best_acc = max((r["accuracy"] for r in runs), default=0.0)
    col3.metric("Най-висока точност", f"{best_acc * 100:.1f}%" if runs else "—")

    st.info(
        "Формат на набора: ZIP архив, в който всяка директория е един субект, "
        "а вътре са видеа с негова походка."
    )
    with st.expander("Примерна структура на ZIP"):
        st.code(
            """dataset.zip
├── subject_01/
│   ├── walk_01.mp4
│   ├── walk_02.mp4
│   └── walk_03.mp4
├── subject_02/
│   ├── walk_01.mp4
│   └── walk_02.mp4
└── subject_03/
    └── ...""",
            language="text",
        )


# ---------------------------------------------------------------------------
# Раздел: Алгоритъм и формули
# ---------------------------------------------------------------------------

elif page == "Алгоритъм и формули":
    st.title("Алгоритъм и математически основи")

    st.header("1. Моделиране на фона – MOG2")
    st.markdown(
        """
Всеки пиксел $(x, y)$ се моделира като **смес от $K$ Гаусови разпределения**.
Вероятността за наблюдение на интензитет $X_t$ в момент $t$ е:
        """
    )
    st.latex(r"P(X_t) = \sum_{i=1}^{K} \omega_{i,t} \cdot \eta\!\left(X_t;\, \mu_{i,t},\, \Sigma_{i,t}\right)")
    st.markdown(
        r"""
Където $\omega_{i,t}$ са тегла, а $\eta$ е Гаусовата плътност. Пикселът се
класифицира като **преден план**, ако не съвпада с нито една от компонентите
с високо тегло. Обновяването е:
        """
    )
    st.latex(r"\mu_{t} = (1 - \rho)\mu_{t-1} + \rho X_t,\quad \rho = \alpha\, \eta(X_t;\,\mu_{t-1},\,\Sigma_{t-1})")

    st.header("2. Морфологична обработка")
    st.markdown(
        r"""
След MOG2 получената маска $M$ съдържа шум. Прилагаме последователно
**отваряне** (отстраняване на малки петна) и **затваряне** (запълване на дупки):
        """
    )
    st.latex(r"M' = (M \ominus B) \oplus B \quad \text{(opening)}")
    st.latex(r"M'' = (M' \oplus B) \ominus B \quad \text{(closing)}")
    st.markdown("където $B$ е структурен елемент (елипса 5×5), $\\ominus$ е ерозия, $\\oplus$ е дилатация.")

    st.header("3. Нормализация на силуета")
    st.markdown(
        r"""
Всеки силует $S$ се изрязва по ограничаващия правоъгълник $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$
и мащабира до фиксирана височина $H$ със запазване на съотношението:
        """
    )
    st.latex(r"s = \frac{H}{y_{\max} - y_{\min} + 1},\quad W' = \lfloor s \cdot (x_{\max} - x_{\min} + 1) \rfloor")
    st.markdown("След мащабирането силуетът се центрира хоризонтално в платно с ширина $W$.")

    st.header("4. Gait Energy Image (GEI)")
    st.markdown(
        """
GEI е осредненото изображение на нормализираните силуети $S_t$ в един
пълен цикъл на походката с дължина $N$:
        """
    )
    st.latex(r"\mathrm{GEI}(x, y) \;=\; \frac{1}{N} \sum_{t=1}^{N} S_t(x, y)")
    st.markdown(
        """
- **По-светли пиксели** → относително неподвижни части (торс, глава).
- **По-тъмни пиксели** → крайници, които се движат бързо (ръце, крака).

GEI е **компактно** (128 × 88 ≈ 11 k пиксела) и **устойчиво** към шум
представяне, защото осредняването потиска стохастичните грешки.
        """
    )

    st.header("5. Сегментиране на цикъла")
    st.markdown(
        r"""
Широчината $W_t$ на ограничаващия правоъгълник е периодичен сигнал с
максимуми при максимално разтваряне на краката:
        """
    )
    st.latex(r"W_t \;=\; \max(x \mid S_t(x,y) > 0) - \min(x \mid S_t(x,y) > 0)")
    st.markdown(
        "Чрез алгоритъма `scipy.signal.find_peaks` локализираме пиковете; "
        "интервалът между два последователни пика е **половин цикъл**."
    )

    st.header("6. Класификация")
    st.subheader("k-Nearest Neighbors")
    st.latex(r"\hat{y}(\mathbf{x}) \;=\; \mathrm{mode}\!\left\{\, y_i \;\big|\; i \in \mathcal{N}_k(\mathbf{x}) \,\right\}")
    st.markdown(
        r"където $\mathcal{N}_k(\mathbf{x})$ са $k$-те най-близки обучаващи примера по евклидово разстояние:"
    )
    st.latex(r"d(\mathbf{x}, \mathbf{x}') \;=\; \sqrt{\sum_{j=1}^{D} (x_j - x'_j)^2}")

    st.subheader("Support Vector Machine (RBF)")
    st.markdown("Задачата е дуална оптимизация:")
    st.latex(
        r"\max_{\alpha} \;\; \sum_i \alpha_i - \tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)"
    )
    st.markdown("с ядро:")
    st.latex(r"K(\mathbf{x}, \mathbf{x}') \;=\; \exp\!\left( -\gamma \, \|\mathbf{x} - \mathbf{x}'\|^2 \right)")

    st.header("7. Мерки за качество")
    st.markdown(
        r"""
За всеки клас $c$ пресмятаме:
        """
    )
    st.latex(r"\mathrm{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \mathrm{Recall}_c = \frac{TP_c}{TP_c + FN_c}")
    st.latex(r"F_1^{(c)} = 2 \cdot \frac{\mathrm{Precision}_c \cdot \mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}")
    st.latex(r"\mathrm{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\!\left[\hat{y}_i = y_i\right]")


# ---------------------------------------------------------------------------
# Раздел: Обучение
# ---------------------------------------------------------------------------

elif page == "Обучение":
    st.title("Обучение на модел")

    st.markdown(
        "Качете ZIP архив с набор данни или изберете вече качен. "
        "Очакваната структура е: `subject_id/video.mp4`."
    )

    tab_upload, tab_existing = st.tabs(["📤 Качи нов ZIP", "📁 Използвай съществуващ"])

    chosen_dataset: Path | None = None

    with tab_upload:
        up = st.file_uploader("ZIP архив с набора", type=["zip"], key="zip_upload")
        ds_name = st.text_input(
            "Име на набора (без интервали)", value=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if up is not None and st.button("📥 Запази набора", type="primary"):
            target = DATA_DIR / ds_name
            if target.exists():
                st.error(f"Вече има набор с име {ds_name}. Изберете друго име.")
            else:
                with st.spinner("Разархивиране..."):
                    tmp = Path(tempfile.mkdtemp())
                    try:
                        _extract_zip_to(up.getvalue(), tmp)
                        inner = _flatten_dataset_root(tmp)
                        shutil.move(str(inner), str(target))
                    finally:
                        shutil.rmtree(tmp, ignore_errors=True)
                st.success(f"Наборът е записан в `{target}`")
                st.rerun()

    with tab_existing:
        datasets = _list_datasets()
        if not datasets:
            st.info("Няма налични набори. Качете ZIP в другия таб.")
        else:
            names = [d.name for d in datasets]
            selected = st.selectbox("Избери набор", names, key="ds_select")
            chosen_dataset = DATA_DIR / selected

            summary = _dataset_summary(chosen_dataset)
            c1, c2 = st.columns(2)
            c1.metric("Субекти", summary["subjects"])
            c2.metric("Видеа", summary["videos"])
            with st.expander("Видеа по субект"):
                st.table(
                    pd.DataFrame(
                        [(k, v) for k, v in summary["breakdown"].items()],
                        columns=["Субект", "Брой видеа"],
                    )
                )
            if st.button(f"🗑️ Изтрий набора {selected}"):
                shutil.rmtree(chosen_dataset)
                st.rerun()

    st.markdown("---")
    st.subheader("Параметри на обучението")

    col1, col2, col3 = st.columns(3)
    classifier_kind = col1.selectbox("Класификатор", ["knn", "svm"])
    test_size = col2.slider("Тестова част", 0.1, 0.5, 0.3, 0.05)
    seed = col3.number_input("Seed", value=42, step=1)

    k_neighbors = 3
    if classifier_kind == "knn":
        k_neighbors = st.slider("k (брой съседи)", 1, 15, 3, 2)

    model_name = st.text_input(
        "Име на модела", value=f"{classifier_kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    start_disabled = chosen_dataset is None
    if st.button("🚀 Старт на обучението", type="primary", disabled=start_disabled):
        progress = st.progress(0.0, text="Извличане на GEI...")
        status = st.empty()

        try:
            t0 = time.time()
            status.info("Извличане на силуети и изчисляване на GEI от всички видеа...")
            X, y, paths = build_gei_dataset(chosen_dataset, verbose=False)
            progress.progress(0.4, text=f"Получени {len(X)} GEI")

            n_per_class = pd.Series(y).value_counts()
            if n_per_class.min() < 2:
                st.warning(
                    f"Субект '{n_per_class.idxmin()}' има само {n_per_class.min()} валидни GEI — "
                    "разделянето train/test може да пропусне този клас."
                )
                stratify = None
            else:
                stratify = y

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(seed), stratify=stratify
            )
            progress.progress(0.6, text="Обучение на класификатора...")

            clf = GaitClassifier(kind=classifier_kind, k=k_neighbors)
            clf.fit(X_train, y_train)
            progress.progress(0.85, text="Оценка върху тестовата извадка...")

            report = clf.evaluate(X_test, y_test)
            elapsed = time.time() - t0

            model_path = MODEL_DIR / f"{model_name}.joblib"
            clf.save(model_path)

            report_subdir = REPORT_DIR / model_name
            report_subdir.mkdir(parents=True, exist_ok=True)
            plot_gei_grid(
                [X[i] for i in range(min(len(X), 20))],
                [y[i] for i in range(min(len(y), 20))],
                report_subdir / "gei_grid.png",
            )
            plot_confusion_matrix(
                report.confusion,
                report.labels,
                report_subdir / "confusion.png",
                title=f"{classifier_kind.upper()}  acc={report.accuracy:.2f}",
            )

            _save_run(
                {
                    "name": model_name,
                    "dataset": chosen_dataset.name,
                    "classifier": classifier_kind,
                    "k": k_neighbors if classifier_kind == "knn" else None,
                    "test_size": test_size,
                    "seed": int(seed),
                    "n_samples": len(X),
                    "n_subjects": len(set(y)),
                    "accuracy": float(report.accuracy),
                    "elapsed_sec": elapsed,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model_path": str(model_path.relative_to(APP_ROOT)),
                }
            )
            progress.progress(1.0, text="Готово!")

            st.success(
                f"Обучението приключи за {elapsed:.1f} секунди. "
                f"Точност: **{report.accuracy * 100:.2f}%**"
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Точност (test)", f"{report.accuracy * 100:.1f}%")
            m2.metric("Обучаващи примери", len(X_train))
            m3.metric("Тестови примери", len(X_test))

            st.subheader("Отчет по класове")
            st.code(report.per_class)

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Confusion matrix")
                st.image(str(report_subdir / "confusion.png"))
            with c2:
                st.caption("Примерни GEI")
                st.image(str(report_subdir / "gei_grid.png"))

        except Exception as exc:
            st.error(f"Обучението се провали: {exc}")
            raise


# ---------------------------------------------------------------------------
# Раздел: Разпознаване
# ---------------------------------------------------------------------------

elif page == "Разпознаване":
    st.title("Разпознаване на идентичност по видео")

    models = _list_models()
    if not models:
        st.warning("Няма обучени модели. Отидете в раздел **Обучение**.")
        st.stop()

    model_names = [m.stem for m in models]
    selected_model = st.selectbox("Избери модел", model_names)
    model_path = MODEL_DIR / f"{selected_model}.joblib"

    runs = _load_runs()
    meta = next((r for r in runs if r["name"] == selected_model), None)
    if meta:
        c1, c2, c3 = st.columns(3)
        c1.metric("Класификатор", meta["classifier"].upper())
        c2.metric("Точност", f"{meta['accuracy'] * 100:.1f}%")
        c3.metric("Субекти", meta["n_subjects"])

    st.markdown("---")

    uploaded = st.file_uploader(
        "Качи видео с походка (mp4/avi/mov)",
        type=[ext.lstrip(".") for ext in VIDEO_EXTS],
    )
    top_k = st.slider("Top-K предсказания", 1, 5, 3)

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        ) as tf:
            tf.write(uploaded.getvalue())
            video_tmp = Path(tf.name)

        try:
            st.video(uploaded.getvalue())

            with st.spinner("Извличане на силуети..."):
                silhouettes = extract_silhouettes(video_tmp)
            st.write(f"Открити **{len(silhouettes)}** силуета.")

            if len(silhouettes) < 10:
                st.error(
                    "Твърде малко силуети. Възможни причини: твърде кратко видео, "
                    "движеща се камера, лошо осветление."
                )
                st.stop()

            with st.spinner("Изчисляване на GEI и сегментиране на цикли..."):
                cycles = detect_gait_cycle(silhouettes)
                gei = compute_gei(silhouettes)

            if gei is None:
                st.error("Неуспешно построяване на GEI.")
                st.stop()

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Gait Energy Image")
                st.image(_gei_to_png_bytes(gei), use_container_width=True)
                st.caption(f"Намерени цикли: {len(cycles)}")
            with c2:
                st.caption("Пример за силует от средата на видеото")
                mid = silhouettes[len(silhouettes) // 2]
                st.image(_gei_to_png_bytes(mid), use_container_width=True)

            with st.spinner("Класифициране..."):
                clf = GaitClassifier.load(model_path)
                proba, classes = clf.predict_proba(gei[None, ...])
                probs = proba[0]

            order = probs.argsort()[::-1][:top_k]

            st.subheader("Резултат")
            st.success(
                f"🎯 Най-вероятен субект: **{classes[order[0]]}**  "
                f"(увереност {probs[order[0]] * 100:.1f}%)"
            )

            df = pd.DataFrame(
                {
                    "Ранг": range(1, len(order) + 1),
                    "Субект": [classes[i] for i in order],
                    "Вероятност": [f"{probs[i] * 100:.2f}%" for i in order],
                }
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

            chart_df = pd.DataFrame(
                {"Субект": [classes[i] for i in order], "Вероятност": probs[order]}
            ).set_index("Субект")
            st.bar_chart(chart_df)

        finally:
            try:
                video_tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Раздел: История
# ---------------------------------------------------------------------------

elif page == "История на експериментите":
    st.title("История на обученията")
    runs = _load_runs()
    if not runs:
        st.info("Все още няма обучени модели.")
        st.stop()

    df = pd.DataFrame(runs)
    df["accuracy_pct"] = (df["accuracy"] * 100).round(2)
    display = df[
        ["timestamp", "name", "dataset", "classifier", "n_samples", "n_subjects", "accuracy_pct", "elapsed_sec"]
    ].rename(
        columns={
            "timestamp": "Дата",
            "name": "Модел",
            "dataset": "Набор",
            "classifier": "Класификатор",
            "n_samples": "GEI",
            "n_subjects": "Субекти",
            "accuracy_pct": "Точност (%)",
            "elapsed_sec": "Време (s)",
        }
    )
    st.dataframe(display, hide_index=True, use_container_width=True)

    st.subheader("Сравнение на моделите")
    chart = df.set_index("name")[["accuracy"]] * 100
    st.bar_chart(chart, y_label="Точност (%)")

    st.subheader("Отчети")
    for run in reversed(runs):
        with st.expander(f"{run['timestamp']} — {run['name']} ({run['accuracy'] * 100:.1f}%)"):
            st.json(run)
            report_subdir = REPORT_DIR / run["name"]
            if (report_subdir / "confusion.png").exists():
                c1, c2 = st.columns(2)
                c1.image(str(report_subdir / "confusion.png"), caption="Confusion")
                if (report_subdir / "gei_grid.png").exists():
                    c2.image(str(report_subdir / "gei_grid.png"), caption="GEI примери")
