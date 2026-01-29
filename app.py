import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import sqlite3
import pandas as pd
from datetime import datetime
import io
import tempfile
import time

TARGET_CLASSES = [41, 39]
DB_FILE = 'coffee_shop.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS orders
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  max_items_detected INTEGER,
                  final_items_at_stop INTEGER,
                  duration_seconds INTEGER)''')
    conn.commit()
    conn.close()


def save_order(max_count, final_count, duration):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO orders (timestamp, max_items_detected, final_items_at_stop, duration_seconds) VALUES (?, ?, ?, ?)",
        (timestamp, max_count, final_count, int(duration)))
    conn.commit()
    conn.close()
    return timestamp


def get_stats():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC", conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

def main():
    st.set_page_config(page_title="Coffee AI Counter", layout="wide")
    init_db()

    if 'video_pos' not in st.session_state: st.session_state.video_pos = 0
    if 'current_max' not in st.session_state: st.session_state.current_max = 0
    if 'last_count' not in st.session_state: st.session_state.last_count = 0

    st.title("☕ Учет заказов в кофейне")

    menu = st.sidebar.selectbox("Меню", ["🎥 Видео-анализ", "📸 Фото-анализ", "📊 Статистика"])
    model = load_model()
    conf_thresh = st.sidebar.slider("Порог уверенности", 0.05, 1.0, 0.15)
    frame_skip = st.sidebar.slider("Ускорение (пропуск кадров)", 1, 10, 2)

    if menu == "🎥 Видео-анализ":
        video_file = st.file_uploader("Загрузить видео", type=['mp4', 'mov', 'avi'])

        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
            with ctrl_col1:
                start_trigger = st.button("▶ ЗАПУСТИТЬ / ПРОДОЛЖИТЬ", use_container_width=True)
            with ctrl_col2:
                stop_trigger = st.button("⏹ ЗАВЕРШИТЬ И СОХРАНИТЬ", use_container_width=True)
            with ctrl_col3:
                if st.button("⏪ СБРОСИТЬ ВИДЕО", use_container_width=True):
                    st.session_state.video_pos = 0
                    st.session_state.current_max = 0
                    st.rerun()

            m_col1, m_col2, m_col3 = st.columns(3)
            curr_metric = m_col1.empty()
            max_metric = m_col2.empty()
            pos_metric = m_col3.empty()

            video_placeholder = st.empty()

            if start_trigger:
                cap = cv2.VideoCapture(tfile.name)
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.video_pos)

                start_time = time.time()
                st.session_state.current_max = 0  # Сброс для нового заказа

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.video_pos = 0
                        st.info("Конец видео.")
                        break

                    curr_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    st.session_state.video_pos = curr_pos

                    if curr_pos % frame_skip != 0:
                        continue

                    results = model(frame, classes=TARGET_CLASSES, conf=conf_thresh)
                    count = len(results[0].boxes)
                    st.session_state.last_count = count

                    if count > st.session_state.current_max:
                        st.session_state.current_max = count

                    curr_metric.metric("Сейчас в кадре", count)
                    max_metric.metric("Максимум в заказе", st.session_state.current_max)
                    pos_metric.write(f"Кадр: {curr_pos}")

                    frame_res = results[0].plot()
                    frame_res = cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_res, use_container_width=True)

                cap.release()

            if stop_trigger:
                if st.session_state.current_max > 0:
                    ts = save_order(st.session_state.current_max, st.session_state.last_count, 0)
                    st.success(
                        f"Заказ сохранен в {ts}! Макс: {st.session_state.current_max}, Финал: {st.session_state.last_count}")
                else:
                    st.warning("Нечего сохранять (0 предметов).")

    elif menu == "📸 Фото-анализ":
        uploaded_file = st.file_uploader("Загрузите фото", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            img = Image.open(uploaded_file)
            if st.button("Проанализировать"):
                results = model(np.array(img), classes=TARGET_CLASSES, conf=conf_thresh)
                count = len(results[0].boxes)
                st.image(results[0].plot(), use_container_width=True)
                save_order(count, count, 0)
                st.success(f"Найдено и сохранено: {count}")

    elif menu == "📊 Статистика":
        df = get_stats()
        if not df.empty:
            st.metric("Всего стаканов продано", df['max_items_detected'].sum())
            st.dataframe(df, use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Скачать отчет (Excel)", buffer.getvalue(), "coffee_report.xlsx")

            if st.button("🗑 Очистить БД"):
                conn = sqlite3.connect(DB_FILE);
                conn.cursor().execute("DELETE FROM orders");
                conn.commit();
                conn.close()
                st.rerun()
        else:
            st.info("Нет данных")


if __name__ == "__main__":
    main()