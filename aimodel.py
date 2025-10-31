import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

flask_app = Flask(__name__)

#Чтобы в консоль не было информировании о запуске модели
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

#Сама модель для работы с запросом
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L6-v2')
model = model.half()

#Наши категории из таблицы
category_names = [
    "Памятники и скульптуры",
    "Парки, скверы и зоны отдыха",
    "Макеты архитектурных объектов",
    "Набережные",
    "Архитектура и исторические здания",
    "Культурно-досуговые центры и библиотеки",
    "Музеи и выставочные пространства",
    "Театры и филармонии",
    "Инфраструктура",
    "Монументально-декоративное искусство",
    "Рестораны и кафе",
    "Кофейни",
    "Кондитерские и пекарни",
    "Торговые центры",
    "Места для развлечения"
]

#Перевод для дальнейшего сравнения
category_embeddings = model.encode(category_names, convert_to_tensor=True, show_progress_bar=False)

#Перегрузка функции, которая возвращает топ категории мест
@flask_app.route('/define_categories', methods=['POST'])
def define_categories_endpoint():
    """
    Определяет топ категорий для запроса пользователя с использованием sentence-transformers.

    Параметры:
        text (str) - текст запроса пользователя
        similarity_threshold (float) - минимальное значение косинусного сходства для включения категории
        min_categories (int) - минимальное количество категорий в топе
        max_categories (int) - максимальное количество категорий в топе

    Возвращает:
        list of tuples: [(category_id, score), ...] — топ категорий с их схожестью
    """
    data = request.json
    text = data.get('text')
    similarity_threshold = data.get('similarity_threshold')
    min_categories = data.get('min_categories')
    max_categories = data.get('max_categories')

#Кодируем запрос в вектор с помощью модели sentence-transformers.
    query_emb = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
#Считаем косинусное сходство запроса с векторами категорий.
    similarities = util.cos_sim(query_emb, category_embeddings)[0]

#Сортируем категории по схожести.
    sorted_indices = torch.argsort(similarities, descending=True).tolist()
    sorted_scores = similarities[sorted_indices].tolist()

    found=[]

#Добавляем в результат только те, у которых сходство ≥ similarity_threshold.
    for idx, score in zip(sorted_indices, sorted_scores):
        if score >= similarity_threshold:
            found.append((idx+1, score))
        if len(found) >= max_categories:
            break
#Если найдено меньше min_categories, добавляем следующие по схожести, чтобы гарантировать минимум.
    if len(found) < min_categories:
        for idx, score in zip(sorted_indices, sorted_scores):
            if (idx + 1, score) not in found:
                found.append((idx + 1, score))
            if len(found) >= min_categories:
                break

#Возвращаем список кортежей (category_id, score).
    return jsonify({'categories': found[:max_categories]})

if __name__ == '__main__':

    flask_app.run(host='0.0.0.0', port=5000, debug=False)
