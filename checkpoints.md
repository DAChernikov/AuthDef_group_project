# План работы над проектом

## 0. Знакомство, план проекта (до 2 октября)

## 1. Разведочный анализ данных и первичная аналитика данных (TBD)
### 1.1. Сбор и подготовка данных
- поиск размеченных данных (текст и/или аннотация), однозначно связанных с конкретными авторами;
- в случае отсутствия готовых подходящих датасетов реализация парсинга данных из открытых текстовых корпусов;
- очистка данных (удаление ненужных символов, метаинформации);
- токенизация и нормализация текста.

### 1.2. EDA (4 ноября)
- проанализировать распределение текстов/слов по авторам, использование уникальных слов разными авторами, длину предложений;
- проверка дисбаланса классов (и его устранение при необходимости: взвешивание / изменение выборок);
- построение графика WordCloud (облако слов) для оценки возможных различий в стиле;
- определение характерных паттернов для каждого автора: лексические, синтаксические и т.п.

## 2. ML задачи, обучение бейзлайна (20 ноября)
- кодирование текстовой информации с помощью классических лексико-статистических подходов (N-gram, TF-IDF) и векторизации текста (Word2Vec, Doc2Vec, BoW);
- определение метрик для оценки моделей;
- обучение различных моделей на полученных закодированных данных (логистическая регрессия, SVM, random forest, бустинговые модели) для классификации авторства и их сравнение по выбранной метрике/-ам;

Дополнительно:
- посмотреть, как эмбеддинги, полученные с помощью Word2Vec и Doc2Vec, расположены в 2-х мерном пространстве (t-sne);
- решение задачи кластеризации авторов с помощью алгоритмов кластеризации (k-means, DBSCAN, agglomerative clustering, spectral clustering);
- извлечение дополнительных признаков из текстов и их конкатенация с заранее полученными эмбеддингами;
- настройка гиперпараметров.

## 3. Cоздание MVP (10 декабря)
Построение минимально рабочего решения.

## ПРОМЕЖУТОЧНАЯ ЗАЩИТА

## 4. Улучшение бейзлайна

## 5. DL задачи (TBD)
- построение нейросетевого эмбеддинга автора текста:
- использование продвинутых контекстных векторов для кодирования текстов (например, BERT) с дальнейшим обучением различных моделей на полученных эмбеддингах;
- построение нейросетевых моделей на основе векторизованного текста для выделения характерных черт стиля автора;
- обучение моделей и создание соответствующих эмбеддингов;
- сравнение полученных эмбеддингов относительно задачи классификации авторства с ML моделями.

## 6. Доработка MVP в полноценный сервис и его развертывание (TBD)

## ИТОГОВАЯ ЗАЩИТА
