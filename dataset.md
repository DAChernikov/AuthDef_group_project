# Сбор данных
## Источник
В качестве источника используется ондайн-библиотека http://az.lib.ru. Тексты загружаются парсером из следующих разделов:
- Куприн Александр Иванович (http://az.lib.ru/k/kuprin_a_i/)
- Тургенев Иван Сергеевич (http://az.lib.ru/t/turgenew_i_s/)
- Пушкин Александр Сергеевич (http://az.lib.ru/p/pushkin_a_s/)
- Салтыков-Щедрин Михаил Евграфович (http://az.lib.ru/s/saltykow_m_e/)
- Мамин-Сибиряк Дмитрий Наркисович (http://az.lib.ru/m/maminsibirjak_d/)
- Карамзин Николай Михайлович (http://az.lib.ru/k/karamzin_n_m/)
- Лермонтов Михаил Юрьевич (http://az.lib.ru/l/lermontow_m_j/)
-  Достоевский Федор Михайлович (http://az.lib.ru/d/dostoewskij_f_m/)
- Есенин Сергей Александрович (http://az.lib.ru/e/esenin_s_a/)
- Бунин Иван Алексеевич (http://az.lib.ru/b/bunin_i_a/)
- Блок Александр Александрович (http://az.lib.ru/b/blok_a_a/)
- Чехов Антон Павлович (http://az.lib.ru/c/chehow_a_p/)
- Гоголь Николай Васильевич (http://az.lib.ru/g/gogolx_n_w/)

Суммарно более 2500 произведений.
## Парсер
Парсер находится в /parser/parser.py и при запуске делает следующее:
1. Загружает список разделов библиотеки из файла urls.txt
2. Парсит каждый раздел и находит ссылки на произведения
3. Парсит страницы произведений и сохраняет тексты в отдельные файлы в какталоге /data/ (он в .gitignore)
4. Создает json-файлы с таким же названием и в том же каталоге, содержащие дополнительную информацию со страницы произведения.

Пример json-файла:
```
{
  "link": "http://az.lib.ru/b/blok_a_a/text_1918_katilina.shtml",
  "author": "Блок Александр Александрович",
  "title": "Катилина",
  "year": "1918",
  "annotation": "Страница из истории мировой Революции.",
  "genres": [
    "Очерк",
    "История",
    "Публицистика",
    "Публицистика"
  ]
}
```
## Загрузка в облако
После того, как тексты скачаны, запускается скрипт /parser/s3_uploader.py для загрузки их в Yandex Cloud (бакет auth-def-2024).

Список загруженных файлов можно увидеть через API https://storage.yandexcloud.net/auth-def-2024/ и скачать любой из них по ключу https://storage.yandexcloud.net/auth-def-2024/{Key}.
Пример:
```
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>auth-def-2024</Name>
  <Prefix/>
  <MaxKeys>1000</MaxKeys>
  <IsTruncated>true</IsTruncated>
  <Contents>
    <Key>blok_a_a_text_0010.txt</Key>
    <LastModified>2024-10-27T21:33:40.523Z</LastModified>
    <Owner>
      <ID/>
      <DisplayName/>
    </Owner>
    <ETag>"19d8e6771720f8d181e8a0d002aaee2b"</ETag>
    <Size>21397</Size>
    <StorageClass>STANDARD</StorageClass>
  </Contents>
  <Contents>
    <Key>blok_a_a_text_0020.txt</Key>
    <LastModified>2024-10-27T21:34:03.399Z</LastModified>
    <Owner>
      <ID/>
      <DisplayName/>
    </Owner>
    <ETag>"c0f773e5085b986dfdfa570684779a7c"</ETag>
    <Size>1859619</Size>
    <StorageClass>STANDARD</StorageClass>
  </Contents>
```

Также s3_uploader генерирует meta_table.csv с информацией о текстах и ссылками на скачанные локальные файлы и на файлы в облаке. Пример:

| author                     | name            | year | genres                        | annotation     | s3_link                                                                 | local_link                                                 |
| -------------------------- | --------------- | ---- | ----------------------------- | -------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------- |
| Пушкин Александр Сергеевич | Евгений Онегин  | 1830 | ['Поэма', 'Поэзия', 'Поэзия'] | Роман в стихах | https://storage.yandexcloud.net/auth-def-2024/pushkin_a_s_text_0170.txt | ../data/Пушкин Александр Сергеевич.Евгений Онегин.1830.txt |

