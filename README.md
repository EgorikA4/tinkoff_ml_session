## 🔥 Pipeline

### 🌋 [LLaVA](https://huggingface.co/llava-hf/llava-1.5-7b-hf) -> 🦕 [Groudning DINO](https://github.com/IDEA-Research/Grounded-Segment-Anything) -> 🐻 [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) -> 💸 [OmniFusion](https://huggingface.co/AIRI-Institute/OmniFusion)

Решение данного кейса представляет собой несколько этапов:
1. Нахождение товаров на картинке.
    - Для решения данной задачи используется квантизированная модель LLaVA-v1.5-7b.

2. Сегментация фона.
    - Для детекции боксов используется Grounding DINO.
    - Для сегментации применяется модель Grounded SAM.

3. Замена фона.
    - Фон изображения меняется на указанный цвет (формат RGB).

4. Генерация описания.
    - Для описания используется OmniFusion, потому что данная модель лучше подходит для работы с русским текстом.

## Запуск проекта
...
