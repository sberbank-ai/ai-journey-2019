import random
from flask import Flask, request, jsonify

def take_exam(tasks):
    answers = {}
    
    for task in tasks:
        question = task['question']
        
        if question['type'] == 'choice':
            # pick a random answer
            choice = random.choice(question['choices'])
            answer = choice['id']
            
        elif question['type'] == 'multiple_choice':
            # pick a random number of random choices
            min_choices = question.get('min_choices', 1)
            max_choices = question.get('max_choices', len(question['choices']))
            n_choices = random.randint(min_choices, max_choices)
            random.shuffle(question['choices'])
            answer = [
                choice['id']
                for choice in question['choices'][:n_choices]
            ]
            
            
        elif question['type'] == 'matching':
            # match choices at random
            random.shuffle(question['choices'])
            answer = {
                left['id']: choice['id']
                for left, choice in zip(question['left'], question['choices'])
            }
            
            
        elif question['type'] == 'text':
            if question.get('restriction') == 'word':
                # pick a random word from the text
                words = [word for word in task['text'].split() if len(word) > 1]
                answer = random.choice(words)
            
            else:
                # random text generated with https://fish-text.ru
                answer = (
                    'Для современного мира реализация намеченных плановых заданий позволяет '
                    'выполнить важные задания по разработке новых принципов формирования '
                    'материально-технической и кадровой базы. Господа, реализация намеченных '
                    'плановых заданий играет определяющее значение для модели развития. '
                    'Сложно сказать, почему сделанные на базе интернет-аналитики выводы призывают '
                    'нас к новым свершениям, которые, в свою очередь, должны быть в равной степени '
                    'предоставлены сами себе. Ясность нашей позиции очевидна: базовый вектор '
                    'развития однозначно фиксирует необходимость существующих финансовых и '
                    'административных условий.'
                )
            
        else:
            raise RuntimeError('Unknown question type: {}'.format(question['type']))
        
        answers[task['id']] = answer
        
    return answers


app = Flask(__name__)

@app.route('/ready')
def http_ready():
    return 'OK'

@app.route('/take_exam', methods=['POST'])
def http_take_exam():
    request_data = request.get_json()
    tasks = request_data['tasks']
    answers = take_exam(tasks)
    return jsonify({
        'answers': answers
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
