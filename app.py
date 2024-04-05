from flask import Flask, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# 모델과 토크나이저 저장 경로
model_path = "C:/Users/user/Desktop/kogpt2/kogpt2_ft_proto_model.pt"
tokenizer_path = "C:/Users/user/Desktop/kogpt2/kogpt2_ft_proto_tok.pt"

# 모델 불러오기
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# 토크나이저 불러오기
tokenizer = torch.load(tokenizer_path)

@app.route('/chatbot', methods=['POST'])
def chatbot(min_length=10,
        max_length=20,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        temperature=1.0):
    try:
        data = request.get_json()
        user_input = data['user_input']
        input_ids = tokenizer.encode(user_input, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                do_sample=True,
                top_p=float(top_p),
                top_k=int(top_k),
                min_length=int(min_length),
                max_length=int(max_length),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                temperature=float(temperature)
            )

        generated_text = tokenizer.decode([el.item() for el in output[0]])
        return jsonify({'response': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)