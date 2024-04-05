from flask import Flask, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# 모델과 토크나이저 저장 경로
model_path = "C:/Users/user/Desktop/kogpt2/kogpt2_ft_model.pt"
tokenizer_path = "C:/Users/user/Desktop/kogpt2/kogpt2_ft_proto_tok.pt"

# 모델 불러오기
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# 토크나이저 불러오기
tokenizer = torch.load(tokenizer_path)

@app.route('/question', methods=['POST'])
def question():
    q = request.get_json()
    answer = model.get_answer(q["text"])
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)