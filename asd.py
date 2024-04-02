from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

with open('C:/Users/dltkd/OneDrive/바탕 화면/kogpt2/list_a.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 개행 문자('\n') 제거하고 리스트에 저장
a_loaded = [line.strip() for line in lines]

# 기존의 GPT2 토크나이저 및 모델 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# 사용자 지정 토큰 추가를 위한 단어 리스트 정의 (예시로 "사과"와 "바나나"를 명사로 추가)
custom_tokens = a_loaded

# 토크나이저에 사용자 지정 토큰 추가
tokenizer.add_tokens(custom_tokens)

# 토크나이저 저장
tokenizer.save_pretrained("tokenizer_with_custom_tokens")

# 추가된 토큰을 포함하는 새로운 토크나이저로 다시 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_with_custom_tokens")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 모델 토크나이저 업데이트
model.resize_token_embeddings(len(tokenizer))

# 추가된 토큰이 제대로 반영되었는지 확인
print(tokenizer.special_tokens_map)