from model.model import Model
from model.model import save_model
from model.model import load_model
from model.model import print_checkpoint
import fasttext

fasttext_model = fasttext.load_model("../fasttext_pretrain_model/java-ftskip-dim100-ws5.bin")
train_model = Model(name='first_model', pre_model=fasttext_model)
for i in range(3):
    train_model.train_step(["dao"], ["getUser"], ["getName"])
save_model(train_model)
# print('save model')
# my_model = load_model(pre_model=fasttext_model)
print_checkpoint()
# print('load')
# for i in range(3):
#     my_model.train_step(["dao"], ["getUser"], ["getName"])
# save_model(my_model)
# empty_model = Model(is_reload=True)
# save_model(empty_model)
# new_model = load_model()