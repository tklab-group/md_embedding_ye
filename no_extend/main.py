import sys
sys.path.append('../')
import torch
from data.data_loader import DataLoader
from trainer import Trainer
from no_extend.cbow import EmbeddingModel
import time
from no_extend.eval import Evaluation

class Main:
    def __init__(self, hidden_size=100, batch_size=32, max_epoch=10, git_name='tomcat', validate_data_num=1000):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.git_name = git_name
        self.validate_data_num = validate_data_num
        self.data_loader = None
        self.model = None

    def train(self, is_load_model=False, load_path=None, device=None, is_load_from_pkl=False):
        data_loader = DataLoader(self.git_name, self.validate_data_num, is_load_from_pkl)
        vocab_size = data_loader.vocab.corpus_length
        print('vocab_size', vocab_size)
        train = data_loader.train_data
        validate = data_loader.validate_data
        contexts = train['contexts']
        target = train['target']

        model = EmbeddingModel(vocab_size, self.hidden_size, device).to(device)
        if is_load_model:
            model.load_state_dict(torch.load(load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        self.data_loader = data_loader
        self.model = model

        trainer = Trainer(model=model, optimizer=optimizer, device=device)
        trainer.fit(contexts=contexts, target=target)
        # save model
        save_path = './model_params/no_extend_cbow_params_' + git_name + '_' + str(time.time()) + '.pth'
        torch.save(model.state_dict(), save_path)

        trainer.plot()

    def eval(self, k):
        evaluation = Evaluation(self.model, self.data_loader)
        Recall, MRR, F_MRR = evaluation.validate(k)
        print('k Recall, MRR, F_MRR', k, Recall, MRR, F_MRR)
        return Recall, MRR, F_MRR


is_can_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_can_cuda else "cpu")
print('is_can_cuda', is_can_cuda)

# git_name = 'tomcat'
git_name = 'LCExtractor'
validate_data_num = 19
main = Main(git_name=git_name, validate_data_num=validate_data_num)
# load_path = './model_params/no_extend_cbow_params_' + git_name + '_' + str(time.time()) + '.pth'
load_path = './model_params/no_extend_cbow_params_tomcat_1631109203.140158.pth'
main.train(is_load_model=False, load_path=load_path, device=device, is_load_from_pkl=False)

# load model, no to train
# data_loader = DataLoader(main.git_name, main.validate_data_num, False)
# vocab_size = data_loader.vocab.corpus_length
# print('model Loading...')
# model = EmbeddingModel(vocab_size, main.hidden_size, device).to(device)
# model.load_state_dict(torch.load(load_path))
# print('model Loading end.')
# main.data_loader = data_loader
# main.model = model

main.eval(k=20)














