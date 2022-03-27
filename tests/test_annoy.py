from annoy import AnnoyIndex
import fasttext
import time


def get_minute(seconds):
    return seconds / 60


def record_time(message, begin_time_):
    print("{}:{}".format(message, time.time()-begin_time_))


def load_model(path_to_model):
    return fasttext.load_model(path_to_model)


def save_annoy(model_, dim_, type_, tree_num_, save_path_):
    annoy_index = AnnoyIndex(dim_, type_)  # Length of item vector that will be indexed
    # index the word and vector to Annoy
    words_ = model_.get_words()
    print(len(words_))
    # print(words)
    # hash_table = {}
    for i in range(len(words_)):
        word = words_[i]
        vector = model.get_word_vector(word)
        annoy_index.add_item(i, vector)
    annoy_index.build(tree_num_)
    annoy_index.save(save_path_)


def load_annoy(dim_, type_, save_path_):
    annoy_index = AnnoyIndex(dim_, type_)
    annoy_index.load(save_path_)
    return annoy_index


begin_time = time.time()
model = load_model("../fasttext_pretrain_model/java-ftskip-dim100-ws5.bin")
words = model.get_words()
record_time("load model time", begin_time)
begin_time = time.time()

dim = 100
annoy_type = "angular"
tree_num = 100
save_path = "model_cosine.ann"
n = 10
# save_annoy(model, dim, annoy_type, tree_num, save_path)
# record_time("save annoy time", begin_time)
# begin_time = time.time()

new_annoy_index = load_annoy(dim, annoy_type, save_path)
record_time("load annoy time", begin_time)
begin_time = time.time()

test_word = "user"
test_vector = model.get_word_vector(test_word)
result = new_annoy_index.get_nns_by_vector(vector=test_vector, n=n, include_distances=True)
print(result)
for i in range(n):
    print("{}, {}".format(result[1][i], words[result[0][i]]))
record_time("search time", begin_time)
begin_time = time.time()

validate_result = model.get_nearest_neighbors(test_word, n)
print(validate_result)
record_time("validate time", begin_time)
begin_time = time.time()
