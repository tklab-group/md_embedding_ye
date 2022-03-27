import sys
sys.path.append('../')
import numpy as np
import copy


# https://github.com/tklab-group/bthesis2020-kitabayashi/blob/master/thesis/bthesis.pdf
class Metric:
    def __init__(self):
        # feedback(i) => self.feedback[i] / self.com[i]
        # feedbackはeval_with_commitした後、コミット毎の推薦回数を保存
        self.feedback = {}
        # feedback_no_newはeval_with_commitした後、新規ファイルを考慮したコミット毎の推薦回数を保存
        self.feedback_no_new = {}
        # recallはeval_with_commitした後、コミット毎の推薦して命中した回数を保存
        self.recall = {}
        # 各コミットのRankデータ集合を保存
        self.rank_i = {}
        # 各コミットのcommit size
        self.com = {}
        # 新規ファイルを除外した各コミットのcommit size
        self.com_no_new = {}

        # 全体の推薦して命中した回数
        self.total_hit_count = 0

    # commit_th コミットの順番、0から始まる
    # rank_i_c cを欠損した時、推薦した結果の中にcのRank
    # rec_i_c_len cを欠損し時、推薦した結果の数
    # is_target_in_train targetは訓練データに属するのか
    def eval_with_commit(self, commit_th, rank_i_c, rec_i_c_len, is_target_in_train):
        # print('(commit_th, rank_i_c, rec_i_c_len)', commit_th, rank_i_c, rec_i_c_len, is_target_in_train)
        # com_no_newを記録
        if is_target_in_train:
            if commit_th in self.com_no_new:
                self.com_no_new[commit_th] += 1
            else:
                self.com_no_new[commit_th] = 1
        else:
            if commit_th not in self.com_no_new:
                self.com_no_new[commit_th] = 0
        # total_hit_countとrecallを記録
        if commit_th in self.com:
            self.com[commit_th] += 1
            self.rank_i[commit_th].append(rank_i_c)
            # 推薦した場合、Feedbackに入れる
            if rec_i_c_len > 0:
                self.feedback[commit_th] += 1
                if is_target_in_train:
                    self.feedback_no_new[commit_th] += 1
            # 命中した場合、recallに保存する
            if rank_i_c > 0:
                self.total_hit_count += 1
                self.recall[commit_th] += 1
        else:
            self.com[commit_th] = 1
            self.rank_i[commit_th] = [rank_i_c]
            # 推薦した場合、Feedbackに入れる
            if rec_i_c_len > 0:
                self.feedback[commit_th] = 1
                if is_target_in_train:
                    self.feedback_no_new[commit_th] = 1
                else:
                    self.feedback_no_new[commit_th] = 0
            else:
                self.feedback[commit_th] = 0
                self.feedback_no_new[commit_th] = 0
            # 命中した場合、recallに保存する
            if rank_i_c > 0:
                self.total_hit_count += 1
                self.recall[commit_th] = 1
            else:
                self.recall[commit_th] = 0

    # is_consider_new_file True 新規ファイルを考慮する、つまりRecallを計算する時新規ファイルを除外する
    def summary(self, is_consider_new_file=True):
        # print('total_hit_count', self.total_hit_count)
        # meta dataをコピー
        recall_copy = copy.deepcopy(self.recall)
        feedback_copy = copy.deepcopy(self.feedback)
        rank_i_copy = copy.deepcopy(self.rank_i)

        # 全体の推薦した回数
        total_recommend_count = 0
        # 全体のコミットの長さ
        commit_len = len(self.com)
        commit_len_no_new = 0
        # Feedbackがゼロではないコミットの長さ
        commit_len_feedback = 0

        # 求めたい結果
        # 正解を推薦できる割合
        macro_recall = 0
        micro_recall = 0
        # 正解を推薦した順位の逆数の平均
        mrr = 0
        # RecallとMRRの調和平均
        f_mrr = 0

        total_mrr = 0
        total_recall = 0

        for i in range(commit_len):
            if i in self.com_no_new and self.com_no_new[i] > 0:
                commit_len_no_new += 1

            # feedback(i) => self.feedback[i] / self.com[i]
            # mrrを計算する
            if is_consider_new_file:
                if self.com_no_new[i] > 0:
                    feedback_copy[i] /= self.com_no_new[i]
                else:
                    feedback_copy[i] = 0
            else:
                if i in self.com and self.com[i] > 0:
                    feedback_copy[i] /= self.com[i]
                else:
                    feedback_copy[i] = 0
            mrr_i_temp = 0
            if feedback_copy[i] != 0:
                commit_len_feedback += 1
                rank_array = rank_i_copy[i]
                rank_sum = 0
                for j in range(len(rank_array)):
                    if rank_array[j] != 0:
                        rank_sum += (1 / rank_array[j])
                if is_consider_new_file:
                    if self.com_no_new[i] > 0:
                        mrr_i_temp = (1 / (feedback_copy[i] * self.com_no_new[i])) * rank_sum
                    else:
                        mrr_i_temp = 0
                else:
                    mrr_i_temp = (1 / (feedback_copy[i] * self.com[i])) * rank_sum
            else:
                mrr_i_temp = 0
            total_mrr += mrr_i_temp

            if is_consider_new_file:
                if i in self.com_no_new and self.com_no_new[i] > 0:
                    recall_copy[i] /= self.com_no_new[i]
                else:
                    recall_copy[i] = 0
            else:
                if i in self.com and self.com[i] > 0:
                    recall_copy[i] /= self.com[i]
                else:
                    recall_copy[i] = 0
            total_recall += recall_copy[i]
            # print('pre recall', recall_copy[i])

            # 推薦した後、かつ新規ファイルを考慮
            if is_consider_new_file:
                # total_recommend_count += self.feedback_no_new[i]
                if i in self.com_no_new and self.com_no_new[i] > 0:
                    total_recommend_count += self.com_no_new[i]
                # print('feedback_no_new', self.feedback_no_new[i])
            else:
                # total_recommend_count += self.feedback[i]
                if i in self.com and self.com[i] > 0:
                    total_recommend_count += self.com[i]

        # if is_consider_new_file:
        #     if commit_len_feedback > 0:
        #         macro_recall = total_recall / commit_len_feedback
        #     else:
        #         macro_recall = 0
        # else:
        #     macro_recall = total_recall / commit_len
        # macro recallを計算
        if is_consider_new_file:
            if commit_len_no_new > 0:
                macro_recall = total_recall / commit_len_no_new
            else:
                macro_recall = 0
        else:
            if commit_len > 0:
                macro_recall = total_recall / commit_len
            else:
                macro_recall = 0

        if commit_len > 0:
            # mrr = total_mrr / commit_len_feedback
            mrr = total_mrr / commit_len
        else:
            mrr = 0

        if (mrr + macro_recall) > 0:
            f_mrr = 2 * (mrr * macro_recall) / (mrr + macro_recall)
        else:
            f_mrr = 0

        if total_recommend_count > 0:
            micro_recall = self.total_hit_count / total_recommend_count

        # print('total_recommend_count', total_recommend_count)
        # print('total_recall', total_recall, commit_len_feedback, commit_len)

        # return micro_recall * 100, macro_recall * 100, mrr * 100, f_mrr * 100
        # return micro_recall * 100, macro_recall * 100
        return micro_recall, macro_recall
