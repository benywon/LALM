from QGevaluation.bleu.bleu import Bleu
from QGevaluation.meteor.meteor import Meteor
from QGevaluation.rouge.rouge import Rouge


class COCOEvalCap:
    def __init__(self, reference, prediction,only_bleu=False):
        self.only_bleu = only_bleu
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = reference
        self.cocoRes = prediction

    def evaluate(self):
        # imgIds = self.coco.getImgIds()
        gts = {}
        for i, one in enumerate(self.coco):
            gts[i] = one
        res = {}
        for i, one in enumerate(self.cocoRes):
            res[i] = [one]

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Cider(), "CIDEr")
        ]
        if not self.only_bleu:
            scorers.extend(
            [(Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")])
        # =================================================
        # Compute scores
        # =================================================
        output = []
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    print("%s: %0.5f" % (m, sc))
                    output.append("%s: %0.5f" % (m, sc))
            else:
                self.setEval(score, method)
                print("%s: %0.5f" % (method, score))
                output.append("%s: %0.5f" % (method, score))
        return output

    def setEval(self, score, method):
        self.eval[method] = score


if __name__ == '__main__':
    for ep in [29]:
        reference = []
        prediction = []
        with open('/search/odin/bingning/program/dis_torch/QuestionGeneration/prediction.1024.12.{}.tmp.txt'.format(ep),
                  encoding='utf-8') as f:
            cc = f.read()
            c = cc.split('************************************************************')
            print(len(c))
            for one in c:
                tt = one.split('\n')
                if len(tt) != 5:
                    # print(tt)
                    continue
                prediction.append(' '.join(list(tt[2])))
                reference.append([' '.join(list(x)) for x in tt[3].split('\t')])

        eval = COCOEvalCap(reference=reference, prediction=prediction)
        eval.evaluate()
