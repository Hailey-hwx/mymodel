import re
import jieba


class DocumentSegment:
    def __init__(self):
        super().__init__()
        patterne1 = '(.*)[现已|依法](.*)审理[终结|完毕]'
        patterns1 = '原告(.*)[求|请|称]'
        patterns2 = '[依据|依照|根据](.*)判决如下'
        patterne2 = '如不服本判决'
        self.pe1 = re.compile(patterne1)
        self.ps1 = re.compile(patterns1)
        self.ps2 = re.compile(patterns2)
        # self.ps3 = re.compile(patterns3)
        self.pe2 = re.compile(patterne2)

    # 利用jieba分词将长句划分为词供后续抽取
    def sentence_split(self, sentence):
        # jieba.load_userdict('./data/user_dict.txt')
        # jieba.initialize()
        result = jieba.lcut(sentence)
        return result

    def get_part(self, document):
        flag1 = True
        flag2 = False
        flag3 = False
        part1 = []
        part1_tmp = []
        part2 = []
        part3 = []
        for sentence in document:
            if flag1:
                flag1 = True
                flag2 = False
                flag3 = False
                part1_tmp.append(sentence)

            if flag1 and self.pe1.search(sentence):
                flag1 = False
                flag2 = True
                flag3 = False
                continue

            if flag1 and self.ps1.search(sentence):
                flag1 = False
                flag2 = True
                flag3 = False

            if flag2 and self.ps2.search(sentence):
                flag2 = False
                flag3 = True
                flag1 = False

            if flag2:
                flag1 = False
                flag2 = True
                flag3 = False
                part2.append(sentence)

            if flag3:
                flag2 = False
                flag3 = True
                flag1 = False
                result = self.sentence_split(sentence)
                for i, s in enumerate(result):
                    if s != '':
                        part3.append(s)

            if self.pe2.search(sentence):
                break
        
        part1_tmp = part1_tmp[0]
        result = self.sentence_split(part1_tmp)
        for s in result:
            if s != '':
                part1.append(s)

        if len(part1) == 0:
            part1_tmp = document[0]
            result = self.sentence_split(part1_tmp)
            for s in result:
                if s != '':
                    part1.append(s)

        if len(part2) == 0:
            for sentence in document[10:-10]:
                part2.append(sentence)

        if len(part3) == 0:
            for sentence in document[-10::]:
                result = self.sentence_split(sentence)
                num3 = num3+1
                for i, s in enumerate(result):
                    if s != '':
                        part3.append(s)

        return part1, part2, part3
