import re


class SummarySegment:
    def __init__(self):
        super().__init__()
        patterns1 = '原告(.*)[求|请|称]'
        patterns2 = '原告(.*)因(.*)起诉'
        patterns3 = '(.*)提出诉讼请求'
        patterns4 = '综上(.*)，'
        patterns5 = '据此，'
        patterns6 = '[依据|依照|根据]《'
        patterns7 = '^《(.*)判决'
        self.ps1 = re.compile(patterns1)
        self.ps2 = re.compile(patterns2)
        self.ps3 = re.compile(patterns3)
        self.ps4 = re.compile(patterns4)
        self.ps5 = re.compile(patterns5)
        self.ps6 = re.compile(patterns6)
        self.ps7 = re.compile(patterns7)

    # 将摘要划分为短句
    def sentence_split(self, sentence):
        start = 0
        result = []
        groups = re.finditer('。|，|；', sentence)

        for i in groups:
            end = i.span()[1]
            result.append(sentence[start:end])
            start = end
        # last one
        result.append(sentence[start:])

        return result

    def get_part(self, summary):
        flag1 = True
        flag2 = False
        flag3 = False
        part1 = []
        part2 = []
        part3 = []
        summary = self.sentence_split(summary)
        for sentence in summary:
            if flag1 and self.ps1.search(sentence) or self.ps2.search(sentence) or self.ps3.search(sentence):
                flag1 = False
                flag2 = True
                flag3 = False

            if flag1:
                flag1 = True
                flag2 = False
                flag3 = False
                part1.append(sentence)

            if flag2 and self.ps4.search(sentence) or self.ps5.search(sentence) \
                    or self.ps6.search(sentence) or self.ps7.search(sentence):
                flag1 = False
                flag2 = False
                flag3 = True

            if flag2:
                flag1 = False
                flag2 = True
                flag3 = False
                part2.append(sentence)

            if flag3:
                flag1 = False
                flag2 = False
                flag3 = True
                if sentence != '':
                    part3.append(sentence)

        if len(part1) == 0:
            part1.append(part2[0])

        if len(part2) == 0:
            part2 = part1
            part1 = part1[0]

        if len(part3) == 0:
            part3 = part2[-5::]

        return part1, part2, part3
