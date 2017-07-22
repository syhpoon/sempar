#!/usr/bin/env python

import sys
import random
import re
import nltk

CJK_RANGE = list(range(0x4e00, 0x9fd6))
QUESTION_MARK_PROB = 0.75
SET = set()
UNK_SET = set()
STOP_WORDS = ["the", "a", "an", "please", "of", "for", "to", "that",
              "could", "do", "does", "is", "are"]


def gen_char():
    return chr(random.choice(CJK_RANGE))


def load_template():
    pairs = []

    f = open("templates.txt")

    data = f.read().rstrip()

    f.close()

    for pair in data.split("\n\n"):
        x, y = [line for line in pair.split("\n") if not line.startswith("#")]

        pairs.append((x, y))

    return pairs


def generate_pair(tpl, words):
    x, y = tpl

    char_sub = {}
    word_sub = {}

    # Generate placeholder values
    for i in range(len(re.findall(r'(@C\d+?@)', x))):
        char_sub[i] = gen_char()

    for i in range(len(re.findall(r'(@W\d+?@)', x))):
        word_sub[i] = random.choice(random.choice(words))

    for k, v in char_sub.items():
        x = x.replace("@C{}@".format(k), v)
        y = y.replace("@C{}@".format(k), v)

    for k, v in word_sub.items():
        x = x.replace("@W{}@".format(k), v)
        y = y.replace("@W{}@".format(k), v)

    qm = "?" if random.randint(1, 100) >= QUESTION_MARK_PROB else ""
    x = x.replace("@?@", qm)

    return x, y


def gen_unk(fq, samples):
    random.seed(None)

    with open(fq) as f:
        lines = f.readlines()

        i = 0

        while i < samples:
            sample = random.choice(lines).rstrip()

            if sample in UNK_SET:
                continue
            else:
                UNK_SET.add(sample)

                yield sample

                i += 1


def tokenize(text):
    tok = nltk.WordPunctTokenizer()

    return " ".join(clear_stop_words([z.lower() for z in tok.tokenize(text)]))


def clear_stop_words(words):
    return [x for x in words if x not in STOP_WORDS]


def read_words(path):
    words = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            split = line.split(" ")

            words.append((split[0], split[1]))

    return words


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: generate.py <number of tpl samples> "
              "<number of unknown samples> <number of classes> "
              "<fq file> <dict file> <out dir>")

        exit(1)

    random.seed(None)

    tplSamples = int(sys.argv[1])
    unkSamples = int(sys.argv[2])
    classes = int(sys.argv[3])
    fq = sys.argv[4]
    dictFile = sys.argv[5]
    out = sys.argv[6]

    d = read_words(dictFile)
    tpl = load_template()

    src = open("{}/sources.txt".format(out), "w")
    dst = open("{}/targets.txt".format(out), "w")

    i = 0
    dups = 0

    while i < tplSamples:
        x, y = generate_pair(random.choice(tpl), d)

        if x in SET:
            dups += 1

            if dups > 10000000:
                break
            else:
                continue
        else:
            SET.add(x)
            i += 1
            dups = 0

        src.write("{}\n".format(tokenize(x)))
        dst.write("{}\n".format(tokenize(y)))

    for unk in gen_unk(fq, min(len(SET) / classes, unkSamples)):
        src.write("{}\n".format(tokenize(unk)))
        dst.write("unknown\n")

    src.close()
    dst.close()
