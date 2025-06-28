import math
from collections import defaultdict, Counter
import nltk

nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown

class HMM_POS_Tagger:
    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.tag_bigram_counts = defaultdict(int)
        self.tag_unigram_counts = Counter()
        self.tag_set = set()

    def train(self, tagged_sentences):
        for sent in tagged_sentences:
            tags = ['<s>', '<s>'] + [tag for (_, tag) in sent]
            words = [word.lower() for (word, _) in sent]

            for i in range(len(words)):
                t_prev2, t_prev1, t_curr = tags[i], tags[i+1], tags[i+2]
                w_curr = words[i]

                self.transition_counts[(t_prev2, t_prev1)][t_curr] += 1
                self.emission_counts[t_curr][w_curr] += 1
                self.tag_bigram_counts[(t_prev2, t_prev1)] += 1
                self.tag_unigram_counts[t_curr] += 1
                self.tag_set.update([t_prev2, t_prev1, t_curr])

    def transition_prob(self, t_prev2, t_prev1, t_curr):
        count = self.transition_counts[(t_prev2, t_prev1)][t_curr]
        total = self.tag_bigram_counts[(t_prev2, t_prev1)]
        return count / total if total > 0 else 1e-6

    def emission_prob(self, tag, word):
        count = self.emission_counts[tag][word]
        total = sum(self.emission_counts[tag].values())
        return count / total if total > 0 else 1e-6

    def viterbi(self, sentence):
        words = [w.lower() for w in sentence]
        n = len(words)
        V = defaultdict(lambda: defaultdict(lambda: (-math.inf, None)))

        for t1 in self.tag_set:
            for t2 in self.tag_set:
                p = math.log(self.transition_prob('<s>', '<s>', t1) + 1e-12) + \
                    math.log(self.transition_prob('<s>', t1, t2) + 1e-12) + \
                    math.log(self.emission_prob(t1, words[0]) + 1e-12) + \
                    math.log(self.emission_prob(t2, words[1]) + 1e-12)
                V[1][(t1, t2)] = (p, ('<s>', t1))

        for i in range(2, n):
            for t_prev1 in self.tag_set:
                for t_prev2 in self.tag_set:
                    for t_curr in self.tag_set:
                        prev_score, _ = V[i-1][(t_prev2, t_prev1)]
                        score = prev_score + math.log(self.transition_prob(t_prev2, t_prev1, t_curr) + 1e-12) + \
                                math.log(self.emission_prob(t_curr, words[i]) + 1e-12)
                        if score > V[i][(t_prev1, t_curr)][0]:
                            V[i][(t_prev1, t_curr)] = (score, (t_prev2, t_prev1))

        last_pos = n - 1
        best_tags = []
        best_score = -math.inf
        last_tags = None

        for (t1, t2), (score, _) in V[last_pos].items():
            if score > best_score:
                best_score = score
                last_tags = (t1, t2)

        best_tags = [last_tags[1], last_tags[0]]
        for i in range(last_pos, 1, -1):
            _, prev = V[i][(best_tags[-1], best_tags[-2])]
            best_tags.append(prev[0])

        return list(reversed(best_tags))
    
if __name__ == "__main__":
    # Load the Brown corpus
    tagged_sentences = brown.tagged_sents(categories='news', tagset='universal')
    print(tagged_sentences[0])  # Print first 5 tagged sentences for verification
    # Train the HMM POS tagger
    tagger = HMM_POS_Tagger()
    #tagger.train(tagged_sentences)

    # Test the tagger on a sample sentence
    #test_sentence = "The quick brown fox jumps over the lazy dog"
    #predicted_tags = tagger.viterbi(test_sentence)

    # Print the results
    #for word, tag in zip(test_sentence.split(), predicted_tags):
    #    print(f"{word}: {tag}")
