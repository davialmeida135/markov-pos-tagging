import math
from collections import defaultdict, Counter
import pickle
import os
from processor import PennTreebankProcessor
class HmmPosTagger:
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

            # Para cada palavra na frase, atualiza as contagens
            for i in range(len(words)):
                # Pega tag atual + 2 anteriores
                t_prev2, t_prev1, t_curr = tags[i], tags[i+1], tags[i+2]
                
                # Pega palavra atual
                w_curr = words[i]

                self.transition_counts[(t_prev2, t_prev1)][t_curr] += 1
                self.emission_counts[t_curr][w_curr] += 1
                self.tag_bigram_counts[(t_prev2, t_prev1)] += 1
                self.tag_unigram_counts[t_curr] += 1
                self.tag_set.update([t_prev2, t_prev1, t_curr])

    def test(self, tagged_sentences):
        correct = 0
        total = 0
        predictions = []
        for sent in tagged_sentences:
            #print("Sent:", sent)
            words = [word.lower() for (word, _) in sent]
            # Prepara as palavras para o Viterbi
            #print("Words:", words)
            predicted_tags = self.viterbi(words)
            #print("L3")
            true_tags = [tag for (_, tag) in sent]
            predictions.extend((word, predicted_tag, true_tag) for word, predicted_tag, true_tag in zip(words, predicted_tags, true_tags))
            print(predictions)
            # Conta predições corretas
            correct += sum(1 for p, t in zip(predicted_tags, true_tags) if p == t)
            total += len(true_tags)

        # Send predictions to a csv file
        with open('predictions.csv', 'w') as f:
            f.write('word,predicted_tag,true_tag\n')
            for word, predicted_tag, true_tag in predictions:
                f.write(f'{word},{predicted_tag},{true_tag}\n')

        return correct / total if total > 0 else 0

    def transition_prob(self, t_prev2, t_prev1, t_curr):
        count = self.transition_counts[(t_prev2, t_prev1)][t_curr]
        total = self.tag_bigram_counts[(t_prev2, t_prev1)]
        return count / total if total > 0 else 1e-6

    def emission_prob(self, tag, word):
        count = self.emission_counts[tag][word]
        total = sum(self.emission_counts[tag].values())
        return count / total if total > 0 else 1e-6

    def viterbi(self, sentence):
        if isinstance(sentence, str):
            words = sentence.split()
        else:
            words = sentence

        words = [w.lower() for w in words]
        n = len(words)
        
        # Handle edge cases
        if n < 2:
            return ['NN'] * n
        
        # Convert tag_set to list for faster iteration and indexing
        tag_list = [tag for tag in self.tag_set if tag != '<s>']
        num_tags = len(tag_list)
        
        # Pré calcular as probabilidades de emissão das palavras da frase
        emission_probs = {}
        for i, word in enumerate(words):
            emission_probs[i] = {}
            for tag in tag_list:
                emission_probs[i][tag] = math.log(self.emission_prob(tag, word) + 1e-12)
        
        # Pré calcular as probabilidades de transição
        transition_probs = {}
        for t1 in tag_list:
            for t2 in tag_list:
                for t3 in tag_list:
                    key = (t1, t2, t3)
                    transition_probs[key] = math.log(self.transition_prob(t1, t2, t3) + 1e-12)
        
        # Use numpy-like arrays for faster access (using lists of lists)
        V = [[[-math.inf for _ in range(num_tags)] for _ in range(num_tags)] for _ in range(n)]
        backpointers = [[[None for _ in range(num_tags)] for _ in range(num_tags)] for _ in range(n)]
        
        # Initialize first position
        # Iterando sobre todas as combinações de tags possíveis para as duas primeiras posições
        for i, t1 in enumerate(tag_list):
            for j, t2 in enumerate(tag_list):

                trans1 = transition_probs.get(('<s>', '<s>', t1), -50)
                trans2 = transition_probs.get(('<s>', t1, t2), -50)
                # Emissão para as duas primeiras palavras
                # Probabilidade de t1 emitir w1 e t2 emitir w2
                # Aqui, assumimos que as duas primeiras palavras são '<s>' e '<s>', que não têm emissões válidas
                # Portanto, usamos -50 como valor padrão para evitar problemas de log(0)
                emit1 = emission_probs[0].get(t1, -50)
                emit2 = emission_probs[1].get(t2, -50)
                
                # Preencher a tabela V com a soma das probabilidades de transição e emissão
                # V[1][prev_tag_id][curr_tag_id] representa a probabilidade de transição e emissão para as duas primeiras palavras
                V[1][i][j] = trans1 + trans2 + emit1 + emit2
                backpointers[1][i][j] = ('<s>', '<s>')
        
                
        # Agora calculando para o resto da sentença
        for pos in range(2, n): # Para cada posição da sentença (a partir da 3ª palavra)
            for curr_idx, curr_tag in enumerate(tag_list): # Para cada tag possível na posição atual
                emit_curr = emission_probs[pos][curr_tag] # Probabilidade de emissão da palavra atual dada a tag na iteração
                for prev1_idx, prev1_tag in enumerate(tag_list): # Para cada tag possível na posição anterior
                    best_score = -math.inf
                    best_prev = None
                    
                    for prev2_idx, prev2_tag in enumerate(tag_list): # Para cada tag possível duas posições atrás
                        trans_prob = transition_probs.get((prev2_tag, prev1_tag, curr_tag), -50)

                        # Calcula a probabilidade total para essa combinação de tags
                        # V[pos][prev1_idx][curr_idx] representa a probabilidade de transição e emissão para a posição atual
                        # A soma das probabilidades de transição e emissão para a posição atual
                        # é a soma da probabilidade de transição da tag anterior e da tag duas posições atrás
                        # e a probabilidade de emissão da palavra atual dada a tag atual
                        score = V[pos-1][prev2_idx][prev1_idx] + trans_prob + emit_curr
                        
                        # Para cada par de tags (prev1_tag, curr_tag), verifica se a probabilidade é a melhor até agora
                        if score > best_score:
                            best_score = score
                            best_prev = (prev2_idx, prev1_idx)

                    # V: armazena o melhor score para chegar ao estado (prev1_tag, curr_tag) na posição pos
                    #Testa todas as combinações:
                    # - (DT, NN, NN) + P(NN|commission)
                    # - (DT, NN, VB) + P(VB|commission)  
                    # - (NB, JJ, NN) + P(NN|commission)
                    V[pos][prev1_idx][curr_idx] = best_score
                    backpointers[pos][prev1_idx][curr_idx] = best_prev
        
        # Find the best final state
        best_score = -math.inf
        best_final = None
        
        # Escolhe a melhor combinação de tags para o final da sentença
        for i in range(num_tags):
            for j in range(num_tags):
                if V[n-1][i][j] > best_score:
                    best_score = V[n-1][i][j]
                    best_final = (i, j)
        
        if best_final is None:
            return ['NN'] * n
        
        # Backtrack
        path = []
        curr_state = best_final
        
        # Faz o backtracking para reconstruir o caminho de tags
        # A partir do final da sentença, percorre os backpointers para encontrar as tags
        # Exemplo: se a última posição é (NN, VB), então o penúltimo estado é (DT, NN)
        # e assim por diante até chegar ao início da sentença
        for pos in range(n-1, 0, -1):
            path.append(tag_list[curr_state[1]])
            if pos > 1:
                prev_state = backpointers[pos][curr_state[0]][curr_state[1]]
                curr_state = prev_state
        
        path.append(tag_list[curr_state[0]])
        return list(reversed(path))
    

    def save_model(self, filepath):
        """Salva modelo com pickle."""
        model_data = {
            'transition_counts': dict(self.transition_counts),
            'emission_counts': dict(self.emission_counts),
            'tag_bigram_counts': dict(self.tag_bigram_counts),
            'tag_unigram_counts': dict(self.tag_unigram_counts),
            'tag_set': self.tag_set
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Carrega arquivo pkl com modelo."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.transition_counts = defaultdict(Counter, model_data['transition_counts'])
        self.emission_counts = defaultdict(Counter, model_data['emission_counts'])
        self.tag_bigram_counts = defaultdict(int, model_data['tag_bigram_counts'])
        self.tag_unigram_counts = Counter(model_data['tag_unigram_counts'])
        self.tag_set = model_data['tag_set']
        print(f"Model loaded from {filepath}")
    
    def get_model_stats(self):
        """Printa stats para debug."""
        stats = {
            'num_tags': len(self.tag_set),
            'num_transitions': sum(len(counter) for counter in self.transition_counts.values()),
            'num_emissions': sum(len(counter) for counter in self.emission_counts.values()),
            'total_tag_occurrences': sum(self.tag_unigram_counts.values()),
            'vocabulary_size': len(set(word for counter in self.emission_counts.values() for word in counter.keys()))
        }
        return stats
    
if __name__ == "__main__":
    # Carregar penn treebank
    data_dir = 'data/raw' 
    processor = PennTreebankProcessor(data_dir)
    processor.process() 


    tagger = HmmPosTagger()
    #tagger.train(processor.train)
    #tagger.save_model('models/hmm_pos_tagger.pkl')
    tagger.load_model('models/hmm_pos_tagger.pkl')
    print(tagger.get_model_stats())
    #tagger.test(processor.dev)

    # Test the tagger on a sample sentence
    #test_sentence = "Brabo ojira ahahaha obs muito bom"
    #test_sentence = ['the', 'arizona', 'corporations', 'commission', 'authorized', 'co.','.','%','11.5']
    #test_sentence = ['the', 'arizona', 'corporations', 'commission', 'authorized', 'an', '11.5', '%', 'rate', 'increase', 'at', 'tucson', 'electric', 'power', 'co.', ',', 'substantially', 'lower', 'than', 'recommended', 'last', 'month', 'by', 'a', 'commission', 'hearing', 'officer', 'and', 'barely', 'half', 'the', 'rise', 'sought', 'by', 'the', 'utility', '.']
    #predicted_tags = tagger.viterbi(test_sentence)
    #for word, tag in zip(test_sentence, predicted_tags):
    #   print(f"{word}: {tag}")
