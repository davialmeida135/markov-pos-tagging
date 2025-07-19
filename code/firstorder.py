import math
from collections import defaultdict, Counter
import pickle
import os
from processor import PennTreebankProcessor

class HmmPosTaggerFirstOrder:
    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.tag_unigram_counts = Counter()            
        self.tag_set = set()

    def train(self, tagged_sentences):
        """
        Treino do modelo HMM de primeira ordem.
        Recebe uma lista de sentenças tagueadas, onde cada sentença é uma lista de tuplas (palavra, tag).
        Faz a contagem de transições e emissões para construir o modelo.
        """
        for sent in tagged_sentences:
            # Add start symbol
            tags = ['<s>'] + [tag for (_, tag) in sent]
            words = [word.lower() for (word, _) in sent]

            for i in range(len(words)):
                t_prev, t_curr = tags[i], tags[i+1]
                w_curr = words[i]

                # Count transitions: P(t_curr | t_prev)
                self.transition_counts[t_prev][t_curr] += 1
                # Count emissions: P(w_curr | t_curr)
                self.emission_counts[t_curr][w_curr] += 1
                # Count tag occurrences
                self.tag_unigram_counts[t_curr] += 1
                # Update tag set
                self.tag_set.update([t_prev, t_curr])

    def test(self, tagged_sentences):
        """
        Recebe uma lista de sentenças tagueadas para teste.
        Gera um csv com as predições do modelo HMM de primeira ordem.
        Calcula a acurácia do modelo comparando as tags preditas com as tags verdadeiras.
        """
        correct = 0
        total = 0
        predictions = []
        
        for sent in tagged_sentences:
            words = [word.lower() for (word, _) in sent]
            #print(words)
            predicted_tags = self.viterbi(words)
            true_tags = [tag for (_, tag) in sent]
            
            predictions.extend((word, predicted_tag, true_tag) 
                             for word, predicted_tag, true_tag in zip(words, predicted_tags, true_tags))
            
            correct += sum(1 for p, t in zip(predicted_tags, true_tags) if p == t)
            total += len(true_tags)

        with open('predictions_first_order.csv', 'w') as f:
            f.write('word,predicted_tag,true_tag\n')
            for word, predicted_tag, true_tag in predictions:
                f.write(f'{word},{predicted_tag},{true_tag}\n')

        return correct / total if total > 0 else 0

    def transition_prob(self, t_prev, t_curr):
        """Calcula probabilidade de transição P(t_atual | t_anterior)."""
        count = self.transition_counts[t_prev][t_curr]
        total = sum(self.transition_counts[t_prev].values())
        return count / total if total > 0 else 1e-6

    def emission_prob(self, tag, word):
        """Calcula probabilidade de emissão P(palavra | tag)."""
        count = self.emission_counts[tag][word]
        total = sum(self.emission_counts[tag].values())
        return count / total if total > 0 else 1e-6

    def viterbi(self, sentence):
        """Algoritmo de Viterbi para HMM de primeira ordem."""
        if isinstance(sentence, str):
            words = sentence.split()
        else:
            words = sentence

        # Normaliza as palavras para minúsculas
        words = [w.lower() for w in words]
        n = len(words)
        
        # Frases de uma palavra = ['NN']
        if n == 0:
            return []
        if n == 1:
            return ['NN']

        # Cria lista de tags, excluindo o símbolo de início '<s>'
        tag_list = [tag for tag in self.tag_set if tag != '<s>']
        num_tags = len(tag_list)
        
        # Pré calcula as probabilidades de emissão para melhor performance
        # emission_probs[i][tag] = log(P(word_i | tag))
        # Usamos log para evitar underflow numérico
        emission_probs = {}
        for i, word in enumerate(words):
            emission_probs[i] = {}
            for tag in tag_list:
                emission_probs[i][tag] = math.log(self.emission_prob(tag, word) + 1e-12)
        
        # Pré calcula as probabilidades de transição
        # transition_probs[(t1, t2)] = log(P(t2 | t1))
        # Inclui o símbolo de início '<s>' para a primeira transição
        transition_probs = {}
        for t1 in ['<s>'] + tag_list:
            for t2 in tag_list:
                key = (t1, t2)
                transition_probs[key] = math.log(self.transition_prob(t1, t2) + 1e-12)
        
        # V[i][j] = probabilidade máxima de chegar ao estado j na posição i
        V = [[-math.inf for _ in range(num_tags)] for _ in range(n)]
        # backpointers[i][j] = melhor tag anterior para alcançar a tag j na posição i
        backpointers = [[None for _ in range(num_tags)] for _ in range(n)]

        # Inicializa a primeira posição
        for j, tag in enumerate(tag_list):
            # P(<s> -> tag) * P(word | tag)
            trans_prob = transition_probs.get(('<s>', tag), -50)
            emit_prob = emission_probs[0].get(tag, -50)
            V[0][j] = trans_prob + emit_prob
            backpointers[0][j] = '<s>'
        
        # Agora o restante do texto
        for i in range(1, n): # Para cada palavra na frase
            for j, curr_tag in enumerate(tag_list):  # Para cada tag possível
                emit_prob = emission_probs[i].get(curr_tag, -50)

                # Encontra a melhor tag anterior
                best_score = -math.inf
                best_prev = None

                for k, prev_tag in enumerate(tag_list):  # Para cada tag anterior possível
                    trans_prob = transition_probs.get((prev_tag, curr_tag), -50)
                    score = V[i-1][k] + trans_prob + emit_prob
                    
                    if score > best_score:
                        best_score = score
                        best_prev = k
                
                V[i][j] = best_score
                backpointers[i][j] = best_prev
        
        best_score = -math.inf
        best_final = None
        
        # A última posição é onde escolhemos a melhor tag final
        for j in range(num_tags):
            if V[n-1][j] > best_score:
                best_score = V[n-1][j]
                best_final = j
        
        if best_final is None:
            return ['NN'] * n
        
        # Reconstrói o caminho de tags a partir dos backpointers
        # Começa do último estado e vai para trás
        path = []
        curr_state = best_final
        
        for i in range(n-1, -1, -1):
            path.append(tag_list[curr_state])
            if i > 0:
                curr_state = backpointers[i][curr_state]
        
        return list(reversed(path))

    def save_model(self, filepath):
        """Salva o modelo HMM treinado em um arquivo."""
        model_data = {
            'transition_counts': dict(self.transition_counts),
            'emission_counts': dict(self.emission_counts),
            'tag_unigram_counts': dict(self.tag_unigram_counts),
            'tag_set': self.tag_set
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modelo de primeira ordem salvo em {filepath}")

    def load_model(self, filepath):
        """Carrega o modelo HMM de primeira ordem de um arquivo."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.transition_counts = defaultdict(Counter, model_data['transition_counts'])
        self.emission_counts = defaultdict(Counter, model_data['emission_counts'])
        self.tag_unigram_counts = Counter(model_data['tag_unigram_counts'])
        self.tag_set = model_data['tag_set']
    
    def get_model_stats(self):
        """Medidas do modelo"""
        stats = {
            'num_tags': len(self.tag_set),
            'num_transitions': sum(len(counter) for counter in self.transition_counts.values()),
            'num_emissions': sum(len(counter) for counter in self.emission_counts.values()),
            'total_tag_occurrences': sum(self.tag_unigram_counts.values()),
            'vocabulary_size': len(set(word for counter in self.emission_counts.values() for word in counter.keys()))
        }
        return stats


if __name__ == "__main__":
    # Carrega corpus Penn Treebank
    data_dir = 'data/raw'
    processor = PennTreebankProcessor(data_dir)
    processor.process()

    tagger = HmmPosTaggerFirstOrder()

    # Treinar novo modelo
    print("Treinando modelo de 1ª ordem...")
    tagger.train(processor.train)
    tagger.save_model('models/hmm_pos_tagger_first_order.pkl')
    
    # Fazer load do modelo treinado
    # tagger.load_model('models/hmm_pos_tagger_first_order.pkl')
    
    print("Estatísticas do modelo:")
    print(tagger.get_model_stats())
    
    print("\nTestando modelo...")
    accuracy = tagger.test(processor.dev)
    print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    #test_sentence = ['the', 'arizona', 'corporations', 'commission', 'authorized', 'an', '11.5', '%', 'rate', 'increase']
    #predicted_tags = tagger.viterbi(test_sentence)
    
    #print("\nExemplo de predição:")
    #for word, tag in zip(test_sentence, predicted_tags):
    #    print(f"{word}: {tag}")