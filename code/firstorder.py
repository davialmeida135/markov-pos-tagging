import math
from collections import defaultdict, Counter
import pickle
import os
from processor import PennTreebankProcessor

class HmmPosTaggerFirstOrder:
    def __init__(self):
        self.transition_counts = defaultdict(Counter)  # P(tag_i | tag_{i-1})
        self.emission_counts = defaultdict(Counter)    # P(word_i | tag_i)
        self.tag_unigram_counts = Counter()            # Count of each tag
        self.tag_set = set()

    def train(self, tagged_sentences):
        """Train the first-order HMM model on tagged sentences."""
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
        """Test the model and generate predictions CSV."""
        correct = 0
        total = 0
        predictions = []
        
        for sent in tagged_sentences:
            words = [word.lower() for (word, _) in sent]
            print(words)
            predicted_tags = self.viterbi(words)
            true_tags = [tag for (_, tag) in sent]
            
            predictions.extend((word, predicted_tag, true_tag) 
                             for word, predicted_tag, true_tag in zip(words, predicted_tags, true_tags))
            
            # Count correct predictions
            correct += sum(1 for p, t in zip(predicted_tags, true_tags) if p == t)
            total += len(true_tags)

        # Save predictions to CSV file
        with open('predictions_first_order.csv', 'w') as f:
            f.write('word,predicted_tag,true_tag\n')
            for word, predicted_tag, true_tag in predictions:
                f.write(f'{word},{predicted_tag},{true_tag}\n')

        return correct / total if total > 0 else 0

    def transition_prob(self, t_prev, t_curr):
        """Calculate transition probability P(t_curr | t_prev)."""
        count = self.transition_counts[t_prev][t_curr]
        total = sum(self.transition_counts[t_prev].values())
        return count / total if total > 0 else 1e-6

    def emission_prob(self, tag, word):
        """Calculate emission probability P(word | tag)."""
        count = self.emission_counts[tag][word]
        total = sum(self.emission_counts[tag].values())
        return count / total if total > 0 else 1e-6

    def viterbi(self, sentence):
        """Viterbi algorithm for first-order HMM."""
        if isinstance(sentence, str):
            words = sentence.split()
        else:
            words = sentence

        words = [w.lower() for w in words]
        n = len(words)
        
        # Handle edge cases
        if n == 0:
            return []
        if n == 1:
            return ['NN']
        
        # Convert tag_set to list for indexing (exclude start symbol)
        tag_list = [tag for tag in self.tag_set if tag != '<s>']
        num_tags = len(tag_list)
        
        # Pre-calculate emission probabilities
        emission_probs = {}
        for i, word in enumerate(words):
            emission_probs[i] = {}
            for tag in tag_list:
                emission_probs[i][tag] = math.log(self.emission_prob(tag, word) + 1e-12)
        
        # Pre-calculate transition probabilities
        transition_probs = {}
        for t1 in ['<s>'] + tag_list:  # Include start symbol for first transition
            for t2 in tag_list:
                key = (t1, t2)
                transition_probs[key] = math.log(self.transition_prob(t1, t2) + 1e-12)
        
        # Initialize Viterbi tables
        # V[i][j] = best probability of reaching tag j at position i
        V = [[-math.inf for _ in range(num_tags)] for _ in range(n)]
        # backpointers[i][j] = best previous tag for reaching tag j at position i
        backpointers = [[None for _ in range(num_tags)] for _ in range(n)]
        
        # Initialize first position
        for j, tag in enumerate(tag_list):
            # P(<s> -> tag) * P(word | tag)
            trans_prob = transition_probs.get(('<s>', tag), -50)
            emit_prob = emission_probs[0].get(tag, -50)
            V[0][j] = trans_prob + emit_prob
            backpointers[0][j] = '<s>'
        
        # Fill the rest of the table
        for i in range(1, n):  # For each word position
            for j, curr_tag in enumerate(tag_list):  # For each possible current tag
                emit_prob = emission_probs[i].get(curr_tag, -50)
                
                # Find best previous tag
                best_score = -math.inf
                best_prev = None
                
                for k, prev_tag in enumerate(tag_list):  # For each possible previous tag
                    trans_prob = transition_probs.get((prev_tag, curr_tag), -50)
                    score = V[i-1][k] + trans_prob + emit_prob
                    
                    if score > best_score:
                        best_score = score
                        best_prev = k
                
                V[i][j] = best_score
                backpointers[i][j] = best_prev
        
        # Find the best final state
        best_score = -math.inf
        best_final = None
        
        for j in range(num_tags):
            if V[n-1][j] > best_score:
                best_score = V[n-1][j]
                best_final = j
        
        if best_final is None:
            return ['NN'] * n
        
        # Backtrack to find the best path
        path = []
        curr_state = best_final
        
        for i in range(n-1, -1, -1):
            path.append(tag_list[curr_state])
            if i > 0:
                curr_state = backpointers[i][curr_state]
        
        return list(reversed(path))

    def save_model(self, filepath):
        """Save the trained HMM model to a file."""
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
        print(f"First-order model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained HMM model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.transition_counts = defaultdict(Counter, model_data['transition_counts'])
        self.emission_counts = defaultdict(Counter, model_data['emission_counts'])
        self.tag_unigram_counts = Counter(model_data['tag_unigram_counts'])
        self.tag_set = model_data['tag_set']
        print(f"First-order model loaded from {filepath}")
    
    def get_model_stats(self):
        """Get statistics about the trained model."""
        stats = {
            'num_tags': len(self.tag_set),
            'num_transitions': sum(len(counter) for counter in self.transition_counts.values()),
            'num_emissions': sum(len(counter) for counter in self.emission_counts.values()),
            'total_tag_occurrences': sum(self.tag_unigram_counts.values()),
            'vocabulary_size': len(set(word for counter in self.emission_counts.values() for word in counter.keys()))
        }
        return stats


if __name__ == "__main__":
    # Load the Penn Treebank corpus
    data_dir = 'data/raw'
    processor = PennTreebankProcessor(data_dir)
    processor.process()

    # Train the first-order HMM POS tagger
    tagger = HmmPosTaggerFirstOrder()
    
    # Option 1: Train new model
    print("Treinando modelo de 1ª ordem...")
    tagger.train(processor.train)
    tagger.save_model('models/hmm_pos_tagger_first_order.pkl')
    
    # Option 2: Load existing model
    # tagger.load_model('models/hmm_pos_tagger_first_order.pkl')
    
    print("Estatísticas do modelo:")
    print(tagger.get_model_stats())
    
    # Test the model
    print("\nTestando modelo...")
    accuracy = tagger.test(processor.dev)
    print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Test on sample sentence
    test_sentence = ['the', 'arizona', 'corporations', 'commission', 'authorized', 'an', '11.5', '%', 'rate', 'increase']
    predicted_tags = tagger.viterbi(test_sentence)
    
    print("\nExemplo de predição:")
    for word, tag in zip(test_sentence, predicted_tags):
        print(f"{word}: {tag}")
    
    # Compare with second-order model
    print("\n" + "="*50)