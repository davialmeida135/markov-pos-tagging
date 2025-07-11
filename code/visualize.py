from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import networkx as nx

def create_confusion_matrix_visualization(self, test_sentences):
    """Create a confusion matrix for POS tag predictions."""
    true_tags = []
    predicted_tags = []
    
    for sent in test_sentences:
        words = [word.lower() for (word, _) in sent]
        pred_tags = self.viterbi(words)
        true_tags.extend([tag for (_, tag) in sent])
        predicted_tags.extend(pred_tags)
    
    # Get most common tags for better visualization
    common_tags = pd.Series(true_tags).value_counts().head(15).index.tolist()
    
    # Filter data to common tags only
    filtered_true = [tag if tag in common_tags else 'OTHER' for tag in true_tags]
    filtered_pred = [tag if tag in common_tags else 'OTHER' for tag in predicted_tags]
    
    # Create confusion matrix
    cm = confusion_matrix(filtered_true, filtered_pred, labels=common_tags + ['OTHER'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=common_tags + ['OTHER'], 
                yticklabels=common_tags + ['OTHER'])
    plt.title('Matriz de Confusão - POS Tagging')
    plt.xlabel('Tags Preditas')
    plt.ylabel('Tags Verdadeiras')
    plt.tight_layout()
    plt.show()


def visualize_transition_graph(self, top_n=20):
    """Visualize the most common POS tag transitions."""
    # Get top transitions
    all_transitions = []
    for (prev2, prev1), counter in self.transition_counts.items():
        for curr, count in counter.items():
            if prev2 != '<s>' and prev1 != '<s>' and curr != '<s>':
                all_transitions.append(((prev1, curr), count))
    
    # Sort and get top transitions
    top_transitions = sorted(all_transitions, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create graph
    G = nx.DiGraph()
    for (prev_tag, curr_tag), count in top_transitions:
        G.add_edge(prev_tag, curr_tag, weight=count)
    
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    
    # Draw edges with varying thickness
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    
    nx.draw_networkx_edges(G, pos, width=[w/max_weight * 5 for w in weights],
                          alpha=0.6, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title('Transições Mais Comuns entre Tags POS')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_performance_by_tag(self, test_sentences):
    """Analyze model performance for each POS tag."""
    tag_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'precision': 0, 'recall': 0})
    
    for sent in test_sentences:
        words = [word.lower() for (word, _) in sent]
        pred_tags = self.viterbi(words)
        true_tags = [tag for (_, tag) in sent]
        
        for pred, true in zip(pred_tags, true_tags):
            tag_stats[true]['total'] += 1
            if pred == true:
                tag_stats[true]['correct'] += 1
    
    # Calculate accuracy for each tag
    for tag in tag_stats:
        if tag_stats[tag]['total'] > 0:
            tag_stats[tag]['accuracy'] = tag_stats[tag]['correct'] / tag_stats[tag]['total']
    
    # Create visualization
    tags = list(tag_stats.keys())
    accuracies = [tag_stats[tag]['accuracy'] for tag in tags]
    counts = [tag_stats[tag]['total'] for tag in tags]
    
    # Sort by frequency
    sorted_data = sorted(zip(tags, accuracies, counts), key=lambda x: x[2], reverse=True)
    tags, accuracies, counts = zip(*sorted_data[:20])  # Top 20 most frequent tags
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Accuracy plot
    bars1 = ax1.bar(range(len(tags)), accuracies, color='skyblue')
    ax1.set_xlabel('Tags POS')
    ax1.set_ylabel('Acurácia')
    ax1.set_title('Acurácia por Tag POS')
    ax1.set_xticks(range(len(tags)))
    ax1.set_xticklabels(tags, rotation=45)
    
    # Frequency plot
    bars2 = ax2.bar(range(len(tags)), counts, color='lightcoral')
    ax2.set_xlabel('Tags POS')
    ax2.set_ylabel('Frequência')
    ax2.set_title('Frequência das Tags no Conjunto de Teste')
    ax2.set_xticks(range(len(tags)))
    ax2.set_xticklabels(tags, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return tag_stats

def visualize_viterbi_path(self, sentence, max_tags=8):
    """Visualize the Viterbi algorithm path for a sentence."""
    if isinstance(sentence, str):
        words = sentence.split()
    else:
        words = sentence
    
    words = [w.lower() for w in words]
    predicted_tags = self.viterbi(words)
    
    # Get most likely tags for each word for visualization
    tag_list = [tag for tag in self.tag_set if tag != '<s>'][:max_tags]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create a grid showing words and possible tags
    for i, word in enumerate(words):
        for j, tag in enumerate(tag_list):
            # Color based on whether this is the predicted tag
            color = 'red' if tag == predicted_tags[i] else 'lightblue'
            rect = plt.Rectangle((i, j), 1, 1, facecolor=color, alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            
            # Add probability text (simplified)
            prob = self.emission_prob(tag, word)
            ax.text(i+0.5, j+0.5, f'{prob:.3f}', ha='center', va='center', fontsize=8)
    
    # Draw the path
    for i in range(len(words)):
        tag_idx = tag_list.index(predicted_tags[i]) if predicted_tags[i] in tag_list else 0
        if i > 0:
            prev_tag_idx = tag_list.index(predicted_tags[i-1]) if predicted_tags[i-1] in tag_list else 0
            ax.arrow(i-0.5, prev_tag_idx+0.5, 0.8, tag_idx-prev_tag_idx, 
                    head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    
    ax.set_xlim(0, len(words))
    ax.set_ylim(0, len(tag_list))
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels([f'{word}\n{tag}' for word, tag in zip(words, predicted_tags)], rotation=45)
    ax.set_yticks(range(len(tag_list)))
    ax.set_yticklabels(tag_list)
    ax.set_xlabel('Palavras')
    ax.set_ylabel('Tags POS')
    ax.set_title('Caminho do Algoritmo Viterbi')
    
    plt.tight_layout()
    plt.show()

def create_model_statistics_dashboard(self):
    """Create a dashboard with model statistics."""
    stats = self.get_model_stats()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Tag frequency distribution
    tag_counts = list(self.tag_unigram_counts.values())
    ax1.hist(tag_counts, bins=30, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Frequência')
    ax1.set_ylabel('Número de Tags')
    ax1.set_title('Distribuição de Frequência das Tags')
    
    # Top 15 most common tags
    top_tags = self.tag_unigram_counts.most_common(15)
    tags, counts = zip(*top_tags)
    ax2.bar(range(len(tags)), counts, color='lightcoral')
    ax2.set_xlabel('Tags')
    ax2.set_ylabel('Frequência')
    ax2.set_title('15 Tags Mais Comuns')
    ax2.set_xticks(range(len(tags)))
    ax2.set_xticklabels(tags, rotation=45)
    
    # Model complexity
    complexity_data = [
        stats['num_tags'],
        stats['num_transitions'],
        stats['num_emissions'],
        stats['vocabulary_size']
    ]
    complexity_labels = ['Tags', 'Transições', 'Emissões', 'Vocabulário']
    
    ax3.bar(complexity_labels, complexity_data, color=['gold', 'lightgreen', 'lightblue', 'pink'])
    ax3.set_ylabel('Quantidade')
    ax3.set_title('Complexidade do Modelo')
    ax3.set_yscale('log')
    
    # Summary text
    ax4.text(0.1, 0.8, f"Número de Tags: {stats['num_tags']}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f"Transições: {stats['num_transitions']}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f"Emissões: {stats['num_emissions']}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f"Vocabulário: {stats['vocabulary_size']}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f"Total de Tags: {stats['total_tag_occurrences']}", fontsize=12, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Resumo do Modelo')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()