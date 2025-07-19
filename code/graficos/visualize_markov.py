import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import pickle
from itertools import combinations

def load_tagger_model(model_path):
    """Load the trained HMM model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def get_most_common_tags(tag_counts, top_n=15):
    """Get most common tags from either Counter or dict."""
    if isinstance(tag_counts, Counter):
        return [tag for tag, _ in tag_counts.most_common(top_n) if tag != '<s>']
    else:
        # Convert dict to Counter if needed
        counter = Counter(tag_counts)
        return [tag for tag, _ in counter.most_common(top_n) if tag != '<s>']

def visualize_transition_heatmap(model_data, top_n=15, order='second'):
    """Create a heatmap of transition probabilities."""
    if order == 'second':
        # For second-order model, show bigram -> tag transitions
        transition_counts = model_data['transition_counts']
        
        # Get most common tags
        tag_counts = model_data['tag_unigram_counts']
        common_tags = get_most_common_tags(tag_counts, top_n)
        
        # Create transition probability matrix
        matrix = np.zeros((len(common_tags), len(common_tags)))
        
        for i, tag1 in enumerate(common_tags):
            for j, tag2 in enumerate(common_tags):
                # Sum over all possible previous tags
                total_count = 0
                for prev_tag in common_tags + ['<s>']:
                    if (prev_tag, tag1) in transition_counts:
                        total_count += transition_counts[(prev_tag, tag1)].get(tag2, 0)
                
                # Normalize
                tag1_total = tag_counts.get(tag1, 0)
                if tag1_total > 0:
                    matrix[i][j] = total_count / tag1_total
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=common_tags, yticklabels=common_tags)
        plt.title('Matriz de Transição - HMM 2ª Ordem\n(Probabilidade de transição entre tags)')
        plt.xlabel('Tag Atual')
        plt.ylabel('Tag Anterior')
        
    else:  # first order
        # For first-order model
        transition_counts = model_data['transition_counts']
        tag_counts = model_data['tag_unigram_counts']
        common_tags = get_most_common_tags(tag_counts, top_n)
        
        # Create transition probability matrix
        matrix = np.zeros((len(common_tags), len(common_tags)))
        
        for i, tag1 in enumerate(common_tags):
            for j, tag2 in enumerate(common_tags):
                count = transition_counts.get(tag1, {}).get(tag2, 0)
                total = sum(transition_counts.get(tag1, {}).values()) if tag1 in transition_counts else 1
                matrix[i][j] = count / total if total > 0 else 0
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=common_tags, yticklabels=common_tags)
        plt.title('Matriz de Transição - HMM 1ª Ordem\n(Probabilidade de transição entre tags)')
        plt.xlabel('Tag Atual')
        plt.ylabel('Tag Anterior')
    
    plt.tight_layout()
    filename = f'transition_heatmap_{order}_order.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_markov_graph(model_data, top_n=10, min_prob=0.05, order='second'):
    """Create a network graph of the Markov chain."""
    transition_counts = model_data['transition_counts']
    tag_counts = model_data['tag_unigram_counts']
    
    # Get most common tags
    common_tags = get_most_common_tags(tag_counts, top_n)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for tag in common_tags:
        G.add_node(tag)
    
    if order == 'second':
        # For second-order, we need to simplify to show main transitions
        for (prev1, prev2), counter in transition_counts.items():
            if prev1 in common_tags and prev2 in common_tags:
                counter_dict = dict(counter) if hasattr(counter, 'items') else counter
                for curr, count in counter_dict.items():
                    if curr in common_tags:
                        # Calculate probability
                        total = sum(counter_dict.values())
                        prob = count / total if total > 0 else 0
                        
                        if prob >= min_prob:
                            # Use weighted edge for bigram transitions
                            if G.has_edge(prev2, curr):
                                G[prev2][curr]['weight'] += prob
                            else:
                                G.add_edge(prev2, curr, weight=prob)
    else:
        # For first-order
        for prev_tag, counter in transition_counts.items():
            if prev_tag in common_tags:
                counter_dict = dict(counter) if hasattr(counter, 'items') else counter
                total = sum(counter_dict.values()) if counter_dict else 1
                for curr_tag, count in counter_dict.items():
                    if curr_tag in common_tags:
                        prob = count / total if total > 0 else 0
                        if prob >= min_prob:
                            G.add_edge(prev_tag, curr_tag, weight=prob)
    
    # Create layout
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Draw nodes
    node_sizes = [tag_counts.get(tag, 100) / 10 for tag in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=node_sizes, alpha=0.8)
    
    # Draw edges with varying thickness
    edges = G.edges()
    if edges:
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(G, pos, width=[w/max_weight * 5 for w in weights],
                              alpha=0.6, edge_color='gray', arrows=True,
                              arrowsize=20, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add edge labels for strong connections
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d['weight'] >= min_prob * 2:  # Only show strong connections
            edge_labels[(u, v)] = f"{d['weight']:.2f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title(f'Cadeia de Markov - {order.title()} Ordem\n(Transições com probabilidade ≥ {min_prob})')
    plt.axis('off')
    plt.tight_layout()
    
    filename = f'markov_chain_{order}_order.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_emission_probabilities(model_data, tag='NN', top_words=15):
    """Visualize emission probabilities for a specific tag."""
    emission_counts = model_data['emission_counts']
    
    if tag not in emission_counts:
        print(f"Tag '{tag}' não encontrada no modelo.")
        return
    
    # Get most common words for this tag
    tag_emissions = emission_counts[tag]
    if hasattr(tag_emissions, 'most_common'):
        word_counts = tag_emissions.most_common(top_words)
        total_count = sum(tag_emissions.values())
    else:
        # Convert dict to Counter if needed
        counter = Counter(tag_emissions)
        word_counts = counter.most_common(top_words)
        total_count = sum(tag_emissions.values())
    
    words = [word for word, _ in word_counts]
    probs = [count / total_count for _, count in word_counts]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(words)), probs, color='lightgreen', alpha=0.7)
    plt.xlabel('Palavras')
    plt.ylabel('Probabilidade de Emissão')
    plt.title(f'Probabilidades de Emissão para a Tag "{tag}"')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    
    # Add probability values on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(probs)*0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = f'emission_probabilities_{tag}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def compare_transition_matrices(first_order_model, second_order_model, top_n=10):
    """Compare transition matrices between first and second order models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get common tags for both models
    tags1 = set(first_order_model['tag_unigram_counts'].keys())
    tags2 = set(second_order_model['tag_unigram_counts'].keys())
    common_tags_set = tags1.intersection(tags2)
    common_tags_set.discard('<s>')
    
    # Get most frequent common tags
    combined_counts = Counter()
    for tag in common_tags_set:
        count1 = first_order_model['tag_unigram_counts'].get(tag, 0)
        count2 = second_order_model['tag_unigram_counts'].get(tag, 0)
        combined_counts[tag] = count1 + count2
    
    common_tags = [tag for tag, _ in combined_counts.most_common(top_n)]
    
    # First order matrix
    matrix1 = np.zeros((len(common_tags), len(common_tags)))
    transition_counts1 = first_order_model['transition_counts']
    
    for i, tag1 in enumerate(common_tags):
        for j, tag2 in enumerate(common_tags):
            count = transition_counts1.get(tag1, {}).get(tag2, 0)
            total = sum(transition_counts1.get(tag1, {}).values()) if tag1 in transition_counts1 else 1
            matrix1[i][j] = count / total if total > 0 else 0
    
    # Second order matrix (simplified)
    matrix2 = np.zeros((len(common_tags), len(common_tags)))
    transition_counts2 = second_order_model['transition_counts']
    tag_counts2 = second_order_model['tag_unigram_counts']
    
    for i, tag1 in enumerate(common_tags):
        for j, tag2 in enumerate(common_tags):
            total_count = 0
            for prev_tag in common_tags + ['<s>']:
                if (prev_tag, tag1) in transition_counts2:
                    counter = transition_counts2[(prev_tag, tag1)]
                    total_count += counter.get(tag2, 0) if hasattr(counter, 'get') else counter[tag2] if tag2 in counter else 0
            
            tag1_total = tag_counts2.get(tag1, 0)
            if tag1_total > 0:
                matrix2[i][j] = total_count / tag1_total
    
    # Plot first order
    sns.heatmap(matrix1, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=common_tags, yticklabels=common_tags, ax=ax1)
    ax1.set_title('HMM 1ª Ordem')
    ax1.set_xlabel('Tag Atual')
    ax1.set_ylabel('Tag Anterior')
    
    # Plot second order
    sns.heatmap(matrix2, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=common_tags, yticklabels=common_tags, ax=ax2)
    ax2.set_title('HMM 2ª Ordem (Simplificado)')
    ax2.set_xlabel('Tag Atual')
    ax2.set_ylabel('Tag Anterior')
    
    plt.suptitle('Comparação: Matrizes de Transição 1ª vs 2ª Ordem', fontsize=16)
    plt.tight_layout()
    plt.savefig('transition_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sentence_tagging_process(model_data, sentence, model_type='second'):
    """Visualize the tagging process for a specific sentence."""
    if model_type == 'second':
        from code.secondorder import HmmPosTagger
        tagger = HmmPosTagger()
    else:
        from firstorder import HmmPosTaggerFirstOrder
        tagger = HmmPosTaggerFirstOrder()
    
    # Load model data into tagger
    tagger.transition_counts = defaultdict(Counter, model_data['transition_counts'])
    tagger.emission_counts = defaultdict(Counter, model_data['emission_counts'])
    tagger.tag_unigram_counts = Counter(model_data['tag_unigram_counts'])
    tagger.tag_set = model_data['tag_set']
    
    if 'tag_bigram_counts' in model_data:
        tagger.tag_bigram_counts = defaultdict(int, model_data['tag_bigram_counts'])
    
    # Get predictions
    words = sentence.split() if isinstance(sentence, str) else sentence
    predicted_tags = tagger.viterbi(words)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot words and their tags
    x_positions = range(len(words))
    
    # Create bars for words
    bars = ax.bar(x_positions, [1] * len(words), color='lightblue', alpha=0.7)
    
    # Add word labels
    for i, (word, tag) in enumerate(zip(words, predicted_tags)):
        ax.text(i, 0.5, word, ha='center', va='center', fontweight='bold', fontsize=10)
        ax.text(i, 1.1, tag, ha='center', va='bottom', fontweight='bold', 
                fontsize=12, color='red')
    
    # Add transition arrows
    for i in range(len(words) - 1):
        ax.annotate('', xy=(i+1-0.3, 0.8), xytext=(i+0.3, 0.8),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    ax.set_xlim(-0.5, len(words) - 0.5)
    ax.set_ylim(0, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Processo de Etiquetagem - {model_type.title()} Ordem\n' + 
                 f'Sentença: "{" ".join(words)}"', fontsize=14)
    
    # Add legend
    ax.text(0.02, 0.98, 'Azul: Palavras\nVermelho: Tags POS\nVerde: Transições', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    filename = f'sentence_tagging_{model_type}_order.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def create_markov_visualization_dashboard():
    """Create a complete dashboard of Markov chain visualizations."""
    print("=== VISUALIZAÇÕES DA CADEIA DE MARKOV ===\n")
    
    # Load models
    try:
        second_order_model = load_tagger_model('models/hmm_pos_tagger.pkl')
        print("Modelo de 2ª ordem carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo de 2ª ordem: {e}")
        return
    
    try:
        first_order_model = load_tagger_model('models/hmm_pos_tagger_first_order.pkl')
        print("Modelo de 1ª ordem carregado com sucesso!")
        has_first_order = True
    except Exception as e:
        print(f"Erro ao carregar modelo de 1ª ordem: {e}")
        has_first_order = False
    
    print("\n1. Gerando heatmap de transições (2ª ordem)...")
    visualize_transition_heatmap(second_order_model, order='second')
    
    if has_first_order:
        print("2. Gerando heatmap de transições (1ª ordem)...")
        visualize_transition_heatmap(first_order_model, order='first')
        
        print("3. Comparando matrizes de transição...")
        compare_transition_matrices(first_order_model, second_order_model)
    
    print("4. Gerando grafo da cadeia de Markov (2ª ordem)...")
    visualize_markov_graph(second_order_model, order='second')
    
    if has_first_order:
        print("5. Gerando grafo da cadeia de Markov (1ª ordem)...")
        visualize_markov_graph(first_order_model, order='first')
    
    print("6. Gerando probabilidades de emissão para tags comuns...")
    common_tags = ['NN', 'VB', 'DT', 'JJ', 'IN']
    for tag in common_tags:
        visualize_emission_probabilities(second_order_model, tag)
    
    print("7. Visualizando processo de etiquetagem...")
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "She is reading a book",
        "They will go to school tomorrow"
    ]
    
    for i, sentence in enumerate(sample_sentences):
        print(f"   Sentença {i+1}: {sentence}")
        visualize_sentence_tagging_process(second_order_model, sentence, 'second')
        if has_first_order:
            visualize_sentence_tagging_process(first_order_model, sentence, 'first')
    
    print("\n✅ Todas as visualizações foram geradas e salvas!")

if __name__ == "__main__":
    create_markov_visualization_dashboard()