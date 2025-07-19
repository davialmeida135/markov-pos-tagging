import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import pickle

def visualize_trigram_transitions_3d(model_data, top_n=8):
    """Visualize true second-order transitions as 3D heatmap."""
    transition_counts = model_data['transition_counts']
    tag_counts = model_data['tag_unigram_counts']
    
    # Get most common tags
    common_tags = [tag for tag, _ in Counter(tag_counts).most_common(top_n) if tag != '<s>']
    
    # Create 3D matrix: [prev2][prev1][curr]
    transition_matrix = np.zeros((len(common_tags), len(common_tags), len(common_tags)))
    
    for i, prev2 in enumerate(common_tags):
        for j, prev1 in enumerate(common_tags):
            bigram_key = (prev2, prev1)
            if bigram_key in transition_counts:
                total = sum(transition_counts[bigram_key].values())
                for k, curr in enumerate(common_tags):
                    count = transition_counts[bigram_key].get(curr, 0)
                    transition_matrix[i][j][k] = count / total if total > 0 else 0
    
    # Create multiple 2D heatmaps, one for each prev2 tag
    n_rows = 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, prev2_tag in enumerate(common_tags[:8]):
        ax = axes[i]
        
        # Get the slice for this prev2_tag
        matrix_slice = transition_matrix[i]
        
        sns.heatmap(matrix_slice, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=common_tags, yticklabels=common_tags, ax=ax)
        ax.set_title(f'P(curr | prev1, "{prev2_tag}")')
        ax.set_xlabel('Tag Atual')
        ax.set_ylabel('Tag Anterior (-1)')
    
    plt.suptitle('Transições de Segunda Ordem: P(tag_atual | tag_{-1}, tag_{-2})', fontsize=16)
    plt.tight_layout()
    plt.savefig('second_order_transitions_complete.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_bigram_states_graph(model_data, top_n=6, min_prob=0.1):
    """Visualize second-order HMM using bigram states as nodes."""
    transition_counts = model_data['transition_counts']
    tag_counts = model_data['tag_unigram_counts']
    
    # Get most common tags
    common_tags = [tag for tag, _ in Counter(tag_counts).most_common(top_n) if tag != '<s>']
    
    # Create directed graph where nodes are bigrams (prev1, prev2)
    G = nx.DiGraph()
    
    # Add nodes (bigram states)
    bigram_nodes = []
    for tag1 in common_tags:
        for tag2 in common_tags:
            bigram = f"{tag1},{tag2}"
            bigram_nodes.append(bigram)
            G.add_node(bigram)
    
    # Add edges (transitions from one bigram to another)
    for (prev2, prev1), counter in transition_counts.items():
        if prev2 in common_tags and prev1 in common_tags:
            source_bigram = f"{prev2},{prev1}"
            total = sum(counter.values())
            
            for curr_tag, count in counter.items():
                if curr_tag in common_tags:
                    # The target bigram is (prev1, curr_tag)
                    target_bigram = f"{prev1},{curr_tag}"
                    prob = count / total if total > 0 else 0
                    
                    if prob >= min_prob and target_bigram in bigram_nodes:
                        G.add_edge(source_bigram, target_bigram, weight=prob)
    
    # Create layout
    plt.figure(figsize=(20, 15))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=5, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.8)
    
    # Draw edges with varying thickness
    edges = G.edges()
    if edges:
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(G, pos, width=[w/max_weight * 3 for w in weights],
                              alpha=0.6, edge_color='gray', arrows=True,
                              arrowsize=20, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Add edge labels for strong connections
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d['weight'] >= min_prob * 1.5:
            edge_labels[(u, v)] = f"{d['weight']:.2f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title(f'HMM 2ª Ordem: Estados como Bigramas\n(Transições com probabilidade ≥ {min_prob})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('second_order_bigram_states.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_conditional_transitions(model_data, condition_tag='DT', top_n=10):
    """Visualize transitions conditioned on a specific second-previous tag."""
    transition_counts = model_data['transition_counts']
    tag_counts = model_data['tag_unigram_counts']
    
    # Get most common tags
    common_tags = [tag for tag, _ in Counter(tag_counts).most_common(top_n) if tag != '<s>']
    
    # Filter transitions where prev2 = condition_tag
    conditional_transitions = {}
    for (prev2, prev1), counter in transition_counts.items():
        if prev2 == condition_tag and prev1 in common_tags:
            conditional_transitions[prev1] = counter
    
    if not conditional_transitions:
        print(f"Nenhuma transição encontrada com tag anterior '{condition_tag}'")
        return
    
    # Create transition matrix
    matrix = np.zeros((len(common_tags), len(common_tags)))
    
    for i, prev1 in enumerate(common_tags):
        if prev1 in conditional_transitions:
            total = sum(conditional_transitions[prev1].values())
            for j, curr in enumerate(common_tags):
                count = conditional_transitions[prev1].get(curr, 0)
                matrix[i][j] = count / total if total > 0 else 0
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=common_tags, yticklabels=common_tags)
    plt.title(f'Transições Condicionadas: P(tag_atual | tag_{{-1}}, "{condition_tag}")')
    plt.xlabel('Tag Atual')
    plt.ylabel('Tag Anterior (-1)')
    plt.tight_layout()
    plt.savefig(f'conditional_transitions_{condition_tag}.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_first_vs_second_order_effects(model_data, top_n=8):
    """Compare how second-order context changes transition probabilities."""
    transition_counts = model_data['transition_counts']
    tag_counts = model_data['tag_unigram_counts']
    
    # Get most common tags
    common_tags = [tag for tag, _ in Counter(tag_counts).most_common(top_n) if tag != '<s>']
    
    # Calculate first-order approximation (marginalizing over prev2)
    first_order_approx = defaultdict(Counter)
    for (prev2, prev1), counter in transition_counts.items():
        if prev1 in common_tags and prev2 in common_tags:
            for curr, count in counter.items():
                if curr in common_tags:
                    first_order_approx[prev1][curr] += count
    
    # Choose a specific prev1 tag for comparison
    focus_tag = 'NN'  # Can be changed
    if focus_tag not in common_tags:
        focus_tag = common_tags[0]
    
    # Compare transitions FROM focus_tag under different prev2 contexts
    contexts = ['DT', 'JJ', 'NN', 'VB']  # Different prev2 contexts
    contexts = [tag for tag in contexts if tag in common_tags]
    
    fig, axes = plt.subplots(1, len(contexts) + 1, figsize=(20, 4))
    
    # First-order approximation
    if focus_tag in first_order_approx:
        total = sum(first_order_approx[focus_tag].values())
        probs = [first_order_approx[focus_tag].get(tag, 0) / total for tag in common_tags]
        
        axes[0].bar(range(len(common_tags)), probs, alpha=0.7, color='blue')
        axes[0].set_title(f'1ª Ordem: P(· | {focus_tag})')
        axes[0].set_xticks(range(len(common_tags)))
        axes[0].set_xticklabels(common_tags, rotation=45)
        axes[0].set_ylim(0, max(probs) * 1.1 if probs else 1)
    
    # Second-order for different contexts
    for i, context in enumerate(contexts):
        bigram_key = (context, focus_tag)
        if bigram_key in transition_counts:
            total = sum(transition_counts[bigram_key].values())
            probs = [transition_counts[bigram_key].get(tag, 0) / total for tag in common_tags]
            
            axes[i+1].bar(range(len(common_tags)), probs, alpha=0.7, color='red')
            axes[i+1].set_title(f'2ª Ordem: P(· | {focus_tag}, {context})')
            axes[i+1].set_xticks(range(len(common_tags)))
            axes[i+1].set_xticklabels(common_tags, rotation=45)
            axes[i+1].set_ylim(0, max(probs) * 1.1 if probs else 1)
    
    plt.suptitle(f'Impacto do Contexto de 2ª Ordem na Tag "{focus_tag}"')
    plt.tight_layout()
    plt.savefig('first_vs_second_order_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_transition_examples(model_data):
    """Show concrete examples of how second-order differs from first-order."""
    transition_counts = model_data['transition_counts']
    
    # Find interesting examples where context matters
    examples = []
    
    for (prev2, prev1), counter in transition_counts.items():
        if prev2 != '<s>' and prev1 != '<s>':
            total = sum(counter.values())
            if total >= 10:  # Only consider frequent bigrams
                for curr, count in counter.most_common(3):
                    prob = count / total
                    examples.append({
                        'prev2': prev2,
                        'prev1': prev1,
                        'curr': curr,
                        'prob': prob,
                        'count': count,
                        'context': f"{prev2} → {prev1} → {curr}"
                    })
    
    # Sort by probability and take top examples
    examples.sort(key=lambda x: x['prob'], reverse=True)
    top_examples = examples[:15]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    
    contexts = [ex['context'] for ex in top_examples]
    probs = [ex['prob'] for ex in top_examples]
    
    bars = ax.barh(range(len(contexts)), probs, color='skyblue', alpha=0.7)
    
    # Add probability labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', ha='left', va='center', fontsize=9)
    
    ax.set_yticks(range(len(contexts)))
    ax.set_yticklabels(contexts)
    ax.set_xlabel('Probabilidade de Transição')
    ax.set_title('Exemplos de Transições de 2ª Ordem Mais Prováveis')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('transition_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some examples
    print("=== EXEMPLOS DE TRANSIÇÕES DE 2ª ORDEM ===")
    for ex in top_examples[:10]:
        print(f"P({ex['curr']} | {ex['prev1']}, {ex['prev2']}) = {ex['prob']:.3f} "
              f"(observado {ex['count']} vezes)")

def create_advanced_markov_visualization_dashboard():
    """Create advanced visualizations for second-order HMM."""
    print("=== VISUALIZAÇÕES AVANÇADAS - HMM 2ª ORDEM ===\n")
    
    # Load model
    try:
        model_data = load_tagger_model('models/hmm_pos_tagger.pkl')
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
    
    print("1. Visualizando transições completas de 2ª ordem...")
    visualize_trigram_transitions_3d(model_data)
    
    print("2. Criando grafo com estados bigramas...")
    visualize_bigram_states_graph(model_data)
    
    print("3. Analisando transições condicionadas...")
    for condition in ['DT', 'JJ', 'NN']:
        visualize_conditional_transitions(model_data, condition)
    
    print("4. Comparando efeitos de 1ª vs 2ª ordem...")
    compare_first_vs_second_order_effects(model_data)
    
    print("5. Mostrando exemplos concretos...")
    visualize_transition_examples(model_data)
    
    print("\n✅ Visualizações avançadas geradas!")

def load_tagger_model(model_path):
    """Load the trained HMM model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

if __name__ == "__main__":
    create_advanced_markov_visualization_dashboard()