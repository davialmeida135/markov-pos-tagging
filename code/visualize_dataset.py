import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
from processor import PennTreebankProcessor
import os

def analyze_corpus_structure():
    """Analyze the basic structure of the Penn Treebank corpus."""
    
    print("=== ANÁLISE DO CORPUS PENN TREEBANK ===\n")
    
    # Load the processor
    data_dir = 'data/raw'
    processor = PennTreebankProcessor(data_dir)
    processor.process()
    
    # Basic statistics
    print("📊 ESTATÍSTICAS BÁSICAS:")
    print(f"Sentenças de treino: {len(processor.train):,}")
    print(f"Sentenças de desenvolvimento: {len(processor.dev):,}")
    print(f"Sentenças de teste: {len(processor.test):,}" if processor.test else "Conjunto de teste: Não encontrado")
    
    # Analyze training set in detail
    train_words = []
    train_tags = []
    train_sentence_lengths = []
    
    for sentence in processor.train:
        sentence_length = len(sentence)
        train_sentence_lengths.append(sentence_length)
        
        for word, tag in sentence:
            train_words.append(word.lower())
            train_tags.append(tag)
    
    # Development set analysis
    dev_words = []
    dev_tags = []
    dev_sentence_lengths = []
    
    for sentence in processor.dev:
        sentence_length = len(sentence)
        dev_sentence_lengths.append(sentence_length)
        
        for word, tag in sentence:
            dev_words.append(word.lower())
            dev_tags.append(tag)
    
    print(f"\n📝 TOKENS:")
    print(f"Tokens de treino: {len(train_words):,}")
    print(f"Tokens de desenvolvimento: {len(dev_words):,}")
    print(f"Total de tokens: {len(train_words) + len(dev_words):,}")
    
    print(f"\n📚 VOCABULÁRIO:")
    train_vocab = set(train_words)
    dev_vocab = set(dev_words)
    print(f"Vocabulário único (treino): {len(train_vocab):,}")
    print(f"Vocabulário único (dev): {len(dev_vocab):,}")
    print(f"Vocabulário total único: {len(train_vocab.union(dev_vocab)):,}")
    print(f"Sobreposição vocab (treino ∩ dev): {len(train_vocab.intersection(dev_vocab)):,}")
    
    # OOV analysis
    oov_words = dev_vocab - train_vocab
    print(f"Palavras OOV no dev: {len(oov_words):,} ({len(oov_words)/len(dev_vocab)*100:.2f}%)")
    
    print(f"\n🏷️ TAGS:")
    train_tag_set = set(train_tags)
    dev_tag_set = set(dev_tags)
    print(f"Tags únicas (treino): {len(train_tag_set)}")
    print(f"Tags únicas (dev): {len(dev_tag_set)}")
    print(f"Tags em comum: {len(train_tag_set.intersection(dev_tag_set))}")
    
    return {
        'train_words': train_words,
        'train_tags': train_tags,
        'dev_words': dev_words,
        'dev_tags': dev_tags,
        'train_sentence_lengths': train_sentence_lengths,
        'dev_sentence_lengths': dev_sentence_lengths,
        'train_vocab': train_vocab,
        'dev_vocab': dev_vocab,
        'oov_words': oov_words
    }

def analyze_sentence_statistics(corpus_data):
    """Analyze sentence length statistics."""
    
    print(f"\n📏 ESTATÍSTICAS DE COMPRIMENTO DAS SENTENÇAS:")
    
    train_lengths = corpus_data['train_sentence_lengths']
    dev_lengths = corpus_data['dev_sentence_lengths']
    
    print(f"\nTreino:")
    print(f"  Comprimento médio: {np.mean(train_lengths):.2f} palavras")
    print(f"  Mediana: {np.median(train_lengths):.0f} palavras")
    print(f"  Mín/Máx: {min(train_lengths)}/{max(train_lengths)} palavras")
    print(f"  Desvio padrão: {np.std(train_lengths):.2f}")
    
    print(f"\nDesenvolvimento:")
    print(f"  Comprimento médio: {np.mean(dev_lengths):.2f} palavras")
    print(f"  Mediana: {np.median(dev_lengths):.0f} palavras")
    print(f"  Mín/Máx: {min(dev_lengths)}/{max(dev_lengths)} palavras")
    print(f"  Desvio padrão: {np.std(dev_lengths):.2f}")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(train_lengths, bins=50, alpha=0.7, label='Treino', color='blue')
    plt.hist(dev_lengths, bins=50, alpha=0.7, label='Dev', color='red')
    plt.xlabel('Comprimento da Sentença')
    plt.ylabel('Frequência')
    plt.title('Distribuição do Comprimento das Sentenças')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Box plot
    plt.boxplot([train_lengths, dev_lengths], labels=['Treino', 'Dev'])
    plt.ylabel('Comprimento da Sentença')
    plt.title('Box Plot - Comprimento das Sentenças')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentence_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_tag_distribution(corpus_data):
    """Analyze POS tag distribution."""
    
    print(f"\n🏷️ DISTRIBUIÇÃO DAS TAGS POS:")
    
    train_tag_counts = Counter(corpus_data['train_tags'])
    dev_tag_counts = Counter(corpus_data['dev_tags'])
    
    # Most common tags
    print(f"\nTags mais frequentes (treino):")
    for tag, count in train_tag_counts.most_common(15):
        percentage = count / len(corpus_data['train_tags']) * 100
        print(f"  {tag:>6}: {count:>8,} ({percentage:>5.2f}%)")
    
    print(f"\nTags mais frequentes (dev):")
    for tag, count in dev_tag_counts.most_common(15):
        percentage = count / len(corpus_data['dev_tags']) * 100
        print(f"  {tag:>6}: {count:>8,} ({percentage:>5.2f}%)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Training set
    top_tags_train = train_tag_counts.most_common(20)
    tags_train, counts_train = zip(*top_tags_train)
    
    ax1.bar(range(len(tags_train)), counts_train, color='skyblue')
    ax1.set_xticks(range(len(tags_train)))
    ax1.set_xticklabels(tags_train, rotation=45, ha='right')
    ax1.set_title('Distribuição das Tags - Conjunto de Treino')
    ax1.set_ylabel('Frequência')
    
    # Development set
    top_tags_dev = dev_tag_counts.most_common(20)
    tags_dev, counts_dev = zip(*top_tags_dev)
    
    ax2.bar(range(len(tags_dev)), counts_dev, color='lightcoral')
    ax2.set_xticks(range(len(tags_dev)))
    ax2.set_xticklabels(tags_dev, rotation=45, ha='right')
    ax2.set_title('Distribuição das Tags - Conjunto de Desenvolvimento')
    ax2.set_ylabel('Frequência')
    
    plt.tight_layout()
    plt.savefig('tag_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_tag_counts, dev_tag_counts

def analyze_word_frequency(corpus_data):
    """Analyze word frequency distribution."""
    
    print(f"\n📊 ANÁLISE DE FREQUÊNCIA DAS PALAVRAS:")
    
    train_word_counts = Counter(corpus_data['train_words'])
    dev_word_counts = Counter(corpus_data['dev_words'])
    
    print(f"\nPalavras mais frequentes (treino):")
    for word, count in train_word_counts.most_common(20):
        percentage = count / len(corpus_data['train_words']) * 100
        print(f"  {word:>15}: {count:>8,} ({percentage:>5.2f}%)")
    
    # Hapax legomena (words that appear only once)
    hapax_train = sum(1 for count in train_word_counts.values() if count == 1)
    hapax_dev = sum(1 for count in dev_word_counts.values() if count == 1)
    
    print(f"\nHapax Legomena (palavras com frequência 1):")
    print(f"  Treino: {hapax_train:,} ({hapax_train/len(train_word_counts)*100:.2f}% do vocabulário)")
    print(f"  Dev: {hapax_dev:,} ({hapax_dev/len(dev_word_counts)*100:.2f}% do vocabulário)")
    
    # Frequency distribution
    freq_dist_train = Counter(train_word_counts.values())
    print(f"\nDistribuição de frequências (treino):")
    for freq in sorted(freq_dist_train.keys())[:10]:
        count = freq_dist_train[freq]
        print(f"  Frequência {freq}: {count:,} palavras")
    
    return train_word_counts, dev_word_counts

def analyze_predictions_vs_corpus():
    """Compare predictions with original corpus data."""
    
    print(f"\n🔍 COMPARAÇÃO: PREDIÇÕES vs CORPUS ORIGINAL:")
    
    # Load prediction files
    try:
        df_first = pd.read_csv('predictions_first_order.csv')
        df_second = pd.read_csv('predictions.csv')
        
        print(f"Predições 1ª ordem: {len(df_first):,}")
        print(f"Predições 2ª ordem: {len(df_second):,}")
        
        # Analyze tag distribution in predictions
        pred_first_tags = Counter(df_first['true_tag'])
        pred_second_tags = Counter(df_second['true_tag'])
        
        print(f"\nTags mais frequentes nas predições (1ª ordem):")
        for tag, count in pred_first_tags.most_common(10):
            percentage = count / len(df_first) * 100
            print(f"  {tag:>6}: {count:>6,} ({percentage:>5.2f}%)")
            
        # Check for data consistency
        if len(df_first) == len(df_second):
            # Compare true tags
            tags_match = (df_first['true_tag'] == df_second['true_tag']).all()
            words_match = (df_first['word'] == df_second['word']).all()
            
            print(f"\nConsistência dos dados:")
            print(f"  Tags verdadeiras iguais: {'✅' if tags_match else '❌'}")
            print(f"  Palavras iguais: {'✅' if words_match else '❌'}")
        else:
            print(f"\n⚠️ Tamanhos diferentes entre arquivos de predição!")
            
    except FileNotFoundError as e:
        print(f"❌ Arquivo de predições não encontrado: {e}")

def analyze_ambiguous_words(corpus_data):
    """Analyze words that can have multiple POS tags."""
    
    print(f"\n🤔 ANÁLISE DE PALAVRAS AMBÍGUAS:")
    
    # Create word-to-tags mapping
    word_tags = defaultdict(set)
    
    for word, tag in zip(corpus_data['train_words'], corpus_data['train_tags']):
        word_tags[word].add(tag)
    
    # Find ambiguous words
    ambiguous_words = {word: tags for word, tags in word_tags.items() if len(tags) > 1}
    
    print(f"Palavras ambíguas (múltiplas tags): {len(ambiguous_words):,}")
    print(f"Percentual do vocabulário: {len(ambiguous_words)/len(word_tags)*100:.2f}%")
    
    # Most ambiguous words
    most_ambiguous = sorted(ambiguous_words.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nPalavras mais ambíguas:")
    for word, tags in most_ambiguous[:20]:
        tag_list = sorted(list(tags))
        print(f"  {word:>15}: {len(tags)} tags → {', '.join(tag_list)}")
    
    # Ambiguity distribution
    ambiguity_dist = Counter(len(tags) for tags in ambiguous_words.values())
    
    print(f"\nDistribuição de ambiguidade:")
    for num_tags, count in sorted(ambiguity_dist.items()):
        print(f"  {num_tags} tags: {count:,} palavras")
    
    return ambiguous_words

def generate_corpus_report():
    """Generate a comprehensive corpus analysis report."""
    
    print("🔍 GERANDO RELATÓRIO COMPLETO DO CORPUS...\n")
    
    # Main analysis
    corpus_data = analyze_corpus_structure()
    
    # Detailed analyses
    analyze_sentence_statistics(corpus_data)
    train_tag_counts, dev_tag_counts = analyze_tag_distribution(corpus_data)
    train_word_counts, dev_word_counts = analyze_word_frequency(corpus_data)
    ambiguous_words = analyze_ambiguous_words(corpus_data)
    
    # Compare with predictions
    analyze_predictions_vs_corpus()
    
    # Generate summary statistics
    print(f"\n" + "="*60)
    print("📋 RESUMO ESTATÍSTICO FINAL")
    print("="*60)
    
    print(f"📊 Corpus:")
    print(f"  • Total de sentenças: {len(corpus_data['train_sentence_lengths']) + len(corpus_data['dev_sentence_lengths']):,}")
    print(f"  • Total de tokens: {len(corpus_data['train_words']) + len(corpus_data['dev_words']):,}")
    print(f"  • Vocabulário único: {len(corpus_data['train_vocab'].union(corpus_data['dev_vocab'])):,}")
    print(f"  • Tags POS únicas: {len(set(corpus_data['train_tags']).union(set(corpus_data['dev_tags'])))}")
    
    print(f"\n🎯 Complexidade:")
    print(f"  • Palavras ambíguas: {len(ambiguous_words):,}")
    print(f"  • OOV rate (dev): {len(corpus_data['oov_words'])/len(corpus_data['dev_vocab'])*100:.2f}%")
    print(f"  • Hapax legomena: {sum(1 for count in train_word_counts.values() if count == 1):,}")
    
    print(f"\n📏 Sentenças:")
    all_lengths = corpus_data['train_sentence_lengths'] + corpus_data['dev_sentence_lengths']
    print(f"  • Comprimento médio: {np.mean(all_lengths):.2f} palavras")
    print(f"  • Comprimento mín/máx: {min(all_lengths)}/{max(all_lengths)} palavras")
    
    print(f"\n✅ Relatório completo gerado!")
    print(f"📁 Gráficos salvos: sentence_length_analysis.png, tag_distribution.png")

if __name__ == "__main__":
    generate_corpus_report()