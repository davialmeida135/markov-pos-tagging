import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from collections import Counter, defaultdict
import csv
import re

def clean_predictions_data(df):
    """Remove special characters and punctuation from predictions for more accurate metrics."""
    # Define punctuation and special character tags to exclude
    special_tags = {
        ',', '.', ':', ';', '!', '?', '"', "'", '`', '``', "''", 
        '-LRB-', '-RRB-', '-LCB-', '-RCB-', '--', '...', '$', '%', '#',
    }
    
    # Also exclude words that are purely punctuation or special characters
    punctuation_pattern = re.compile(r'^[^\w\s]+$')
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows where either predicted or true tag is special punctuation
    mask = ~(cleaned_df['predicted_tag'].isin(special_tags) | 
             cleaned_df['true_tag'].isin(special_tags))
    
    cleaned_df = cleaned_df[mask]
    
    # Remove rows where the word is purely punctuation
    word_mask = ~cleaned_df['word'].str.match(punctuation_pattern, na=False)
    cleaned_df = cleaned_df[word_mask]
    
    # Remove empty or very short words that might be artifacts
    cleaned_df = cleaned_df[cleaned_df['word'].str.len() > 1]
    
    print(f"Original data: {len(df)} predictions")
    print(f"Cleaned data: {len(cleaned_df)} predictions")
    print(f"Removed: {len(df) - len(cleaned_df)} punctuation/special character predictions")
    
    return cleaned_df

def load_predictions(csv_path='predictions.csv', clean_data=True):
    """Load predictions from CSV file with proper handling of commas in data."""
    try:
        # Try reading with quoting to handle commas in fields
        df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL)
    except Exception as e:
        print(f"Erro ao ler {csv_path} com quoting. Tentando método alternativo...")
        try:
            # Alternative: read with different separator or manual parsing
            df = pd.read_csv(csv_path, sep=',', quotechar='"', skipinitialspace=True)
        except Exception as e2:
            print(f"Erro ao carregar {csv_path}: {e2}")
            print("Tentando reparar o arquivo...")
            df = repair_and_load_csv(csv_path)
    
    if df is not None and clean_data:
        df = clean_predictions_data(df)
    
    return df

def repair_and_load_csv(csv_path='predictions.csv'):
    """Repair and load a malformed CSV file."""
    try:
        # Read the raw file and fix it
        repaired_data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header
        header = lines[0].strip().split(',')
        if len(header) != 3:
            header = ['word', 'predicted_tag', 'true_tag']
        
        # Process each line
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            if not line:
                continue
                
            # Try to split properly - expecting exactly 3 fields
            parts = line.split(',')
            
            if len(parts) == 3:
                repaired_data.append(parts)
            elif len(parts) > 3:
                # Merge extra parts into the first field (word)
                word = ','.join(parts[:-2])  # Everything except last 2 fields
                predicted_tag = parts[-2]
                true_tag = parts[-1]
                repaired_data.append([word, predicted_tag, true_tag])
            else:
                print(f"Linha {i} ignorada - campos insuficientes: {line}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(repaired_data, columns=header)
        
        # Clean the data
        df['word'] = df['word'].str.strip(' "\'')
        df['predicted_tag'] = df['predicted_tag'].str.strip(' "\'')
        df['true_tag'] = df['true_tag'].str.strip(' "\'')
        
        print(f"Arquivo reparado com sucesso. {len(df)} linhas carregadas.")
        return df
        
    except Exception as e:
        print(f"Erro ao reparar arquivo: {e}")
        return None

def create_confusion_matrix_visualization(csv_path='predictions.csv', top_n=15, clean_data=True):
    """Create a confusion matrix for POS tag predictions from CSV."""
    df = load_predictions(csv_path, clean_data=clean_data)
    if df is None:
        return
    
    # Get most common tags for better visualization
    common_tags = df['true_tag'].value_counts().head(top_n).index.tolist()
    
    # Filter data to common tags only
    mask = (df['true_tag'].isin(common_tags)) & (df['predicted_tag'].isin(common_tags))
    filtered_df = df[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(filtered_df['true_tag'], filtered_df['predicted_tag'], 
                         labels=common_tags)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=common_tags, 
                yticklabels=common_tags)
    
    title = 'Matriz de Confusão - POS Tagging'
    if clean_data:
        title += ' (Sem Pontuação)'
    plt.title(title)
    plt.xlabel('Tags Preditas')
    plt.ylabel('Tags Verdadeiras')
    plt.tight_layout()
    
    filename = 'confusion_matrix_clean.png' if clean_data else 'confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_performance_by_tag(csv_path='predictions.csv', top_n=20, clean_data=True):
    """Analyze model performance for each POS tag from CSV."""
    df = load_predictions(csv_path, clean_data=clean_data)
    if df is None:
        return None
    
    # Calculate statistics for each tag
    tag_stats = df.groupby('true_tag').agg({
        'predicted_tag': lambda x: (x == x.name).sum(),  # correct predictions
        'word': 'count'  # total occurrences
    }).rename(columns={'predicted_tag': 'correct', 'word': 'total'})
    
    tag_stats['accuracy'] = tag_stats['correct'] / tag_stats['total']
    tag_stats = tag_stats.sort_values('total', ascending=False).head(top_n)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Accuracy plot
    tags = tag_stats.index
    accuracies = tag_stats['accuracy']
    counts = tag_stats['total']
    
    bars1 = ax1.bar(range(len(tags)), accuracies, color='skyblue')
    ax1.set_xlabel('Tags POS')
    ax1.set_ylabel('Acurácia')
    title = 'Acurácia por Tag POS'
    if clean_data:
        title += ' (Sem Pontuação)'
    ax1.set_title(title)
    ax1.set_xticks(range(len(tags)))
    ax1.set_xticklabels(tags, rotation=45)
    ax1.set_ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Frequency plot
    bars2 = ax2.bar(range(len(tags)), counts, color='lightcoral')
    ax2.set_xlabel('Tags POS')
    ax2.set_ylabel('Frequência')
    ax2.set_title('Frequência das Tags no Conjunto de Teste')
    ax2.set_xticks(range(len(tags)))
    ax2.set_xticklabels(tags, rotation=45)
    
    # Add count values on bars
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    filename = 'performance_by_tag_clean.png' if clean_data else 'performance_by_tag.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return tag_stats

def create_error_analysis(csv_path='predictions.csv', clean_data=True):
    """Analyze the most common errors made by the model."""
    df = load_predictions(csv_path, clean_data=clean_data)
    if df is None:
        return
    
    # Find incorrect predictions
    errors = df[df['true_tag'] != df['predicted_tag']]
    
    # Most common error types
    error_types = errors.groupby(['true_tag', 'predicted_tag']).size().reset_index(name='count')
    error_types = error_types.sort_values('count', ascending=False).head(20)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Most common errors
    error_labels = [f"{row['true_tag']} → {row['predicted_tag']}" 
                   for _, row in error_types.iterrows()]
    
    ax1.barh(range(len(error_labels)), error_types['count'], color='salmon')
    ax1.set_yticks(range(len(error_labels)))
    ax1.set_yticklabels(error_labels)
    ax1.set_xlabel('Número de Erros')
    title = '20 Tipos de Erro Mais Comuns'
    if clean_data:
        title += ' (Sem Pontuação)'
    ax1.set_title(title)
    ax1.invert_yaxis()
    
    # Error rate by tag
    total_by_tag = df['true_tag'].value_counts()
    errors_by_tag = errors['true_tag'].value_counts()
    error_rate = (errors_by_tag / total_by_tag).fillna(0).sort_values(ascending=False).head(15)
    
    ax2.bar(range(len(error_rate)), error_rate.values, color='lightcoral')
    ax2.set_xlabel('Tags POS')
    ax2.set_ylabel('Taxa de Erro')
    ax2.set_title('Taxa de Erro por Tag POS')
    ax2.set_xticks(range(len(error_rate)))
    ax2.set_xticklabels(error_rate.index, rotation=45)
    
    plt.tight_layout()
    filename = 'error_analysis_clean.png' if clean_data else 'error_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def create_word_difficulty_analysis(csv_path='predictions.csv', clean_data=True):
    """Analyze which words are most difficult to tag correctly."""
    df = load_predictions(csv_path, clean_data=clean_data)
    if df is None:
        return
    
    # Calculate accuracy for each word
    word_stats = df.groupby('word').agg({
        'predicted_tag': lambda x: (x == df.loc[x.index, 'true_tag']).sum(),
        'true_tag': 'count'
    }).rename(columns={'predicted_tag': 'correct', 'true_tag': 'total'})
    
    word_stats['accuracy'] = word_stats['correct'] / word_stats['total']
    
    # Filter words that appear at least 5 times
    frequent_words = word_stats[word_stats['total'] >= 5]
    
    # Most difficult words (lowest accuracy)
    difficult_words = frequent_words.sort_values('accuracy').head(20)
    
    # Easiest words (highest accuracy, but not 100%)
    easy_words = frequent_words[frequent_words['accuracy'] < 1.0].sort_values('accuracy', ascending=False).head(20)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Most difficult words
    ax1.barh(range(len(difficult_words)), difficult_words['accuracy'], color='red', alpha=0.7)
    ax1.set_yticks(range(len(difficult_words)))
    ax1.set_yticklabels(difficult_words.index)
    ax1.set_xlabel('Acurácia')
    title = '20 Palavras Mais Difíceis de Classificar'
    if clean_data:
        title += ' (Sem Pontuação)'
    ax1.set_title(title)
    ax1.invert_yaxis()
    
    # Easiest words
    ax2.barh(range(len(easy_words)), easy_words['accuracy'], color='green', alpha=0.7)
    ax2.set_yticks(range(len(easy_words)))
    ax2.set_yticklabels(easy_words.index)
    ax2.set_xlabel('Acurácia')
    ax2.set_title('20 Palavras Mais Fáceis de Classificar')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    filename = 'word_difficulty_clean.png' if clean_data else 'word_difficulty.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def create_overall_metrics_dashboard(csv_path='predictions.csv', clean_data=True):
    """Create a dashboard with overall model metrics."""
    df = load_predictions(csv_path, clean_data=clean_data)
    if df is None:
        return
    
    # Calculate overall metrics
    total_predictions = len(df)
    correct_predictions = (df['true_tag'] == df['predicted_tag']).sum()
    overall_accuracy = correct_predictions / total_predictions
    
    # Tag distribution
    tag_distribution = df['true_tag'].value_counts().head(10)
    
    # Accuracy by sentence length (approximated by grouping consecutive words)
    df_copy = df.copy()
    df_copy['sentence_id'] = (df_copy['word'] == df_copy['word'].shift()).cumsum()
    sentence_lengths = df_copy.groupby('sentence_id').size()
    sentence_accuracy = df_copy.groupby('sentence_id').apply(
        lambda x: (x['true_tag'] == x['predicted_tag']).mean()
    )
    
    # Create bins for sentence length
    length_bins = pd.cut(sentence_lengths, bins=[0, 10, 20, 30, 50, 100], 
                        labels=['1-10', '11-20', '21-30', '31-50', '51+'])
    length_accuracy = sentence_accuracy.groupby(length_bins).mean()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall metrics
    data_type = "(Sem Pontuação)" if clean_data else "(Com Pontuação)"
    metrics_text = f"""
    Métricas Gerais do Modelo {data_type}:
    
    Acurácia Total: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)
    Total de Predições: {total_predictions:,}
    Predições Corretas: {correct_predictions:,}
    Predições Incorretas: {total_predictions - correct_predictions:,}
    
    Número de Tags Únicas: {df['true_tag'].nunique()}
    Número de Palavras Únicas: {df['word'].nunique()}
    """
    
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Resumo Geral')
    
    # Tag distribution
    ax2.bar(range(len(tag_distribution)), tag_distribution.values, color='skyblue')
    ax2.set_xlabel('Tags POS')
    ax2.set_ylabel('Frequência')
    ax2.set_title('Top 10 Tags Mais Frequentes')
    ax2.set_xticks(range(len(tag_distribution)))
    ax2.set_xticklabels(tag_distribution.index, rotation=45)
    
    # Accuracy by sentence length
    length_accuracy_clean = length_accuracy.dropna()
    ax3.bar(range(len(length_accuracy_clean)), length_accuracy_clean.values, color='lightgreen')
    ax3.set_xlabel('Tamanho da Sentença (palavras)')
    ax3.set_ylabel('Acurácia Média')
    ax3.set_title('Acurácia por Tamanho da Sentença')
    ax3.set_xticks(range(len(length_accuracy_clean)))
    ax3.set_xticklabels(length_accuracy_clean.index)
    ax3.set_ylim(0, 1)
    
    # Prediction confidence (simplified as correct/incorrect ratio)
    confidence_data = [correct_predictions, total_predictions - correct_predictions]
    labels = ['Corretas', 'Incorretas']
    colors = ['lightgreen', 'lightcoral']
    
    ax4.pie(confidence_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Distribuição de Predições')
    
    plt.tight_layout()
    filename = 'overall_metrics_clean.png' if clean_data else 'overall_metrics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def generate_classification_report(csv_path='predictions.csv', clean_data=True):
    """Generate and display a detailed classification report."""
    df = load_predictions(csv_path, clean_data=clean_data)
    if df is None:
        return
    
    # Generate classification report
    report = classification_report(df['true_tag'], df['predicted_tag'], 
                                 output_dict=True, zero_division=0)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    data_type = "(Sem Pontuação)" if clean_data else "(Com Pontuação)"
    # Display top and bottom performing tags
    print(f"=== RELATÓRIO DE CLASSIFICAÇÃO {data_type} ===\n")
    print(f"Acurácia Geral: {report['accuracy']:.4f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    # Filter out summary rows
    tag_metrics = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    tag_metrics = tag_metrics.sort_values('f1-score', ascending=False)
    
    print("\n=== TOP 10 TAGS (por F1-Score) ===")
    print(tag_metrics.head(10)[['precision', 'recall', 'f1-score', 'support']].to_string())
    
    print("\n=== BOTTOM 10 TAGS (por F1-Score) ===")
    print(tag_metrics.tail(10)[['precision', 'recall', 'f1-score', 'support']].to_string())
    
    return report_df

def compare_metrics_with_without_punctuation(csv_path='predictions.csv'):
    """Compare metrics with and without punctuation for analysis."""
    print("=== COMPARAÇÃO: COM vs SEM PONTUAÇÃO ===\n")
    
    # Load data both ways
    df_with_punct = load_predictions(csv_path, clean_data=False)
    df_clean = load_predictions(csv_path, clean_data=True)
    
    if df_with_punct is None or df_clean is None:
        print("Erro ao carregar dados")
        return
    
    # Calculate accuracies
    acc_with_punct = (df_with_punct['true_tag'] == df_with_punct['predicted_tag']).mean()
    acc_clean = (df_clean['true_tag'] == df_clean['predicted_tag']).mean()
    
    print(f"Dados originais (com pontuação):")
    print(f"  - Total de predições: {len(df_with_punct):,}")
    print(f"  - Acurácia: {acc_with_punct:.4f} ({acc_with_punct*100:.2f}%)")
    print(f"  - Tags únicas: {df_with_punct['true_tag'].nunique()}")
    
    print(f"\nDados limpos (sem pontuação):")
    print(f"  - Total de predições: {len(df_clean):,}")
    print(f"  - Acurácia: {acc_clean:.4f} ({acc_clean*100:.2f}%)")
    print(f"  - Tags únicas: {df_clean['true_tag'].nunique()}")
    
    print(f"\nDiferença na acurácia: {acc_clean - acc_with_punct:.4f}")
    print(f"Predições removidas: {len(df_with_punct) - len(df_clean):,}")

# Função principal para gerar todas as visualizações
def generate_all_visualizations(csv_path='predictions.csv', clean_data=True):
    """Generate all visualizations from the predictions CSV."""
    data_type = "dados limpos (sem pontuação)" if clean_data else "dados originais (com pontuação)"
    print(f"Gerando visualizações a partir do arquivo de predições usando {data_type}...")
    
    print("1. Matriz de Confusão...")
    create_confusion_matrix_visualization(csv_path, clean_data=clean_data)
    
    print("2. Análise de Performance por Tag...")
    tag_stats = analyze_performance_by_tag(csv_path, clean_data=clean_data)
    
    print("3. Análise de Erros...")
    create_error_analysis(csv_path, clean_data=clean_data)
    
    print("4. Análise de Dificuldade das Palavras...")
    create_word_difficulty_analysis(csv_path, clean_data=clean_data)
    
    print("5. Dashboard de Métricas Gerais...")
    create_overall_metrics_dashboard(csv_path, clean_data=clean_data)
    
    print("6. Relatório de Classificação...")
    report = generate_classification_report(csv_path, clean_data=clean_data)
    
    print("\nTodas as visualizações foram geradas e salvas!")
    
    return tag_stats, report

if __name__ == "__main__":
    # Compare metrics first
    compare_metrics_with_without_punctuation()
    
    # print("\n" + "="*50)
    # print("Gerando visualizações COM pontuação...")
    # tag_stats_with, report_with = generate_all_visualizations(clean_data=False)
    
    print("\n" + "="*50)
    print("Gerando visualizações SEM pontuação (mais precisas)...")
    tag_stats_clean, report_clean = generate_all_visualizations(clean_data=True)