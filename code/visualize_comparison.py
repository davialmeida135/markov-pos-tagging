import pandas as pd
from collections import defaultdict, Counter
import csv

def load_predictions_safe(csv_path):
    """Safely load predictions CSV handling malformed lines."""
    try:
        # First try with quoting
        df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL)
        return df
    except Exception as e:
        print(f"Error reading {csv_path} with quoting. Trying repair...")
        
        # Manual repair
        repaired_data = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header
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
                    print(f"Line {i} skipped - insufficient fields: {line}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(repaired_data, columns=header)
            
            # Clean the data
            df['word'] = df['word'].str.strip(' "\'')
            df['predicted_tag'] = df['predicted_tag'].str.strip(' "\'')
            df['true_tag'] = df['true_tag'].str.strip(' "\'')
            
            print(f"File repaired successfully. {len(df)} lines loaded from {csv_path}")
            return df
            
        except Exception as e2:
            print(f"Error repairing file {csv_path}: {e2}")
            return None

def analyze_order_differences():
    """Analyze where second-order actually helps vs first-order."""
    
    # Load both predictions with safe loading
    print("Loading predictions files...")
    df_first = load_predictions_safe('predictions_first_order.csv')
    df_second = load_predictions_safe('predictions.csv')
    
    if df_first is None or df_second is None:
        print("Failed to load prediction files.")
        return None
    
    print("=== ANÁLISE DETALHADA: 1ª vs 2ª ORDEM ===\n")
    
    # Basic stats
    acc_first = (df_first['true_tag'] == df_first['predicted_tag']).mean()
    acc_second = (df_second['true_tag'] == df_second['predicted_tag']).mean()
    
    print(f"Primeira ordem: {acc_first:.4f} ({acc_first*100:.2f}%)")
    print(f"Segunda ordem: {acc_second:.4f} ({acc_second*100:.2f}%)")
    print(f"Melhoria absoluta: {acc_second - acc_first:.4f}")
    print(f"Melhoria relativa: {((acc_second / acc_first) - 1) * 100:.2f}%")
    
    # Check if they have the same number of predictions
    print(f"Predições primeira ordem: {len(df_first):,}")
    print(f"Predições segunda ordem: {len(df_second):,}")
    
    # Align predictions (use minimum length)
    min_len = min(len(df_first), len(df_second))
    df_first_aligned = df_first.head(min_len).reset_index(drop=True)
    df_second_aligned = df_second.head(min_len).reset_index(drop=True)
    
    # Find where they differ
    first_correct = (df_first_aligned['true_tag'] == df_first_aligned['predicted_tag'])
    second_correct = (df_second_aligned['true_tag'] == df_second_aligned['predicted_tag'])
    
    # Cases where models differ
    second_better = (~first_correct) & second_correct
    first_better = first_correct & (~second_correct)
    both_wrong = (~first_correct) & (~second_correct)
    both_right = first_correct & second_correct
    
    print(f"\n=== COMPARAÇÃO DETALHADA ===")
    print(f"Ambos acertaram: {both_right.sum():,} ({both_right.mean()*100:.1f}%)")
    print(f"Ambos erraram: {both_wrong.sum():,} ({both_wrong.mean()*100:.1f}%)")
    print(f"Só 2ª ordem acertou: {second_better.sum():,} ({second_better.mean()*100:.1f}%)")
    print(f"Só 1ª ordem acertou: {first_better.sum():,} ({first_better.mean()*100:.1f}%)")
    
    # Analyze cases where second-order helps
    if second_better.sum() > 0:
        print(f"\n=== CASOS ONDE 2ª ORDEM AJUDA ===")
        second_better_cases = df_second_aligned[second_better]
        
        # Most common tags where second-order helps
        helped_tags = second_better_cases['true_tag'].value_counts().head(10)
        print("Tags mais beneficiadas pela 2ª ordem:")
        for tag, count in helped_tags.items():
            print(f"  {tag}: {count} casos")
        
        # Show some examples
        print(f"\nExemplos onde 2ª ordem acertou e 1ª errou:")
        for i, (idx, row) in enumerate(second_better_cases.head(10).iterrows()):
            first_pred = df_first_aligned.loc[idx, 'predicted_tag']
            print(f"  {i+1}. '{row['word']}' → True: {row['true_tag']}, "
                  f"1ª: {first_pred}, 2ª: {row['predicted_tag']}")
    
    # Analyze cases where first-order is better
    if first_better.sum() > 0:
        print(f"\n=== CASOS ONDE 1ª ORDEM É MELHOR ===")
        first_better_cases = df_first_aligned[first_better]
        
        print(f"Exemplos onde 1ª ordem acertou e 2ª errou:")
        for i, (idx, row) in enumerate(first_better_cases.head(5).iterrows()):
            second_pred = df_second_aligned.loc[idx, 'predicted_tag']
            print(f"  {i+1}. '{row['word']}' → True: {row['true_tag']}, "
                  f"1ª: {row['predicted_tag']}, 2ª: {second_pred}")
    
    return {
        'accuracy_first': acc_first,
        'accuracy_second': acc_second,
        'improvement': acc_second - acc_first,
        'second_better_count': second_better.sum(),
        'first_better_count': first_better.sum(),
        'total_compared': min_len
    }

def analyze_ambiguous_cases():
    """Find cases where context really matters."""
    
    print("\n" + "="*50)
    print("Loading files for ambiguous case analysis...")
    
    df_first = load_predictions_safe('predictions_first_order.csv')
    df_second = load_predictions_safe('predictions.csv')
    
    if df_first is None or df_second is None:
        print("Failed to load prediction files for ambiguous analysis.")
        return
    
    # Align the datasets
    min_len = min(len(df_first), len(df_second))
    df_first = df_first.head(min_len)
    df_second = df_second.head(min_len)
    
    # Find words that appear with multiple tags
    word_tag_variety = df_second.groupby('word')['true_tag'].nunique()
    ambiguous_words = word_tag_variety[word_tag_variety > 1].index.tolist()
    
    print(f"\n=== PALAVRAS AMBÍGUAS ===")
    print(f"Palavras com múltiplas tags possíveis: {len(ambiguous_words)}")
    
    # Check performance on ambiguous words
    mask_ambiguous = df_second['word'].isin(ambiguous_words)
    
    if mask_ambiguous.sum() > 0:
        acc_first_ambiguous = (df_first[mask_ambiguous]['true_tag'] == 
                              df_first[mask_ambiguous]['predicted_tag']).mean()
        acc_second_ambiguous = (df_second[mask_ambiguous]['true_tag'] == 
                               df_second[mask_ambiguous]['predicted_tag']).mean()
        
        print(f"Acurácia em palavras ambíguas:")
        print(f"  1ª ordem: {acc_first_ambiguous:.4f}")
        print(f"  2ª ordem: {acc_second_ambiguous:.4f}")
        print(f"  Melhoria: {acc_second_ambiguous - acc_first_ambiguous:.4f}")
        
        # Most ambiguous words
        print(f"\nPalavras mais ambíguas:")
        for word in word_tag_variety.nlargest(10).index:
            tags = df_second[df_second['word'] == word]['true_tag'].unique()
            print(f"  '{word}': {list(tags)}")
    else:
        print("No ambiguous words found in the aligned dataset.")

def generate_comparison_summary():
    """Generate a comprehensive comparison summary."""
    print("="*60)
    print("RESUMO DA COMPARAÇÃO ENTRE MODELOS")
    print("="*60)
    
    stats = analyze_order_differences()
    
    if stats is not None:
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"• Acurácia 1ª ordem: {stats['accuracy_first']*100:.2f}%")
        print(f"• Acurácia 2ª ordem: {stats['accuracy_second']*100:.2f}%")
        print(f"• Melhoria absoluta: {stats['improvement']*100:.2f} pontos percentuais")
        print(f"• Casos onde 2ª ordem foi melhor: {stats['second_better_count']:,}")
        print(f"• Casos onde 1ª ordem foi melhor: {stats['first_better_count']:,}")
        print(f"• Total de predições comparadas: {stats['total_compared']:,}")
        
        # Calculate net improvement
        net_improvement = stats['second_better_count'] - stats['first_better_count']
        print(f"• Melhoria líquida (casos): {net_improvement:,}")
        
        analyze_ambiguous_cases()
        
        print(f"\n🎯 CONCLUSÕES:")
        if stats['improvement'] > 0:
            print(f"✅ O modelo de 2ª ordem teve melhor performance geral")
            print(f"✅ Ganhou {stats['second_better_count']:,} casos a mais do que perdeu")
        else:
            print(f"❌ O modelo de 1ª ordem teve melhor performance geral")
        
        if stats['improvement'] < 0.01:  # Less than 1% improvement
            print(f"⚠️  A diferença é marginal - complexidade adicional pode não valer a pena")
        else:
            print(f"🚀 A melhoria é significativa para aplicações críticas")

if __name__ == "__main__":
    generate_comparison_summary()