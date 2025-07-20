# markov-pos-tagging

RELATÓRIO DE TRABALHO FINAL DA DISCIPLINA IMD3001 - INTRODUÇÃO À INTELIGÊNCIA ARTIFICIAL.
ALUNOS: DAVI DIÓGENES FERREIRA DE ALMEIDA E VICTOR GABRIEL RIBEIRO MENEZES.
O presente relatório baseia-se na implementação deste repositório, como proposto, e também - mas não restrito - aos slides referentes à apresentação.
Este projeto implementa um Part-of-Speech (POS) Tagger baseado em um Modelo Oculto de Markov (HMM) de segunda ordem, utilizando o corpus Penn Tree Bank. A proposta é demonstrar como modelos probabilísticos com memória de dois estados anteriores (ordem superior) podem ser usados para realizar anotação gramatical de sentenças em língua inglesa.

1. Objetivos e Aplicações
Modelo Escolhido: Hidden Markov Model (HMM)
O principal objetivo deste projeto é desenvolver e avaliar a eficácia dos Modelos Ocultos de Markov na identificação de classes gramaticais em um texto. Foram implementadas duas variantes do modelo:


HMM de Primeira Ordem: Um modelo que cno caso onsidera apenas a tag anterior para prever a tag atual.


HMM de Segunda Ordem: Um modelo que utiliza as duas tags anteriores para a predição, visando capturar contextos mais longos e, teoricamente, melhorar a acurácia.

A aplicação de HMMs para POS tagging é uma abordagem clássica e robusta em Processamento de Linguagem Natural (PLN), sendo fundamental para tarefas mais complexas como parsing sintático, extração de informação e tradução automática.

Dataset Utilizado: Penn Treebank
Para o treinamento e avaliação dos modelos, foi utilizado o corpus do Penn Treebank, um dos mais conhecidos e utilizados para essa finalidade. O corpus é composto por textos com palavras já anotadas no formato 

PALAVRA_TAG.

As estatísticas do corpus são:


Total de Tags: 46 


Tokens no Corpus de Treino: 912.344 


Tokens no Corpus de Teste: 109.381 

2. Algoritmos e Representação de Conhecimento
O projeto se baseia em um algoritmo de aprendizado de máquina probabilístico, o Hidden Markov Model.

Representação de Conhecimento: O conhecimento é representado por meio de probabilidades. O modelo aprende dois tipos principais de probabilidades a partir do corpus de treino:


Probabilidades de Transição: A probabilidade de uma tag gramatical ser seguida por outra (ex: a probabilidade de um Artigo ser seguido por um Substantivo).


Probabilidades de Emissão: A probabilidade de uma determinada palavra ser associada a uma tag gramatical específica (ex: a probabilidade da palavra "homens" ser um Substantivo).

Algoritmo de IA: O algoritmo de Viterbi é utilizado para encontrar a sequência mais provável 
de tags (estados ocultos) dada uma sequência de palavras (observações). A implementação compara um modelo de primeira ordem, que usa P(Tag_i | Tag_{i-1}), com um de segunda ordem, que usa P(Tag_i | Tag_{i-1}, Tag_{i-2}).

3. Modelagem PEAS e Arquitetura do Agente
A modelagem PEAS (Performance, Environment, Actuators, Sensors) para o agente de POS tagging pode ser descrita da seguinte forma:

Performance (Desempenho): Acurácia na classificação das tags gramaticais. O objetivo é maximizar o número de palavras corretamente etiquetadas. A métrica principal utilizada foi a acurácia total.


Environment (Ambiente): O corpus do Penn Treebank, composto por frases e sentenças em inglês. O ambiente é parcialmente observável (o agente vê as palavras, mas não as tags corretas) e determinístico (a probabilidade de transição e emissão são fixas após o treino).

Actuators (Atuadores): A saída do agente é a sequência de tags gramaticais atribuída à sentença de entrada.

Sensors (Sensores): O agente "lê" a sequência de palavras (tokens) de uma frase ou texto.

Arquitetura do Agente
A arquitetura do agente é baseada em um modelo de estados ocultos.


Estados Ocultos: As classes gramaticais (tags como Artigo, Substantivo, Verbo). O agente não as observa diretamente.


Estados Observáveis: As palavras do texto (como "Os", "homens", "sorriram").

Funcionamento: O agente calcula a sequência de estados ocultos mais provável que poderia ter gerado a sequência de estados observáveis de entrada, utilizando as probabilidades de transição e emissão aprendidas durante o treino.

4. Análise do Código
Uma visão geral da implementação do código.

processor.py
Este módulo é responsável pelo pré-processamento dos dados. A classe PennTreebankProcessor tem como função:

Localizar os arquivos do corpus nos diretórios train, dev e test.

Analisar (parse) cada arquivo, lendo as linhas e dividindo os tokens no formato palavra_tag.

Estruturar os dados em listas de sentenças, onde cada sentença é uma lista de tuplas (palavra, tag), pronta para ser usada nos modelos.

firstorder.py
Implementa o HMM de Primeira Ordem.

train(self, tagged_sentences): Itera sobre as sentenças de treino para contar as ocorrências e popular as estruturas de dados transition_counts, emission_counts e tag_unigram_counts. Um símbolo <s> é adicionado no início de cada sentença para marcar a transição inicial.

transition_prob e emission_prob: Calculam as probabilidades P(tag_atual | tag_anterior) e P(palavra | tag), respectivamente. Utilizam uma pequena constante (1e-6) para evitar divisão por zero (suavização de Laplace).

viterbi(self, sentence): É o núcleo do agente. Implementa o algoritmo de Viterbi para encontrar o caminho mais provável de tags.

Log-Probabilities: As probabilidades são convertidas para o espaço logarítmico (math.log) para evitar underflow numérico em sentenças longas e transformar multiplicações em somas, o que é computacionalmente mais estável.

Backpointers: Uma matriz backpointers é usada para reconstruir a sequência de tags mais provável ao final do processo, movendo-se do final da sentença para o início.

secondorder.py
Implementa o HMM de Segunda Ordem. A lógica é uma extensão do modelo de primeira ordem.

train(self, tagged_sentences): A principal diferença é que o estado anterior agora é um bigrama de tags (tag_anterior_2, tag_anterior_1). Dois símbolos <s> são adicionados ao início para o contexto inicial. As contagens refletem essa mudança, usando tuplas como chaves para as transições.

transition_prob: Calcula a probabilidade P(tag_atual | tag_anterior_2, tag_anterior_1).

viterbi(self, sentence): A implementação do Viterbi é mais complexa.

Matrizes V e backpointers: São tridimensionais (posição, tag_anterior, tag_atual) para acomodar o estado de bigrama.

Complexidade: A complexidade do algoritmo aumenta significativamente, pois para cada palavra e cada par de tags possíveis, é preciso iterar sobre todas as tags anteriores para encontrar o melhor caminho.

5. Métricas, Desempenho e Validação
Métricas Empregadas
A principal métrica para avaliar o desempenho dos modelos foi a 

Acurácia Total, que mede a porcentagem de tokens classificados corretamente.

Além disso, foram geradas 

Matrizes de Confusão para uma análise mais detalhada, permitindo visualizar quais tags eram mais confundidas entre si. A análise de erros também foi detalhada, mostrando os 20 tipos de erro mais comuns.


Desempenho de Teste e Validação
Os resultados obtidos (sem considerar pontuação) foram:

Modelo	Acurácia Total
Rede de Primeira Ordem	
92.55% 

Rede de Segunda Ordem	
92.60% 


Exportar para as Planilhas
Apesar da maior complexidade, o modelo de segunda ordem apresentou uma melhoria marginal de apenas 0.05% na acurácia total.

As matrizes de confusão mostram que ambos os modelos têm um desempenho muito bom para as tags mais frequentes, como NN (substantivo singular) e IN (preposição), mas encontram mais dificuldade em tags menos frequentes ou mais ambíguas. Por exemplo, o erro mais comum foi a confusão entre 

NNP (substantivo próprio) e NN (substantivo comum).


6. Discussão sobre Limitações e Dificuldades
Limitações e Dificuldades do Projeto

Aumento de Complexidade vs. Ganho de Performance: Como dito em sala na apresentação dos slides, uma das principais conclusões foi que, no contexto apresentado, o aumento da complexidade computacional e de implementação para um modelo de segunda ordem não se traduziu em um ganho significativo de performance. Isso sugere que para este dataset e tarefa, o contexto adicional captura pouca informação relevante que já não estivesse presente no modelo de primeira ordem.

Tratamento de Palavras Desconhecidas: Embora não explicitamente detalhado na apresentação, uma dificuldade inerente a modelos HMM é lidar com palavras que não estavam presentes no vocabulário de treino. A apresentação sugere a possibilidade de um tratamento específico para "palavras numéricas", indicando que outros tipos de palavras desconhecidas podem ter sido um desafio.


Ambiguidade de Tags: A análise de erros mostra que muitas confusões ocorrem entre classes gramaticais semanticamente próximas, como NN e NNP, ou diferentes formas verbais (VBD e VBN). Essa é uma limitação inerente à tarefa de POS tagging.


Limitações da Arquitetura Escolhida
Independência Local: O HMM assume que a observação atual (palavra) depende apenas do estado atual (tag) e que a transição de estado depende apenas de um número fixo de estados anteriores. Essa é uma simplificação forte que ignora dependências de longo alcance na linguagem.

Falta de Generalização para Palavras Raras: O modelo depende inteiramente das probabilidades de emissão aprendidas. Palavras raras ou ausentes no treino terão probabilidades de emissão nulas ou muito baixas para a maioria das tags, dificultando a classificação correta.

7. Sugestões de Melhorias
Com base nas conclusões e limitações identificadas, algumas melhorias poderiam ser implementadas:

Tratamento de Palavras Desconhecidas: Implementar uma heurística mais robusta para palavras não vistas no treino. Uma sugestão do próprio relatório é um tratamento específico para "palavras numéricas". Outras abordagens poderiam incluir a análise de sufixos ou prefixos para inferir a classe gramatical.

Suavização (Smoothing): Aplicar técnicas de suavização (como Laplace ou Good-Turing) às probabilidades de emissão e transição para lidar com eventos (palavra-tag ou tag-tag) não vistos no treino.

Modelos Híbridos: Combinar o HMM com outras abordagens, como modelos baseados em regras para casos específicos ou modelos de aprendizado profundo (como LSTMs ou Transformers), que são o estado da arte atual para a tarefa e podem capturar dependências de longo prazo de forma mais eficaz.

Engenharia de Features: Para o HMM, poderiam ser adicionadas features como "a palavra começa com letra maiúscula?" ou "a palavra contém hifens?" para melhorar a precisão, especialmente na distinção entre substantivos comuns e próprios.
