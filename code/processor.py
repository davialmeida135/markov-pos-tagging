import os
"""
Classe para parsear o dataset Penn Treebank.
O dataset é composto por arquivos de texto onde cada linha contém uma palavra seguida de sua tag.
Cada palavra e tag são separadas por um underscore (_).
A função processa os arquivos e organiza as sentenças em listas de tuplas (palavra, tag).
"""
class PennTreebankProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train = None
        self.dev = None
        self.test = None

    def _get_files(self):
        files = []
        for root, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files
    
    def _parse_file(self, file_path):
        
        with open(file_path, "r", encoding="utf-8") as f:
            sentences = []
            
            for line in f:
                current_sentence = []
                line = line.strip()
                
                #print(line)
                if not line:
                    continue
                    
                tokens = line.split()
                
                for token in tokens:
                    if "_" not in token:
                        continue
                    word, tag = token.rsplit('_', 1)
                    if word and tag:
                        current_sentence.append((word, tag))
                sentences.append(current_sentence)
                
            return sentences
    
    def process(self):
        files = self._get_files()
        for file_path in files:
            if 'train' in file_path:
                self.train = self._parse_file(file_path)
            elif 'dev' in file_path:
                self.dev = self._parse_file(file_path)
            elif 'test' in file_path:
                self.test = self._parse_file(file_path)
    

if __name__ == "__main__":
    data_dir = 'data/raw'
    processor = PennTreebankProcessor(data_dir)
    processor.process()

    print(processor.train[0])