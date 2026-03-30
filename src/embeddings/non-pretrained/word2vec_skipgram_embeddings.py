import numpy as np 
import numpy.typing as npt 
from typing import TypeAlias
from gensim.models import Word2Vec 

# Type Alias 
Sentences: TypeAlias = list[str] # original string sentences 
TokenisedSentences: TypeAlias = list[list[str]] # sentence into words 
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32] # sentence embdding matrix 

class SkipGramEmbedder: 
    def __init__( 
            self, 
            vector_size: int = 100, # need to sync with other text methods' sentence size for  cnn input dim 
            window: int = 3, # maximum distance between the center word and a context word. 
            min_count: int = 1, # below count 1, the word is ignored. So rare vocabs are included. 
            epochs: int = 10, 
            seed: int = 42
    ): 
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed

        self.model = Word2Vec | None = None

    def fit(self, sentences: Sentences) -> 'SkipGramEmbedder': 
        tokenised = [s.lower().split() for s in sentences] # TokenisedSentences = list[list[str]]
        self.model = Word2Vec(
            sentences=tokenised, 
            vector_size=self.vector_size, 
            window=self.window, 
            sg=1, #skip-gram 
            min_count=self.min_count, 
            epochs=self.epochs, 
            seed=self.seed, 
            workers=1 # fix CPU thread num to 1 for reproducibilty. 
        )
        
        # self.wv = KeyedVectors(vector_size) 
        print(f"SkipGramEmbbeder trained. vocab size: {len(self.model.wv)}  dim of sentence embedding: {self.vector_size}") 
        
        return self  # update class object with fitted self.model 
    

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:  # (N, 100) matrix, N: number of sentences
        # mean-pool (word vectors >> single sentence vector)
        sentence_embeddings = []
        for s in sentences:
            tokens = s.lower().split() 
            word_embeddings_found = []
            for t in tokens: 
                if t not in self.model.wv: 
                    continue
                word_embeddings_found.append(self.model.wv[t])
            if not word_embeddings_found: 
                sentence_embeddings.append(np.zeros(self.vector_size, dtype=np.float32))
            else: 
                sentence_embeddings.append(np.mean(word_embeddings_found, axis=0).astype(np.float32))
        return np.array(sentence_embeddings, dtype=np.float32)
    
    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix: 
        return self.fit(sentences).transform(sentences)



    