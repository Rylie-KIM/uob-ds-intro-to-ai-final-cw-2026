import re
import numpy as np
from gensim import downloader as api
from sklearn.metrics.pairwise import cosine_similarity


GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"
EMBEDDING_DIM = 100

_glove_model = None


def load_glove_model(model_name=GLOVE_MODEL_NAME):

    global _glove_model
    if _glove_model is None:
        print(f"Loading GloVe model: {model_name} ...")
        _glove_model = api.load(model_name)
        print("GloVe model loaded.")
    return _glove_model


def normalize_text(text):

    text = str(text)

    text = text.replace("X", "cross")
    text = text.replace("O", "nought")

    return text.lower().strip()


def tokenize_text(text):

    text = normalize_text(text)
    return re.findall(r"[a-z0-9]+", text)


def sentence_to_glove_vector(sentence, glove_model=None, embedding_dim=EMBEDDING_DIM):

    if glove_model is None:
        glove_model = load_glove_model()

    tokens = tokenize_text(sentence)

    if len(tokens) == 0:
        return np.zeros(embedding_dim), 0.0, []

    valid_vectors = []
    matched_tokens = []

    for token in tokens:
        if token in glove_model:
            valid_vectors.append(glove_model[token])
            matched_tokens.append(token)

    if len(valid_vectors) == 0:
        return np.zeros(embedding_dim), 0.0, []

    sentence_vector = np.mean(valid_vectors, axis=0)
    coverage = len(valid_vectors) / len(tokens)

    return sentence_vector, coverage, matched_tokens


def batch_sentences_to_glove_vectors(sentences, glove_model=None, embedding_dim=EMBEDDING_DIM):

    if glove_model is None:
        glove_model = load_glove_model()

    vectors = []
    coverages = []
    matched_tokens_list = []

    for sentence in sentences:
        vec, coverage, matched_tokens = sentence_to_glove_vector(
            sentence,
            glove_model=glove_model,
            embedding_dim=embedding_dim
        )
        vectors.append(vec)
        coverages.append(coverage)
        matched_tokens_list.append(matched_tokens)

    return np.array(vectors), coverages, matched_tokens_list


def glove_similarity(sentence1, sentence2, glove_model=None):

    if glove_model is None:
        glove_model = load_glove_model()

    vec1, _, _ = sentence_to_glove_vector(sentence1, glove_model=glove_model)
    vec2, _, _ = sentence_to_glove_vector(sentence2, glove_model=glove_model)

    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0

    sim = cosine_similarity([vec1], [vec2])[0][0]
    return float(sim)


def find_most_similar_sentence(query_sentence, candidate_sentences, glove_model=None):

    if glove_model is None:
        glove_model = load_glove_model()

    query_vec, _, _ = sentence_to_glove_vector(query_sentence, glove_model=glove_model)
    candidate_vecs, _, _ = batch_sentences_to_glove_vectors(candidate_sentences, glove_model=glove_model)

    if np.all(query_vec == 0):
        return None, 0.0, -1

    sims = cosine_similarity([query_vec], candidate_vecs)[0]
    best_index = int(np.argmax(sims))
    best_score = float(sims[best_index])
    best_sentence = candidate_sentences[best_index]

    return best_sentence, best_score, best_index


if __name__ == "__main__":
    model = load_glove_model()

    sentence_a = "X is at the top left"
    sentence_b = "cross is in the upper left corner"
    sentence_c = "O is at the bottom right"

    vec_a, coverage_a, tokens_a = sentence_to_glove_vector(sentence_a, model)
    vec_b, coverage_b, tokens_b = sentence_to_glove_vector(sentence_b, model)

    print("Sentence A:", sentence_a)
    print("Coverage A:", coverage_a)
    print("Matched tokens A:", tokens_a)
    print()

    print("Sentence B:", sentence_b)
    print("Coverage B:", coverage_b)
    print("Matched tokens B:", tokens_b)
    print()

    sim_ab = glove_similarity(sentence_a, sentence_b, model)
    sim_ac = glove_similarity(sentence_a, sentence_c, model)

    print("Similarity A-B:", sim_ab)
    print("Similarity A-C:", sim_ac)

    candidates = [
        "cross is in the upper left corner",
        "nought is in the center",
        "cross is at the bottom right"
    ]

    best_sentence, best_score, best_index = find_most_similar_sentence(sentence_a, candidates, model)

    print()
    print("Best matched sentence:", best_sentence)
    print("Best score:", best_score)
    print("Best index:", best_index)
