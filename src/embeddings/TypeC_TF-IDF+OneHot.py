# =========================================
# Tic-Tac-Toe: TF-IDF + OneHot Concatenation
# =========================================

# Input file columns:
# sentence_id, sentence, notation, n_moves, winner
# Example:
# sentence = "Wenjia is X and has taken center; Kim is O and has gone for top left corner."
# notation = "X:C O:TL"

# -----------------------------------------
# Step 0. Load dataset
# -----------------------------------------
data = load_csv("sentences_c.csv")

sentences  = data["sentence"]
notations  = data["notation"]
move_count = data["n_moves"]
winner_col = data["winner"]

# -----------------------------------------
# Step 1. Optional text normalisation
# -----------------------------------------
# Because your current sentence generator adds:
# - random player names
# - random verbs: "gone for", "taken"
# - random position phrases: "center" / "middle", etc.
#
# So before TF-IDF, normalise text to reduce noise.

def normalise_sentence(s):
    s = lowercase(s)

    # replace random names with stable tags
    # e.g. "Wenjia is X" -> "player_x is x"
    #      "Kim is O"    -> "player_o is o"
    s = replace_all_known_names_with_role_tags(s)

    # unify position paraphrases
    # e.g. "top left corner" -> "top_left"
    #      "up and left from center" -> "top_left"
    #      "middle" -> "center"
    s = canonicalise_position_phrases(s)

    # unify verbs
    # "gone for" / "taken" -> "occupies"
    s = canonicalise_move_verbs(s)

    return s

norm_sentences = [normalise_sentence(s) for s in sentences]

# -----------------------------------------
# Step 2. TF-IDF on sentence text
# -----------------------------------------
# baseline text feature
tfidf_vectorizer = TFIDF(
    ngram_range=(1,2),     # unigram + bigram
    min_df=1,
    lowercase=False        # already normalised
)

X_tfidf = tfidf_vectorizer.fit_transform(norm_sentences)
# shape: [N, V]

# -----------------------------------------
# Step 3. OneHot from board structure
# -----------------------------------------
# Best practice for your dataset:
# derive one-hot from notation / Board
# not from raw sentence tokens

POSITIONS = ["TL","TM","TR","ML","C","MR","BL","BM","BR"]
MARKS = ["X", "O", "EMPTY"]

def notation_to_board_onehot(notation):
    # parse notation like "X:C,TL O:BR"
    board = parse_notation(notation)    # uses your existing function

    # 9 cells × 3 states = 27 dims
    vec = zeros(27)

    for i, pos in enumerate(POSITIONS):
        if pos in board.x:
            state = "X"
        elif pos in board.o:
            state = "O"
        else:
            state = "EMPTY"

        # one-hot within each cell block
        # [X, O, EMPTY]
        offset = i * 3
        if state == "X":
            vec[offset + 0] = 1
        elif state == "O":
            vec[offset + 1] = 1
        else:
            vec[offset + 2] = 1

    return vec

X_board = stack([notation_to_board_onehot(n) for n in notations])
# shape: [N, 27]

# -----------------------------------------
# Step 4. Optional extra structured one-hot
# -----------------------------------------
# These are good "enhanced symbolic" features for Tic-Tac-Toe
# but optional, not mandatory.

def extra_structured_features(notation, n_moves, winner):
    board = parse_notation(notation)

    center_is_x = 1 if "C" in board.x else 0
    center_is_o = 1 if "C" in board.o else 0
    center_empty = 1 if ("C" not in board.x and "C" not in board.o) else 0

    x_wins = 1 if winner == "X" else 0
    o_wins = 1 if winner == "O" else 0
    no_winner = 1 if winner == "" else 0

    # move-count one-hot: 0..9
    move_vec = zeros(10)
    move_vec[n_moves] = 1

    return concat([
        [center_is_x, center_is_o, center_empty],
        [x_wins, o_wins, no_winner],
        move_vec
    ])

X_extra = stack([
    extra_structured_features(n, m, w)
    for n, m, w in zip(notations, move_count, winner_col)
])

# -----------------------------------------
# Step 5. Concatenate features
# -----------------------------------------
# option A: TF-IDF + board one-hot
X_text_struct = hstack([X_tfidf, X_board])

# option B: TF-IDF + board one-hot + extra symbolic features
X_text_struct_plus = hstack([X_tfidf, X_board, X_extra])

# -----------------------------------------
# Step 6. Train / eval split
# -----------------------------------------
train_idx, val_idx, test_idx = split_dataset(data, stratify=winner_col)

X_train = X_text_struct[train_idx]
X_val   = X_text_struct[val_idx]
X_test  = X_text_struct[test_idx]

# -----------------------------------------
# Step 7. Use as text-side target/embedding
# -----------------------------------------
# For a dual model:
# image_encoder(image) -> image_embedding
# text_head(X_text_struct) -> text_embedding
# optimise cosine similarity / contrastive loss
#
# or:
# image_encoder(image) -> predict fused text vector directly

for batch in train_loader:
    images = batch["image"]
    ids    = batch["sentence_id"]

    y_text = X_text_struct[ids]

    img_vec = image_encoder(images)
    pred_vec = projection_head(img_vec)

    loss = similarity_loss(pred_vec, y_text)
    update(loss)

# -----------------------------------------
# Step 8. Compare against baselines
# -----------------------------------------
# Baseline 1: TF-IDF only
# Baseline 2: board one-hot only
# Proposed : TF-IDF + board one-hot
# Optional : TF-IDF + board one-hot + extra features