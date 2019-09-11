import pandas as pd
from clean import make_input_vocab, make_output_vocab, read_train, read_gold


def preprocessing(
        filename: str
):
    """
    Preprocess the input file, building the co-occurence matrix.
    :param filename: Path of the input file.
    :param max_vocab: Number of most frequent words to keep.
    :param min_count: Lower limit such that words which occur fewer than <int> times are discarded.
    :param window: Number of context words to the left and to the right.
    :return: The co-occurence matrix unpacked.
    """
    train_sentences = read_train()
    gold_vocab = read_gold()
    output_vocab = make_output_vocab()
    input_ = make_input_vocab()

    return train_sentences, gold_vocab, output_vocab, input_


def main(
        path_data: str,
        epochs: int,
        batch: int,
        vector_size: int,
        window: int,
        path_vectors: str,
        max_vocab: int,
        min_count: int,
        alpha: float,
        lr: float,
        x_max: int,
        save_mode: int,
):
    print("Preprocessing...")
    first_indices, second_indices, freq, word_index, word_counts = preprocessing(
        path_data, max_vocab, min_count, window
    )

    vocab_size = len(word_counts) + 1
    print("Vocab size:", vocab_size)
    print("Training...")
    model = train(
        first_indices=first_indices,
        second_indices=second_indices,
        frequencies=freq,
        epochs=epochs,
        batch=batch,
        vector_size=vector_size,
        vocab_size=vocab_size,
        alpha=alpha,
        lr=lr,
        x_max=x_max,
    )
    print("Saving vocab...")
    utils.save_vocab(config.VOCAB, word_counts)
    print("Saving embeddings file...")
    # path_folder = config.EMBEDDINGS.split("/")[0]
    # if not os.path.isdir(path_folder):
    #     os.mkdir(path_folder)
    utils.save_word2vec_format(
        model, path_vectors, word_index, vector_size, save_mode
    )


if __name__ == "__main__":
    preprocessing()
