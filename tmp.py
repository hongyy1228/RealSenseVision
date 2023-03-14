# BEGIN SOLUTION
def _convert(tree, leaves, is_word_end, out, start):
    pass


# END SOLUTION

def encode_tree(tree):
    """Converts a tree into subword token ids and a list of labeled spans.

    Args:
      tree: an nltk.tree.Tree object

    Returns:
      A tuple (ids, is_word_end, spans)
        ids: a list of token ids in the subword vocabulary
        is_word_end: a list with elements of type bool, where True indicates that
                     the word piece at that position is the last within its word.
        spans: a list of tuples of the form (start, end, label), where `start` is
               the position in ids where the span starts, `end` is the ending
               point in the span (exclusive), and `label` is a string indicating
               the syntactic label for the constituent.
    """
    tree = collapse_unary_strip_pos(tree)


# Implementation tip: it may help to look at encode_sentence, provided earlier
"""YOUR CODE HERE"""
    ids, is_word_end = encode_sentence(tree.leaves())
    start = []
    end = []
    idx = 0
    for i in range(len(tree.leaves())):
        start.append(j)
        while not is_word_end[idx]:
            idx += 1
        idx += 1
        end.append(idx)
    word_spans = []


    def get_spans(root, start_index, output):
        if isinstance(root, str):
            return 1
        if root.label() != 'TOP':
            output.append((start_index, start_index + len(root.leaves()) - 1, root.label()))
        for child in root:
            start_index += get_spans(child, start_index, output)
        return len(root.leaves())


    get_spans(tree, 0, word_spans)
    spans = [(word_starts[ws[0]], word_ends[ws[1]], ws[2]) for ws in word_spans]
    return ids, is_word_end, spans

# BEGIN SOLUTION

# END SOLUTION