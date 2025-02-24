"""
Entropy calculation module (Shannon, Lempel-Ziv, Plug-In, Konto)
"""

import math
from typing import Union

import numpy as np
from numba import njit
from numba.typed import List
from numba import types


def get_shannon_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, page 263-264.

    Get Shannon entropy from message

    :param message: (str) Encoded message
    :return: (float) Shannon entropy
    """
    exr = {}
    entropy = 0
    for each in message:
        try:
            exr[each] += 1
        except KeyError:
            exr[each] = 1
    textlen = len(message)
    for value in exr.values():
        freq = 1.0 * value / textlen
        entropy += freq * math.log(freq) / math.log(2)
    entropy *= -1
    return entropy


def get_lempel_ziv_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.2, page 266.

    Get Lempel-Ziv entropy estimate

    :param message: (str) Encoded message
    :return: (float) Lempel-Ziv entropy
    """
    i, lib = 1, [message[0]]
    while i < len(message):
        for j in range(i, len(message)):
            message_ = message[i:j + 1]
            if message_ not in lib:
                lib.append(message_)
                break
        i = j + 1
    return len(lib) / len(message)


def _prob_mass_function(message: str, word_length: int) -> dict:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 266.

    Compute probability mass function for a one-dim discete rv

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (dict) Dict of pmf for each word from message
    """
    lib = {}
    if not isinstance(message, str):
        message = ''.join(map(str, message))
    for i in range(word_length, len(message)):
        message_ = message[i - word_length:i]
        if message_ not in lib:
            lib[message_] = [i - word_length]
        else:
            lib[message_] = lib[message_] + [i - word_length]
    pmf = float(len(message) - word_length)
    pmf = {i: len(lib[i]) / pmf for i in lib}
    return pmf


def get_plug_in_entropy(message: str, word_length: int = None) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 265.

    Get Plug-in entropy estimator

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (float) Plug-in entropy
    """
    if word_length is None:
        word_length = 1
    pmf = _prob_mass_function(message, word_length)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length
    return out


@njit()
def _match_length(message: str, start_index: int, window: int) -> Union[int, str]:    # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 18.3, page 267.

    Function That Computes the Length of the Longest Match

    :param message: (str or array) Encoded message
    :param start_index: (int) Start index for search
    :param window: (int) Window length
    :return: (int, str) Match length and matched string
    """
    # Maximum matched length+1, with overlap.
    sub_str = ''
    for length in range(window):
        msg1 = message[start_index: start_index + length + 1]
        for j in range(start_index - window, start_index):
            msg0 = message[j: j + length + 1]
            if len(msg1) != len(msg0):
                continue
            if msg1 == msg0:
                sub_str = msg1
                break  # Search for higher l.
    return len(sub_str) + 1, sub_str  # Matched length + 1


def get_konto_entropy(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Implementations of Algorithms Discussed in Gao et al.[2008]

    Get Kontoyiannis entropy

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """
    out = {
        'h': 0,
        'r': 0,
        'num': 0,
        'sum': 0,
        'sub_str': []
    }
    if window <= 0:
        points = range(1, len(message) // 2 + 1)
    else:
        window = min(window, len(message) // 2)
        points = range(window, len(message) - window + 1)
    for i in points:
        if window <= 0:
            length, msg_ = _match_length(message, i, i)
            out['sum'] += np.log2(i + 1) / length  # To avoid Doeblin condition
        else:
            length, msg_ = _match_length(message, i, window)
            out['sum'] += np.log2(window + 1) / length  # To avoid Doeblin condition
        out['sub_str'].append(msg_)
        out['num'] += 1
    try:
        out['h'] = out['sum'] / out['num']
    except ZeroDivisionError:
        out['h'] = 0
    out['r'] = 1 - out['h'] / (np.log2(len(message)) if np.log2(len(message)) > 0 else 1)  # Redundancy, 0<=r<=1
    return out['h']



@njit
def _get_konto_entropy_nb(message, window):
    """
    Compiled helper function that computes Kontoyiannis entropy.
    This version iterates over the message entirely within Numba,
    thus avoiding the overhead of repeated Python calls.
    """
    n = len(message)
    total = 0.0
    count = 0
    if window <= 0:
        for i in range(1, n // 2 + 1):
            length, _ = _match_length(message, i, i)
            #total += math.log2(i + 1) / length  # To avoid Doeblin condition.
            total += np.log2(i + 1) / length
            count += 1
    else:
        win = window if window < n // 2 else n // 2
        for i in range(win, n - win + 1):
            length, _ = _match_length(message, i, win)
            #total += math.log2(win + 1) / length  # To avoid Doeblin condition.
            total += np.log2(i + 1) / length
            count += 1
    if count == 0:
        return 0.0
    return total / count


def get_konto_entropy_nb(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Get Kontoyiannis entropy using a fully compiled Numba implementation for speedup.
    This improved version removes the overhead of per-iteration Python calls by delegating
    the entire loop to the compiled helper function _get_konto_entropy_nb.

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """
    return _get_konto_entropy_nb(message, window)









def str_to_int_array(message: str) -> np.ndarray:
    """
    Convert a string into a numeric array of Unicode code points.
    """
    return np.array([ord(c) for c in message], dtype=np.int64)


@njit
def get_lempel_ziv_entropy_numba(arr):
    """
    Fully compiled Numba version of the Lempel-Ziv entropy estimator.
    Operates on a numeric array (e.g. an array of Unicode code points).

    :param arr: (np.ndarray) Numeric array representing the message.
    :return: (float) Lempel-Ziv entropy estimate.
    """
    n = arr.shape[0]
    # Create a typed list to store phrases, each as a tuple (start_index, length)
    phrases = List()
    phrases.append((0, 1))  # the first character is our first phrase

    i = 1
    while i < n:
        for j in range(i, n):
            candidate_length = j - i + 1
            candidate_found = False
            # Iterate over all stored phrases
            for k in range(len(phrases)):
                phrase_start, phrase_length = phrases[k]
                if phrase_length == candidate_length:
                    match = True
                    for x in range(candidate_length):
                        if arr[phrase_start + x] != arr[i + x]:
                            match = False
                            break
                    if match:
                        candidate_found = True
                        break
            if not candidate_found:
                phrases.append((i, candidate_length))
                i = j + 1
                break
        else:
            i = n

    return len(phrases) / n


def get_lempel_ziv_entropy_numba_wrapper(message: str) -> float:
    """
    Convert a string message to a numeric array and compute its Lempel-Ziv entropy
    using the fully compiled Numba version.
    
    :param message: (str) Encoded message.
    :return: (float) Lempel-Ziv entropy estimate.
    """
    arr = str_to_int_array(message)
    return get_lempel_ziv_entropy_numba(arr)


def get_lempel_ziv_entropy_fast(message: str) -> float:
    """
    Optimized Lempel-Ziv entropy estimator using a set for faster membership tests.
    
    :param message: (str) Encoded message (e.g., a string of symbols like 'abracadabra')
    :return: (float) Lempel-Ziv entropy estimate, computed as the number of distinct phrases divided by the message length.
    """
    if not message:
        return 0.0

    # Start with the first character already included as the first phrase.
    i = 1
    phrases = {message[0]}
    
    # Loop until we reach the end of the message.
    while i < len(message):
        # For each position i, extend the substring until a new phrase is found.
        for j in range(i, len(message)):
            phrase = message[i:j + 1]
            # If this substring hasn't been seen before, add it to the set and break.
            if phrase not in phrases:
                phrases.add(phrase)
                break
        # Advance i to the next position after the candidate substring.
        i = j + 1
    
    # Return the entropy estimate: number of unique phrases divided by the total message length.
    return len(phrases) / len(message)


if __name__ == '__main__':
    import timeit
    # Create a test message by repeating a base string.
    message = "1001001" * 100 # "your_test_string_here" "10010"

    # Make sure to use the appropriate function wrappers.
    # t_shannon = timeit.timeit(lambda: get_shannon_entropy(message), number=10)
    # t_plugin = timeit.timeit(lambda: get_plug_in_entropy(message), number=10)

    # t_original = timeit.timeit(lambda: get_lempel_ziv_entropy(message), number=10)
    # t_fast = timeit.timeit(lambda: get_lempel_ziv_entropy_fast(message), number=10)
    # t_numba = timeit.timeit(lambda: get_lempel_ziv_entropy_numba_wrapper(message), number=10)

    t_konto = timeit.timeit(lambda: get_konto_entropy(message, window=100), number=10)
    t_kontonb = timeit.timeit(lambda: get_konto_entropy_nb(message, window=100), number=10)

    # print(f"shannon: {t_shannon:.6f} sec, plugin: {t_plugin:.6f} sec, Original: {t_original:.6f} sec, fast: {t_fast:.6f} sec, numba: {t_numba:.6f} sec, konto: {t_konto:.6f} sec, kontonb: {t_kontonb:.6f} sec")
    #print(f"Original: {t_original:.6f} sec, fast: {t_fast:.6f} sec, numba: {t_numba:.6f} sec, konto: {t_konto:.6f} sec, kontonb: {t_kontonb:.6f} sec")
    #print(f"Original: {t_original:.6f} sec, fast: {t_fast:.6f} sec, konto: {t_konto:.6f} sec, kontonb: {t_kontonb:.6f} sec")
    print(f"konto: {t_konto:.6f} sec, kontonb: {t_kontonb:.6f} sec")

    # print("Shannon entropy:", get_shannon_entropy(message))
    # print("Lempel-Ziv entropy:", get_lempel_ziv_entropy(message))
    # print("Plug-In entropy:", get_plug_in_entropy(message))
    # print("Kontoyiannis entropy (improved):", get_konto_entropy(message))
    # print("Lempel-Ziv entropy (full numba):", get_lempel_ziv_entropy_numba_wrapper(message))

"""

<Konto>
I did not modify the original code, but instead added get_konto_entropy_nb and _get_konto_entropy_nb
-> _match_length are common btw the original and revised version

<LZ>
numba is somehow slow so using fast version makes sense for the time being at least

++++++++++++++++

https://chatgpt.com/share/67baf1d3-48a0-8000-953c-45b455586d7f
Often, it's more meaningful to analyze relative changes or correlations with known data features rather than comparing raw numbers.




https://chatgpt.com/share/67babc22-cc08-8000-ad6e-c45ff07193fe

There are cases, however, where one might outperform the other:

Numba Advantages:
If your data is already in numeric form or can be batched effectively, 
a fully Numba-compiled version (avoiding any Python-level operations) can sometimes offer further speed-ups, 
especially on very large inputs or when the algorithm is restructured to minimize Python overhead entirely.

Set-Based Advantages:
On the other hand, if the input is naturally a string and the set operations are highly efficient (implemented in C under the hood), 
the set-based approach might be simpler and just as effective."""