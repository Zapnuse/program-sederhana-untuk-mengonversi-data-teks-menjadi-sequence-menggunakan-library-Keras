from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# sample text data
texts = ['This is an example sentence.',
         'Another example sentence.',
         'I love natural language processing.']

# define tokenizer
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

# convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)

# pad sequences to make them of equal length
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

print(padded_sequences)
