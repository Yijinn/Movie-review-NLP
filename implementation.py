import tensorflow as tf
import string

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """

    processed_review = review.lower()

    #some <br /> is in text
    processed_review = processed_review.replace("<br />", " ")

    #Remove all punctuation except "-", as words like 3-year-old should be considered as one word
    for punc in string.punctuation.replace("-",""):
      processed_review = processed_review.replace(punc, " ")

    # Remove stop words
    processed_review = [word for word in processed_review.split() if word not in stop_words]
    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    hidden_unit = 128

    #Parameter settings and decision making details can be found in report.

    input_data = tf.placeholder(name="input_data", dtype=tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE])
    labels = tf.placeholder(name="labels", dtype=tf.float32, shape=[BATCH_SIZE,2])
    dropout_keep_prob = tf.placeholder_with_default(0.8, shape=(), name='dropout_keep_prob')

    FW_lstmCell = tf.nn.rnn_cell.LSTMCell(hidden_unit, initializer=tf.initializers.orthogonal, forget_bias=1.0)
    BW_lstmCell = tf.nn.rnn_cell.LSTMCell(hidden_unit, initializer=tf.initializers.orthogonal, forget_bias=1.0)

    FW_lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=FW_lstmCell, output_keep_prob=dropout_keep_prob)
    BW_lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=BW_lstmCell, output_keep_prob=dropout_keep_prob)
        
    (FW_outputs, BW_outputs),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=FW_lstmCell, cell_bw=BW_lstmCell, dtype=tf.float32, inputs=input_data)


    lastOutput = tf.concat((FW_outputs, BW_outputs),2)
    lastOutput = lastOutput[:,-1,:]

    logits = tf.layers.dense(lastOutput,2)



    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits = logits), name="loss")

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
