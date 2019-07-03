
# coding: utf-8

# # Unilever encoder decoder RNN model - Synthetic 4
# 
# This version includes the following update:
# 
# - Calculating BLEU score for training and testing sets for different epochs
# - Allows to choose among:LSTM, GRU and NN for RNN1
# - Balance real commentaries
# - Add detection when a commentary is generated

# ## Model
# 
# <img src="./images/architecture2.png">

# In[1]:


# ---- Imports -----
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd, numpy as np


# In[2]:


from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, GRU
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint


# ## General parameters

# In[92]:


#Different combinations
# Syn=0 , epoch=100  (real data)
#GRU/GRU, NN/GRU, LSTM/LSTM, NN/LSTM

#Syn=1, epoch=50 (synthetic data)
#GRU/GRU, NN/GRU, LSTM/LSTM, NN/LSTM

Encoder_list=['GRU', 'NN', 'LSTM', 'NN'] * 2
Deccoder_list=['GRU', 'GRU', 'LSTM', 'LSTM'] * 2
Epochs_list = [100, 100, 100, 100, 50, 50, 50, 50]
Synth_list = [0,0,0,0,1,1,1,1]

for iteration in range(len(Synth_list)):

    IGNORE_NULL_COMMENTS = 0        #if =1 : Erase from the dataset null commentaries
    UNDERSAMPLE_NULL_COMMENTS = 1   #if =1 : undersamples the null commentaries to match the number of not null commentaries

    USE_SYNTHETIC = Synth_list[iteration]               #if =1 : use synthetic dataset for simulations
    SYNTHETIC_METHOD = 2            #MEthod 1 : only 1, Method 2: random numbers with peaks and also empty commentaries
    BALANCE_SYNTH = 1               #if =1 : balance synthetic commentaries dataset

    TRAIN_TEST_SETS = 0             #if =1 : use training and test sets, otherwise use % validation during training
    ENC_TYPE = Encoder_list[iteration]                 #encoder type, "NN" or same as decoder
    DEC_TYPE = Deccoder_list[iteration]               #type of decoder: "LSTM", "GRU"
    SHOW_MODEL_IMG = 0              #if =1 : shows the model graphically in the notebook

    batch_size = 20
    EPOCHS = Epochs_list[iteration]
    EPOCHS_STEPS = 1
    
    print("\n\n====================================================")
    print('Iteration', iteration, ' ', ENC_TYPE , '/', DEC_TYPE, ' Epochs=', EPOCHS)
    print("====================================================")

    # ## Reading data for the encoder
    # Get the data of Brand/territory variance matrix by month.
    # Not all brands are shipped to all territories. Therefore, filtering on a specific brand may return only some territories and not all of them. That is why it is important to get all territories and associate a zero to the ones who are missing. 

    # In[4]:


    #Import dataframes from pickle file (saved previously)
    from helper_save_load import load_from_pickle
    df_a, df_f, df_v = load_from_pickle("dataframes_Dollars.pickle")
    del df_a, df_f


    # ### Grouping territories

    # In[5]:


    territories = [territory for territory, values in df_v.groupby(['Territory']).groups.items()]
#     print(territories)
    print(len(territories),' territories')


    # In[6]:


    empty_df = pd.DataFrame(0.0, index=[0], columns=territories)
    varv_length = len(empty_df.columns)


    # ### Retrieving variance vector from brand and month
    # This function gets from the A/F dataset the variance by territory for a given month and brand. Multibrands are not supported, only `Brand_1` is considered in this study. The order of territories is the same as the `territories` vector. In case no data is available, a zero vector is returned

    # In[7]:


    #return pivot table for the required month in Millions of $
    def get_pivot_month_Territory_by_brand(month, brand, flatten=1):
        #Group by Territory and Brand
        df_group_Br_Tr = df_v[df_v['Brand'] == brand].groupby(['Brand', 'Territory']).sum()
        result = pd.pivot_table(df_group_Br_Tr, values=[month], index=['Brand'], 
                                columns=['Territory'], aggfunc=np.sum, fill_value=0) / 1e6
        result.columns = result.columns.droplevel()  #drop month level as there is only one month
        if len(result.index)>0:   #if no data is available, return a zero vector
        #Align with empty_df that includes all territories
            result = empty_df.append(result, sort=True).fillna(0)      
            result.drop(0, inplace=True)  #drop line 0 of empty_df
        else:
            result = empty_df 
        if (flatten==1): result = result.values.flatten()
        return (result)  

    get_pivot_month_Territory_by_brand('Jan_2018', '05-AXE SA Brand', 1)


    # ## Preparing data for the decoder
    # Get commentaries and dictionary from file

    # In[8]:


    #Import dataframes from pickle file (saved previously)
    from helper_save_load import load_from_pickle
    dfc, vocab, word_to_ix, ix_to_word = load_from_pickle("commentaries.pickle")
#     display(dfc.head(2))
#     print('index of word lcl:', word_to_ix['lcl'])
#     print('word at index 0:', ix_to_word[0])


    # In[9]:


    dfc['Comment_w'].replace('[NOC]', '[SOS] [EOS]', inplace=True)   #replace NoComment with StartOfSentence + EndOfSentence


    # In[10]:


    #Filter only non empty commentaries
    if IGNORE_NULL_COMMENTS:
        dfc = dfc[dfc['Comment_w']!='[SOS] [EOS]']


    # In[11]:


    commentaries = dfc['Comment_w']
#     for comment in commentaries:
#         print(comment)


    # In[12]:


    print(len(commentaries),' commentaries in total')
    print('-----------------------------------------')
    print(len(dfc[dfc['Comment_w']!='[SOS] [EOS]']),' Non NULL commentaries')
    print(len(dfc[dfc['Comment_w']=='[SOS] [EOS]']),' NULL commentaries')


    # ### Balancing data if empty commentaries are considered synthetic data
    # https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

    # In[13]:


    # random undersampling of empty commentaries
    # Class count
    if (IGNORE_NULL_COMMENTS==0) & UNDERSAMPLE_NULL_COMMENTS:
        class_0 = dfc['Comment_w']=='[SOS] [EOS]'
        class_1 = dfc['Comment_w']!='[SOS] [EOS]'

        # # Divide by class
        df_class_0 = dfc[class_0]
        df_class_1 = dfc[class_1]

        count_class_0 = len(df_class_0)
        count_class_1 = len(df_class_1)

        print(len(commentaries),' commentaries in total')
        print('-----------------------------------------')
        print(count_class_1,' Non NULL commentaries')
        print(count_class_0,' NULL commentaries')

        df_class_0_under = df_class_0.sample(count_class_1)
        df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)


        print('\nRandom under-sampling:')
        print('Null commentaries:     ', len(df_test_under[df_test_under['Comment_w']=='[SOS] [EOS]']))
        print('Not Null commentaries: ', len(df_test_under[df_test_under['Comment_w']!='[SOS] [EOS]']))

        dfc = df_test_under.sort_values(by='Num').reset_index(drop=True)   #Sort by original order
        commentaries = dfc['Comment_w']

    #     display(df_test_under)
    else:
        print('No undersampling Null comments')


    # In[14]:


    dfc.head(10)


    # ### Creating synthetic data

    # In[15]:


    # seed the pseudorandom number generator
    from numpy.random import seed
    from numpy.random import rand
    import time
#     get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt

    seed(int(time.time()))

    def random_vector(dimension, peak_indices):
        # seed random number generator


        x = rand(dimension)*.7 - 0.5
        #random peak
        a = rand(len(peak_indices)) + 1
        #random sign
        s = rand(1) - 0.5
        if s[0]<0:
            a = -a
        for i, index in enumerate(peak_indices):
            x[index] = a[i]
        return(x)



    x = random_vector(varv_length, [10, 15, 20])
#     plt.plot(x)


    # In[16]:


    from numpy.random import shuffle

    #### Creation on synthetic data, random numbers with random peaks, empty comments
    if (USE_SYNTHETIC==1) & (SYNTHETIC_METHOD==2):
        syn_var_X = []
        syn_commentaries = []
        n = varv_length

        for k in range(5984):    #no commentary
            v = random_vector(n, [])
            syn_var_X.append(v)
            comment = '[SOS] [EOS]'
            syn_commentaries.append(comment)


        for k in range(int(5984/34)):    #to oversample compared to 3 territories
            for i in range(n):
                v = random_vector(n, [i])
                syn_var_X.append(v)
                comment = '[SOS] driven only by territory %d [EOS]' % (i)
                syn_commentaries.append(comment)

        for k in range(int(5984/561)):    #to oversample compared to 3 territories
            for i in range(n-1):
                for j in range(i+1, n):
                    v = random_vector(n, [i, j])
                    syn_var_X.append(v)
                    comment = '[SOS] Both territories %d and %d drive the variance [EOS]' % (i, j)
                    syn_commentaries.append(comment)

        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    v = random_vector(n, [i, j, k])
                    syn_var_X.append(v)
                    comment = '[SOS] Territories %d, %d and %d are behind the observed variance [EOS]' % (i, j, k)
                    syn_commentaries.append(comment)


        syn_var_X = np.asarray(syn_var_X)  





        # prepare a randomly suffled sequence
        sequence = [i for i in range(len(syn_var_X))] 
        # randomly shuffle the sequence
        shuffle(sequence)
        syn_var_X = syn_var_X[sequence]
        syn_commentaries = np.asarray(syn_commentaries)[sequence]


    # In[17]:





    # In[18]:


    #### Creation on synthetic data, only 0 and ones, no empty comments
    if (USE_SYNTHETIC==1) & (SYNTHETIC_METHOD==1) :
        syn_var_X = []
        syn_commentaries = []
        n = varv_length

        for i in range(n):
            v = [0] * n
            v[i] = 1
            syn_var_X.append(v)
            comment = '[SOS] driven only by territory %d [EOS]' % (i)
            syn_commentaries.append(comment)

        for i in range(n-1):
            for j in range(i+1, n):
                v = [0] * n
                v[i] = 1
                v[j] = 1
                syn_var_X.append(v)
                comment = '[SOS] Both territories %d and %d drive the variance [EOS]' % (i, j)
                syn_commentaries.append(comment)

        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    v = [0] * n
                    v[i] = 1
                    v[j] = 1
                    v[k] = 1
                    syn_var_X.append(v)
                    comment = '[SOS] Territories %d, %d and %d are behind the observed variance [EOS]' % (i, j, k)
                    syn_commentaries.append(comment)


        syn_var_X = np.asarray(syn_var_X)  

        #Balancing dataset repeating first 34 rows by 170 times, and rows 34:34+561 by 10 times
        if BALANCE_SYNTH:
            c1 = syn_var_X[0:34]
            d1 = syn_commentaries[0:34]
            for i in range(169):
                c1 = np.concatenate((c1, syn_var_X[0:34]), axis=0)
                d1 = np.concatenate((d1, syn_commentaries[0:34]), axis=0)

            c2 = syn_var_X[34:34+561]
            d2 = syn_commentaries[34:34+561] 
            for i in range(9):
                c2 = np.concatenate((c2, syn_var_X[34:34+561]), axis=0) 
                d2 = np.concatenate((d2, syn_commentaries[34:34+561]), axis=0)

            syn_var_X = np.concatenate((syn_var_X, c1, c2), axis=0) 
            syn_commentaries = np.concatenate((syn_commentaries, d1, d2), axis=0)


    # In[19]:


    if USE_SYNTHETIC==1:
        commentaries = syn_commentaries
        var_X = syn_var_X


    # In[20]:


    num_comments = len(commentaries)


    # ### Tokenizing words (RNN2 inputs and outputs)

    # In[21]:


    # fit a tokenizer
    def create_tokenizer(lines):
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(lines)
        return tokenizer

    # max sentence length
    def max_length(lines):
        return max(len(line.split()) for line in lines)


    # In[22]:


    # prepare tokenizer for commentaries
    tokenizer = create_tokenizer(commentaries)

    for i,token in enumerate(tokenizer.word_index, start=0):
#         print("{} : {}". format(i,token))
        if i==16: break

    #example:
    # for num, name in enumerate(presidents, start=1):
    #     print("President {}: {}".format(num, name))


    # In[23]:


    #Calculate vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    #Calculate maximum length of commentaries
    com_length = max_length(commentaries)

    print('Vocabulary size (vocab_size):', vocab_size)
    print('Max length of commentary (com_length):', com_length)
    print('Number of commentaries :', len(commentaries))


    # In[24]:


    # encode and pad sequences
    def encode_sequences(tokenizer, length, lines):
        # integer encode sequences
        X = tokenizer.texts_to_sequences(lines)
        # pad sequences with 0 values
        X = pad_sequences(X, maxlen=length, padding='post')
        return X


    # In[25]:


    #Tokenizing all comments
    X = encode_sequences(tokenizer, com_length, commentaries)
#     print('X=\n', X)

    #Shifting tokenized words by 1 to predict next word in RNN2
    y_tokenized = np.zeros((len(commentaries), com_length), dtype='int')
    y_tokenized[:,0:com_length-1] = X[:,1:com_length]
#     print('\ny_tokenized=\n', y_tokenized)


    # ### One-hot encoding RNN2 outputs

    # In[26]:


    # one hot encode target sequence
    def encode_output(sequences, vocab_size):
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
        y = array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y

    y = encode_output(y_tokenized, vocab_size)
#     print('y[0,0,:] = ', y[0,0,:])


    # ### Preparing training data for encoder

    # In[27]:


    if USE_SYNTHETIC == 0:
        var_X = []
        for i, (index, row) in zip(range(len(commentaries)),dfc.iterrows()):
            vector = get_pivot_month_Territory_by_brand(row['Month_f'], row['Brand_1'], 0)
            var_X.append(vector.values.tolist()[0])
        #     if i<5:
        #         print(i, '**', index, '**', row['Month_f'], '**', row['Comment_w'], '**', row['Brand_1'])  
        #         print(trainX[i])
        #         display(vector)  

        var_X = np.asarray(var_X)
#         print(var_X)


    # ### Reshaping encoder data according to its type

    # In[28]:


    var_X.shape


    # In[29]:


    #Reshaping dataset for training
    def reshape_enc_data(data):
        if ENC_TYPE != "NN":
            return(data.reshape(num_comments, varv_length, 1))
        else:
            return (data)

    #Reshaping one vector for sampling
    def reshape_enc_data_1vector(data):
        data2 = np.asarray(data)
        if ENC_TYPE != "NN":
            return(data2.reshape(1, varv_length, 1))
        else:
            return (data)


    # In[30]:


    var_X = reshape_enc_data(var_X)


    # In[31]:


    var_X.shape


    # ### Create training and testing datasets

    # In[32]:


    if TRAIN_TEST_SETS:
        from sklearn.model_selection import train_test_split
        trainX, testX, var_trainX, var_testX, trainY, testY = train_test_split(
            X, var_X, y,  test_size=0.2)
        print ('testX: {}, var_testX: {}, testY: {}'.format(testX.shape, var_testX.shape, testY.shape))
    else:
        trainX = X
        testX = [[]]
        var_trainX = var_X
        var_testX = [[]]
        trainY = y
        testY_tokenized = [[]]        

    print ('trainX: {}, var_trainX: {}, trainY: {}'.format(trainX.shape, var_trainX.shape, trainY.shape))


    # In[33]:


#     print("trainX[0] = {}".format(trainX[0]))
#     print("var_trainX[0] = {}".format(var_trainX[0]))
#     print("trainY[0,0] = {}".format(trainY[0,0]))


    # In[34]:


#     if TRAIN_TEST_SETS:
#         print(testX[0])
#         print(var_testX[0])
#         print(testY[0])


    # ## 2. Decoder (RNN2)
    # 
    # Receives the variance vector that is concatenated with the embedding vector of the word, then is trained to predict the next word using the current word from the commentary of month i related to brand k. 
    # 
    # **It makes senses also to classify the commentaries in classes, such as: over delivery, driven by territory, orders phased, ...**
    # 
    # <img src="./images/decoder-arch.png">

    # In[35]:


    #Embedding size
    if USE_SYNTHETIC:
        embed_size = 10
    else:
        embed_size = 200
    # Preparing parameters
    if USE_SYNTHETIC:
        hidden_size = 8
    else:
        hidden_size = 256
    # Number of words in vocabulary
    src_vocab = vocab_size
    tar_vocab = src_vocab
    # Max length of input/ouput sentence
    src_timesteps = com_length #max(len(line.split()) for line in dfc['Comment_w'])
    tar_timesteps = src_timesteps
    # Length of variance vector
    varv_length = len(empty_df.columns)
    # Number of commentaries
    num_comments = len(commentaries)

    #Overview of the parameters calculated from dataset
    print('Embedding size: embed_size =', embed_size)
    print('Size of LSTM: hidden_size =', hidden_size)
    print('Commentaries vocabulary length: src_vocab =', src_vocab)
    print('Commentaries length (output): src_timesteps =', src_timesteps)
    print('Variance vector length: varv_length =', varv_length)
    print('Number of commentaries: num_comments =', num_comments)


    # encoder_input_data = (comment_num, variance pos, variance value)  => dimension (comment len, variance vector len, 1)
    # 
    # decoder_input_data = (comment num, word pos, word one-hot encoded vector) => (comments number, comment len, )
    # 
    # decoder_target_data = (comment num, word pos, word one-hot encoded vector) - words are shifted of 1

    # In[36]:


    #Uncomment and Run this is plot_model does not work
    import os
    os.environ["PATH"] =  os.environ["PATH"] + os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'


    # In[37]:


    def show_image(imagefile, w, h):
        print()
#         get_ipython().run_line_magic('pylab', 'inline')
#         import matplotlib.pyplot as plt
#         import matplotlib.image as mpimg
#         img=mpimg.imread(imagefile)
#         plt.figure(figsize=(w, h))
#         imgplot = plt.imshow(img);
#         plt.show();

    def show_model(model, w, h):
            print()
#         from keras.utils.vis_utils import plot_model
#         plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True);
#         show_image('model_plot.png', w, h)


    # In[38]:


    if ENC_TYPE =="NN":
        #Method 2 NN as encoder
        encoder_inputs = Input(shape=(varv_length,), name='var_vector_input')  #we feed encoder with one variance by timestep
        encoder = Dense(hidden_size, name='variance_Encoder_Dense')
        encoder_outputs = encoder(encoder_inputs)
    else:
        if DEC_TYPE == "LSTM":
            # Method 1 LSTM as encoder (DEC_TYPE)
            # Define an input sequence and process it.
            encoder_inputs = Input(shape=(varv_length, 1), name='var_vector_input')  #we feed encoder with one variance by timestep
            encoder = LSTM(hidden_size, return_state=True, name='variance_Encoder_LSTM')
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            # We discard `encoder_outputs` and only keep the states.
            encoder_states = [state_h, state_c]
        else:   #GRU
            encoder_inputs = Input(shape=(varv_length, 1), name='var_vector_input')  #we feed encoder with one variance by timestep
            encoder = GRU(hidden_size, return_state=True, name='variance_Encoder_GRU')
            encoder_outputs, state_h = encoder(encoder_inputs)
            # We discard `encoder_outputs` and only keep the states.
            encoder_states = state_h

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,), name='comment_input')    #src_timesteps #we feed the decoder with tokenized word
    word_Embedding = Embedding(src_vocab, embed_size,  mask_zero=True, name='word_Embedding')  #input_length=src_timesteps,
    embded_out = word_Embedding(decoder_inputs)

    if DEC_TYPE == "LSTM":
        #Method 1 LSTM
        encoder_states = [encoder_outputs, encoder_outputs]
        decoder = LSTM(hidden_size, return_sequences=True, return_state=True, name='comment_Decoder_LSTM')
        decoder_outputs, _, _ = decoder(embded_out, initial_state=encoder_states)
    else:
        #Method 2 GRU
        encoder_states = encoder_outputs
        decoder = GRU(hidden_size, return_sequences=True, return_state=True, name='comment_Decoder_GRU')
        decoder_outputs, _ = decoder(embded_out, initial_state=encoder_states)

    decoder_dense = Dense(src_vocab, activation='softmax', name='words_Output')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')      #rmsprop


    if SHOW_MODEL_IMG:
        print(model.summary())
        show_model(model, 20, 20)


    # #### Reshaping input/output vectors for LSTM
    # 
    # inputs are 3-dim with the following format: (samples, time steps, features)
    # - **Samples**. One sequence is one sample. A batch is comprised of one or more samples.
    # - **Time Steps**. One time step is one point of observation in the sample.
    # - **Features**. One feature is one observation at a time step.
    # 
    # 
    # 
    # **RNN1** :
    # - **samples**: number of comments: `num_comments` 
    # - **time steps**: number of territories: `varv_length`
    # - **Features**: one element per territory: `1`

    # **RNN2** :
    # - **samples**: number of comments: `num_comments` 
    # - **time steps**: max length of a commentary: `com_length`
    # - **Features**: token of each time step: `1`

    # ## Sampling model
    # 
    # 1) Encode input and retrieve initial decoder state
    # 
    # 2) Run one step of decoder with this initial state and a "start of sequence" token as target. Output will be the next target token.
    # 
    # 3) Repeat with the current target token and current states

    # In[39]:


    # Define sampling models

    encoder_model = Model(encoder_inputs, encoder_states)   #(input,output)
    decoder_state_input_h = Input(shape=(hidden_size,))
    embedding_out = word_Embedding(decoder_inputs)  

    if DEC_TYPE == "LSTM":
        decoder_state_input_c = Input(shape=(hidden_size,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder(embedding_out, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
    else:
        decoder_states_inputs = decoder_state_input_h
        decoder_outputs, state_h = decoder(embedding_out, initial_state=decoder_states_inputs)
        decoder_states = state_h

    decoder_outputs = decoder_dense(decoder_outputs)

    if DEC_TYPE == "LSTM":
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
    else:
        decoder_model = Model([decoder_inputs, decoder_states_inputs],
            [decoder_outputs, decoder_states])



    # In[40]:


    #shows encoder model
    if SHOW_MODEL_IMG:
        encoder_model.summary()
        show_model(encoder_model, 7, 7)


    # In[41]:


    #shows decoder model
    if SHOW_MODEL_IMG:
        decoder_model.summary()
        show_model(decoder_model, 20, 20)


    # In[42]:


    reverse_input_word_index = dict((i+1, word) for i, word in enumerate(tokenizer.word_index))
    reverse_input_word_index[0] = ''


    # ## Model Sampling model functions

    # In[43]:


    def remove_start_end_token(comment):
        return(comment.replace('[sos] ','').replace(' [eos]',''))


    # In[44]:


    def evaluate_model(indextrain, verbose=0):
        c = indextrain

        input_seq = var_trainX[c:c+1]    #To have 3 dim. One variance vector
        if USE_SYNTHETIC:
            commentary = commentaries[c].lower()
        else:
            commentary = commentaries[commentaries.index[c]].lower()

        comment = commentary.split()

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Populate the first character of target sequence with the start character.
        target_seq = np.zeros((1, 1))  
        target_seq[0, 0] = tokenizer.word_index['[sos]']   

        stop_condition = False
        i = 0
        decoded_sentence = []

        while not stop_condition:
            if DEC_TYPE == "LSTM":
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            else:
                output_tokens, h = decoder_model.predict([target_seq, states_value])
        #         print('max probability =',np.max(output_tokens[0, -1, :]))

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
    #         print('token of max =', sampled_token_index)

            sampled_word = reverse_input_word_index[sampled_token_index]
            decoded_sentence.append (sampled_word)
            proba = np.max(output_tokens[0, -1, :])

            # Exit condition: either hit max length or find stop character.
            if (sampled_word == '[eos]' or len(decoded_sentence) > src_timesteps):
                stop_condition = True

            if verbose:
                print('> %s \t (p = %0.3f)' % (sampled_word, proba))

                if proba<0.999:
                    nextwords = ''
                    token = sampled_token_index
                    for j in range(4):
                        output_tokens[0,0,token] = -1
                        token = np.argmax(output_tokens[0, -1, :])
                        proba = max(output_tokens[0, -1, :])
                        if proba<0.001: break
                        sampled_word = reverse_input_word_index[token]
                        nextwords += '%s (p = %0.3f)  ' % (sampled_word ,proba)
                    print(nextwords)   

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            if DEC_TYPE == "LSTM":
                states_value = [h, c]
            else:
                states_value = h

        return(commentary, '[sos] '+' '.join(decoded_sentence))


    # In[45]:


    def evaluate_model_direct(variance_vector, verbose=0):
        input_seq = variance_vector

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Populate the first character of target sequence with the start character.
        target_seq = np.zeros((1, 1))  
        target_seq[0, 0] = tokenizer.word_index['[sos]']   

        stop_condition = False
        i = 0
        decoded_sentence = []

        while not stop_condition:
            if DEC_TYPE == "LSTM":
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            else:
                output_tokens, h = decoder_model.predict([target_seq, states_value])
        #         print('max probability =',np.max(output_tokens[0, -1, :]))

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
    #         print(output_tokens)
        #         print('token of max =', sampled_token_index)

            sampled_word = reverse_input_word_index[sampled_token_index]
            decoded_sentence.append (sampled_word)
            proba = np.max(output_tokens[0, -1, :])

            # Exit condition: either hit max length or find stop character.
            if (sampled_word == '[eos]' or len(decoded_sentence) > src_timesteps):
                stop_condition = True

            if verbose:
                print('> %s \t (p = %0.3f)' % (sampled_word, proba))

            if verbose==2:
                if proba<0.999:
                    nextwords = ''
                    token = sampled_token_index
                    for j in range(4):
                        output_tokens[0,0,token] = -1
                        token = np.argmax(output_tokens[0, -1, :])
                        proba = max(output_tokens[0, -1, :])
                        if proba<0.001: break
                        sampled_word = reverse_input_word_index[token]
                        nextwords += '%s (p = %0.3f)  ' % (sampled_word ,proba)
                    print(nextwords)   


            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            if DEC_TYPE == "LSTM":
                states_value = [h, c]
            else:
                states_value = h

        return('[sos] '+' '.join(decoded_sentence))


    # ## Metrics evaluation (BLEU score + TP/TN/FP/FN) function

    # In[89]:


    def write_actual_predict_to_file(actualf, predictedf):
        filename = "result_synth_" + str(USE_SYNTHETIC)  + "_epochs_" + str(EPOCHS)                   + "_dec_" + DEC_TYPE + "_enc_" + ENC_TYPE + "_batchsize_" + str(batch_size)
        with open(filename+'_actual.txt', 'w') as f:
            for item in actualf:
                f.write("%s\n" % item)

        with open(filename+'_predicted.txt', 'w') as f:
            for item in predictedf:
                f.write("%s\n" % item)


    # In[91]:


    from nltk.translate.bleu_score import corpus_bleu

    #Positives and negatives are related to whether the model generate a commentary or not
    def calculate_metrics(verbose=0):
        TP=0; TN=0; FP=0; FN=0;
        actual, predicted = list(), list()
        actualf, predictedf = list(), list()
        for i in range(0,len(var_trainX)):
            source, prediction = evaluate_model(i)
            source = remove_start_end_token(source)
            prediction = remove_start_end_token(prediction)

            actualf.append(source.replace('[', '').replace(']', ''))
            predictedf.append(prediction.replace('[', '').replace(']', ''))        

            if (prediction == '[eos]'):
                if (source == '[eos]'):
                    TN += 1
    #                 print("TN")
                else:
                    FN += 1
    #                 print("FN")
            else:
                if (source == '[eos]'):
                    FP += 1
    #                 print("FP")
                else:
                    TP += 1  
    #                 print("TP")


            if (i < 10) & verbose:
                print('src =[%s]  \npred=[%s]\n' % (source, prediction))
            actual.append([source.split()])
            predicted.append(prediction.split())

        # calculate BLEU score
        BLEU1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
        BLEU2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        BLEU3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
        BLEU4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

        #write to files
        write_actual_predict_to_file(actualf, predictedf)



        if verbose:
            print('BLEU')
            print('| %0.3f  | %0.3f  | %0.3f  | %0.3f  |' % (BLEU1, BLEU2, BLEU3, BLEU4))
            print('METRICS')
            print('| TP     | TN     | FP     | FN     |')
            print('| %5d  | %5d  | %5d  | %5d  |' % (TP, TN, FP, FN))

        return([BLEU1, BLEU2, BLEU3, BLEU4, TP, TN, FP, FN])

#     calculate_metrics(1)


    # In[47]:


    #Dataframe to store the results
    columns = ['epoch', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'TP', 'TN', 'FP', 'FN' ]
    BLEU = pd.DataFrame(columns=columns)





    # ## Visualization functions

    # In[48]:


#     get_ipython().run_line_magic('matplotlib', 'inline')
#     import matplotlib.pyplot as plt

    def plot_loss(epoch_count, training_loss, test_loss):
        print()
#         # Visualize loss history
#         plt.plot(epoch_count, training_loss, 'b')
#         plt.plot(epoch_count, test_loss, 'r--')
#         plt.legend(['Training Loss', 'Test Loss'])
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Loss with '+DEC_TYPE)
#         plt.show();
#         # plt.savefig('loss.eps', format='eps', dpi=1000)


    # ## Training the model

    # In[49]:


    import time
    start = time. time()

    # model.load_weights(filename)
    training_loss = []
    test_loss = []
    filename = 'unilever-synthetic-3.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    for i in range(EPOCHS_STEPS):
        print('==========================')
        print("Step: {}/{}  Model".format(i+1, EPOCHS_STEPS))

        if TRAIN_TEST_SETS:
            history = model.fit([var_trainX, trainX], trainY, batch_size=batch_size, epochs=int(EPOCHS/EPOCHS_STEPS),
                                validation_data=([var_testX, testX], testY),
                                callbacks=[checkpoint], verbose=1) 
        else:
            history = model.fit([var_trainX, trainX], trainY, batch_size=batch_size, epochs=int(EPOCHS/EPOCHS_STEPS),
                                 validation_split=0.2,
                                callbacks=[checkpoint], verbose=1)

        # Get training and test loss histories
        training_loss = training_loss + history.history['loss']
        test_loss = test_loss + history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)       

        epoch = EPOCHS/EPOCHS_STEPS * (i+1);
        print()
        print('Metrics calculation')
        BLEUs = calculate_metrics(verbose=0)

        print('| %5d  | %0.3f  | %0.3f  | %0.3f  | %0.3f  | %5d  | %5d  | %5d  | %5d  |' 
              % (epoch, BLEUs[0], BLEUs[1], BLEUs[2], BLEUs[3], BLEUs[4], BLEUs[5], BLEUs[6], BLEUs[7]))

        BLEU.loc[i,'epoch'] =  epoch
        BLEU.loc[i,'BLEU1'] =  BLEUs[0]
        BLEU.loc[i,'BLEU2'] =  BLEUs[1]
        BLEU.loc[i,'BLEU3'] =  BLEUs[2]
        BLEU.loc[i,'BLEU4'] =  BLEUs[3]
        BLEU.loc[i,'TP'] =  BLEUs[4]
        BLEU.loc[i,'TN'] =  BLEUs[5]
        BLEU.loc[i,'FP'] =  BLEUs[6]
        BLEU.loc[i,'FN'] =  BLEUs[7]
#         display(BLEU)

    end = time.time()
    execution_time = end - start   #in seconds
    print('Execution time = {} seconds'.format(execution_time))


    # In[89]:


#     plot_loss(epoch_count, training_loss, test_loss)


    # In[90]:


    #Save commentaries with brands dataframe to pickle file

#     display(BLEU)

    from helper_save_load import save_to_pickle
    result_filename = "result_synth_" + str(USE_SYNTHETIC)  + "_epochs_" + str(EPOCHS)                   + "_dec_" + DEC_TYPE + "_enc_" + ENC_TYPE + "_batchsize_" + str(batch_size) +".pickle"
    save_to_pickle(result_filename, (BLEU, epoch_count, training_loss, test_loss, execution_time))
    print('Data saved to: '+result_filename)

