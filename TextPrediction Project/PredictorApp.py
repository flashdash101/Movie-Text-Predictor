import tkinter as tk
from tkinter import Entry, Listbox, Scrollbar, END
import tensorflow as tf
import tkinter
import pandas as pd
import numpy as np
import time
last_update_time = time.time()
from fuzzywuzzy import fuzz, process
import threading
#Initialize the global variables

last_update_time = time.time()
global key_press_timer
key_press_timer = None

#Reading the csv file to get the list of movie titles into a data frame.
df = pd.read_csv('movies5000.csv', usecols=['original_title'])

movie_name = df['original_title'].tolist()
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(movie_name)

root = tk.Tk()
root.geometry("400x240")
root.title("Movie Prediction")

# Create an Entry widget for user input
entry = Entry(root,width=50)
entry.place(x=100,y=5)
entry.pack()

# Create a Listbox to display predictions
predictions_listbox = Listbox(root, height=5, width=50)
predictions_listbox.pack()

# Create a scrollbar for the Listbox
scrollbar = Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
vocab_array = np.array(list(tokenizer.word_index.keys()))

#Load the model
model = tf.keras.models.load_model('TextPredict2')


# Connect the Listbox to the scrollbar
predictions_listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=predictions_listbox.yview)

# Lock for thread synchronization
thread_lock = threading.Lock()

#Uses a thread to make predictions
#Optimizes for smoother user experience
def make_pred_threaded(user_input, callback):
    def make_predictions():
        with thread_lock:
            predictions = make_pred(user_input, 5)  # Adjust the number of predictions as needed
            callback(predictions)

    thread = threading.Thread(target=make_predictions)
    thread.start()



def make_pred(text, n_words, beam_width=3, similarity_threshold=100):
    final_predictions = set()  # Initialize a set to store final predictions
    added_predictions = set()  # Keep track of added predictions to avoid duplicates

    # Split the input text into words
    input_words = text.split()

    # Find the closest match for each word in the input text
    closest_matches = []
    for word in input_words:
        closest_match = process.extractOne(word, vocab_array, scorer=fuzz.ratio)[0]
        closest_matches.append(closest_match)

    # Combine the closest matches for each word to form a new input text
    text = " ".join(closest_matches)

    # Initialize the beam with the best match
    beam = [(text, 0)]

    for i in range(n_words):
        new_beam = []

        for seq, score in beam:
            text_tokenize = tokenizer.texts_to_sequences([seq])[-1]
            text_padded = tf.keras.preprocessing.sequence.pad_sequences([text_tokenize], maxlen=14)

            # Predict the next word
            prediction = model.predict(text_padded)[0]

            # Get the top predicted word indices
            top_indices = np.argsort(prediction)[-beam_width:][::-1]

            # Get the corresponding words and probabilities from the vocabulary
            top_words = [vocab_array[index - 1] for index in top_indices]
            top_probs = [prediction[index] for index in top_indices]

            # Add the new sequences and their scores to the new beam
            new_beam.extend([(seq + " " + word, score + np.log(prob)) for word, prob in zip(top_words, top_probs)])

        # Keep only the top sequences in the beam
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

        # Check if any of the sequences in the beam are similar to an entry in the movie_name list
        for seq, score in beam:
            best_match, similarity = process.extractOne(seq, movie_name, scorer=fuzz.ratio)
            if similarity >= similarity_threshold and best_match not in added_predictions:
                final_predictions.add(best_match)
                added_predictions.add(best_match)

    if not final_predictions:  # If no similar sequences were found
        closest_matches = process.extractBests(text, movie_name, scorer=fuzz.ratio,score_cutoff=70)
        #Try the find the best match with a cutoff score of 70%
        closest_match = closest_matches[0][0] if closest_matches else movie_name[0]
        
        final_predictions.add(closest_match)
        #Set the closest match as the final prediction    
    return sorted(list(final_predictions))
#Return the sorted list of predictions


def update_predictions(event):
    global key_press_timer

    user_input = entry.get()
    
    if not user_input:
        predictions_listbox.delete(0, END)
        print('Detected no user inut')
        print('Deleting fields')
        return

    if key_press_timer is not None:
        # Cancel the previous timer if it exists
        
        root.after_cancel(key_press_timer)
    
    # Set a new timer to call update_predictions after a delay (e.g., 500 ms)
    key_press_timer = root.after(400, make_predictions_threaded, user_input)



def make_predictions_threaded(user_input):
    # Clear previous predictions
    predictions_listbox.delete(0, END)

    # Utilize threading to make predictions
    make_pred_threaded(user_input, update_ui)

def update_ui(predictions):
    # Run the function in the main thread and update the UI
    for prediction in predictions:
        predictions_listbox.insert(END, prediction)

# Bind the key release event to the update_predictions function
entry.bind("<KeyRelease>", update_predictions)



# Run the Tkinter main loop
root.mainloop()





