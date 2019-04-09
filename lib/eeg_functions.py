import pandas as pd
import csv
import numpy as np

def process_eeg(DIR, FILENAMES, NUM_INTERVALS, NUM_SAMPLES_PER_INTERVAL, OPTION):

    NUM_IMAGES = NUM_INTERVALS * NUM_SAMPLES_PER_INTERVAL

    # dictionary of EEG chunks
    all_eeg_chunks = {}
    for i in range(NUM_IMAGES):
        all_eeg_chunks[i] = []

    # reading files and storing into dictionary
    for FILENAME in FILENAMES:
        # reading into dataframe
        df = pd.read_csv(DIR + FILENAME, delimiter = '\t', engine = 'python', header = None, index_col=None)

        # getting timestamps of markers
        start, end = df.loc[df[35]==2].index.values
        breaks = df.loc[df[35]==4].index.values
        image_indices = df.loc[df[35]==1].index.values
        colour_indices = df.loc[df[35]==3].index.values

        # getting chunks of eeg for each image
        for i in range(NUM_IMAGES):

            read_start, read_end = -1, -1

            if (OPTION == 1 or OPTION == 3):
                # find start and end marker for eeg
                read_start = image_indices[i]
                if i == NUM_IMAGES - 1: read_end = end
                else: image_indices[i+1]

            elif (OPTION == 2 or OPTION == 4):
                # find start and end marker for eeg
                read_start = colour_indices[i]
                if i == NUM_IMAGES - 1: read_end = end
                else: colour_indices[i+1]

            elif OPTION == 5:
                # find start and end marker for eeg
                read_start = image_indices[i]
                if i % NUM_SAMPLES_PER_INTERVAL == NUM_SAMPLES_PER_INTERVAL - 1:
                    read_end = breaks[int(i / NUM_SAMPLES_PER_INTERVAL)]
                else:
                    read_end = image_indices[i+1]

            # store df in chunks dictionary
            all_eeg_chunks[i].append(df[read_start : read_end])

    return all_eeg_chunks


def average_eeg(all_eeg_chunks, NUM_IMAGES):

    # dictionary for eeg stats
    stats_dict = {'differences': [], 'mins': [], 'maxs': []}

    # dictionary for extacting usable chunks
    chosen_eeg_chunks = {}

    # creating list for each dictionary index
    for i in range(NUM_IMAGES): chosen_eeg_chunks[i] = []

    # dictionary to store average chunks
    averaged_eeg_chunks = {}

    # finding stats from chunks
    for im in range(NUM_IMAGES):
        all_chunk_row_sizes = [chunk.shape[0] for chunk in all_eeg_chunks[im]]
        stats_dict['differences'].append(max(all_chunk_row_sizes) - min(all_chunk_row_sizes))
        stats_dict['mins'].append(min(all_chunk_row_sizes))
        stats_dict['maxs'].append(max(all_chunk_row_sizes))

    # selecting min
    CHUNK_SIZE_CHOSEN = min(stats_dict['mins'])

    # extracting chunks of chosen size
    for im in range(NUM_IMAGES):
        for chunk in all_eeg_chunks[im]:
            eeg_df = pd.DataFrame(chunk[:CHUNK_SIZE_CHOSEN], index = None)
            eeg_df = eeg_df.reset_index(drop = True)
            eeg_df = eeg_df[eeg_df.columns[:35]] # dropping marker and timestep data
            chosen_eeg_chunks[im].append(eeg_df)

    # averaging chosen chunks
    for im in range(NUM_IMAGES):
        avg_chunk = None
        for chunk in chosen_eeg_chunks[im]:
            if avg_chunk is None:
                avg_chunk = chunk
            else:
                avg_chunk = avg_chunk.add(chunk, fill_value=0)
        avg_chunk = avg_chunk/len(chosen_eeg_chunks[im])
        averaged_eeg_chunks[im] = avg_chunk

    return (CHUNK_SIZE_CHOSEN, averaged_eeg_chunks)


def write_eeg_to_file(averaged_eeg_chunks, NUM_IMAGES, EEG_RECORDINGS_DIR):

    # making one big dataframe from all chunks
    combined_df = pd.DataFrame()
    for im in range(NUM_IMAGES):
        combined_df = combined_df.append(averaged_eeg_chunks[im])

    # Storing as one dataframe
    EEG_FILENAME = EEG_RECORDINGS_DIR + 'averaged_eeg_chunks.csv'
    combined_df.to_csv(EEG_FILENAME)

    print('Saved to ' + EEG_FILENAME)
    return EEG_FILENAME


def read_eeg_from_file(EEG_FILENAME, NUM_IMAGES, CHUNK_SIZE_CHOSEN):

    # Reading dataframe
    combined_df = pd.read_csv(EEG_FILENAME, index_col = 0)

    # Separating image/chunk wise
    separated_chunks_array = np.zeros((NUM_IMAGES, CHUNK_SIZE_CHOSEN, combined_df.shape[1]))
    for im in range(NUM_IMAGES):
        read_start, read_end = im * CHUNK_SIZE_CHOSEN, (im + 1) * CHUNK_SIZE_CHOSEN
        separated_chunks_array[im, :, :] = combined_df[read_start : read_end].values

    return separated_chunks_array
