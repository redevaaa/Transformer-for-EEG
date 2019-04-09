from io import BytesIO
import IPython.display
import numpy as np
import PIL.Image
from scipy.stats import truncnorm
import pandas as pd

class biggan_functions:

    def __init__(self, inputs = None, outputs = None):

        if inputs != None:
            # setting parameters
            self.input_z = inputs['z']
            self.input_y = inputs['y']
            self.input_trunc = inputs['truncation']

            self.dim_z = self.input_z.shape.as_list()[1]
            self.vocab_size = self.input_y.shape.as_list()[1]

        if outputs != None:
            self.output = outputs

        print('Set success')
        return


    # Primary functions
    def truncated_z_sample(self, batch_size, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, self.dim_z), random_state=state)
        return truncation * values

    def one_hot(self, index):
        index = np.asarray(index)
        if len(index.shape) == 0:
            index = np.asarray([index])
        assert len(index.shape) == 1
        num = index.shape[0]
        output = np.zeros((num, self.vocab_size), dtype=np.float32)
        output[np.arange(num), index] = 1
        return output

    def one_hot_if_needed(self, label):
        label = np.asarray(label)
        if len(label.shape) <= 1:
            label = self.one_hot(label)
        assert len(label.shape) == 2
        return label

    def sample(self, sess, noise, label, truncation=1., batch_size=8):
        noise = np.asarray(noise)
        label = np.asarray(label)
        num = noise.shape[0]
        if len(label.shape) == 0:
            label = np.asarray([label] * num)
        if label.shape[0] != num:
            raise ValueError('Got # noise samples ({}) != # label samples ({})'
                          .format(noise.shape[0], label.shape[0]))
        label = self.one_hot_if_needed(label)
        ims = []
        for batch_start in range(0, num, batch_size):
            s = slice(batch_start, min(num, batch_start + batch_size))
            feed_dict = {self.input_z: noise[s], self.input_y: label[s], self.input_trunc: truncation}
            ims.append(sess.run(self.output, feed_dict=feed_dict))
        ims = np.concatenate(ims, axis=0)
        assert ims.shape[0] == num
        ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
        ims = np.uint8(ims)
        return ims

    def interpolate(self, A, B, num_interps):
        alphas = np.linspace(0, 1, num_interps)
        if A.shape != B.shape:
            raise ValueError('A and B must have the same shape to interpolate.')
        return np.array([(1-a)*A + a*B for a in alphas])

    def imgrid(self, imarray, cols=5, pad=1):
        if imarray.dtype != np.uint8:
            raise ValueError('imgrid input imarray must be uint8')
        pad = int(pad)
        assert pad >= 0
        cols = int(cols)
        assert cols >= 1
        N, H, W, C = imarray.shape
        rows = int(np.ceil(N / float(cols)))
        batch_pad = rows * cols - N
        assert batch_pad >= 0
        post_pad = [batch_pad, pad, pad, 0]
        pad_arg = [[0, p] for p in post_pad]
        imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
        H += pad
        W += pad
        grid = (imarray
                .reshape(rows, cols, H, W, C)
                .transpose(0, 2, 1, 3, 4)
                .reshape(rows*H, cols*W, C))
        if pad:
            grid = grid[:-pad, :-pad]
        return grid

    def imshow(self, a, format='png', jpeg_fallback=True):
        a = np.asarray(a, dtype=np.uint8)
        str_file = BytesIO()
        PIL.Image.fromarray(a).save(str_file, format)
        im_data = str_file.getvalue()
        try:
            disp = IPython.display.display(IPython.display.Image(im_data))
        except IOError:
            if jpeg_fallback and format != 'jpeg':
                print ('Warning: image was too large to display in format "{}"; '
                  'trying jpeg instead.').format(format)
                return imshow(a, format='jpeg')
            else:
                raise
        return disp


    # Secondary functions
    def interpolate_and_shape(self, A, B, num_interps, num_samples):
        interps = self.interpolate(A, B, num_interps)
        return (interps.transpose(1, 0, *range(2, len(interps.shape)))
                     .reshape(num_samples * num_interps, *interps.shape[2:]))

    def return_images(self, sess, num_samples, num_interps, truncation,
                      noise_seed_A, category_A,
                      noise_seed_B, category_B):

        z_A, z_B = [self.truncated_z_sample(num_samples, truncation, noise_seed)
                    for noise_seed in [noise_seed_A, noise_seed_B]]
        y_A, y_B = [self.one_hot([int(category.split(')')[0])] * num_samples)
                    for category in [category_A, category_B]]

        z_interp = self.interpolate_and_shape(z_A, z_B, num_interps, num_samples)
        y_interp = self.interpolate_and_shape(y_A, y_B, num_interps, num_samples)

        ims = self.sample(sess, z_interp, y_interp, truncation=truncation)

        return ims, z_interp, y_interp


    # CSV functions
    def expand_latent(self, DF_LATENT_VALUES):

        # Expanding array values
        z_s = DF_LATENT_VALUES['z_interp'].apply(pd.Series) # expand df.tags into its own dataframe
        z_s = z_s.rename(columns = lambda x : 'z_interp_' + str(x)) # rename each variable is tags
        DF_LATENT_VALUES = pd.concat([DF_LATENT_VALUES[:], z_s[:]], axis=1)

        y_s = DF_LATENT_VALUES['y_interp'].apply(pd.Series) # expand df.tags into its own dataframe
        y_s = y_s.rename(columns = lambda x : 'y_interp_' + str(x)) # rename each variable is tags
        DF_LATENT_VALUES = pd.concat([DF_LATENT_VALUES[:], y_s[:]], axis=1)

        return DF_LATENT_VALUES

    def convert_latent_to_images(self, sess, Z_LENGTH, Y_LENGTH, DF_LATENT_VALUES_READ, TRUNCATION):

        ims = []
        # Looping through all latent variables
        for i in range(DF_LATENT_VALUES_READ.shape[0]):
            # Read back from dataframe
            df_to_consider = DF_LATENT_VALUES_READ[i : i+1]

            # Convert to list
            z_val = list(df_to_consider[df_to_consider.columns[3 : 3 + Z_LENGTH]].iloc[0])
            y_val = list(df_to_consider[df_to_consider.columns[3 + Z_LENGTH : 3 + Z_LENGTH + Y_LENGTH]].iloc[0])

            # Convert to image
            z_val, y_val = np.array(z_val).reshape(1, len(z_val)), np.array(y_val).reshape(1,len(y_val))
            imx = self.sample(sess, z_val, y_val, truncation=TRUNCATION)
            ims.append(imx)

        # Converting to numpy array
        ims = np.array(ims)
        ims = ims.reshape(ims.shape[0] * ims.shape[1], ims.shape[2], ims.shape[3], ims.shape[4])

        return ims

    def convert_latent_to_array(self, Z_LENGTH, Y_LENGTH, DF_LATENT_VALUES_READ):

        latent_z = np.zeros((DF_LATENT_VALUES_READ.shape[0], Z_LENGTH)) # latent vector
        latent_y = np.zeros((DF_LATENT_VALUES_READ.shape[0], Y_LENGTH)) # one hot encoding

        # Looping through all latent variables
        for i in range(DF_LATENT_VALUES_READ.shape[0]):
            # Read back from dataframe
            df_to_consider = DF_LATENT_VALUES_READ[i : i+1]

            # Convert to list
            z_val = list(df_to_consider[df_to_consider.columns[3 : 3 + Z_LENGTH]].iloc[0])
            y_val = list(df_to_consider[df_to_consider.columns[3 + Z_LENGTH : 3 + Z_LENGTH + Y_LENGTH]].iloc[0])

            # store in dictionary
            latent_z[i,:] = z_val
            latent_y[i,:] = y_val

        return latent_z, latent_y

    def convert_latent_to_images(self, sess, Z_LENGTH, Y_LENGTH, DF_LATENT_VALUES_READ, TRUNCATION):

        ims = []
        latent_z, latent_y = self.convert_latent_to_array(Z_LENGTH, Y_LENGTH, DF_LATENT_VALUES_READ)

        # Looping through all latent variables
        for i in range(DF_LATENT_VALUES_READ.shape[0]):
            # Convert to list
            z_val = latent_z[i]
            y_val = latent_y[i]

            # Convert to image
            z_val, y_val = np.array(z_val).reshape(1, len(z_val)), np.array(y_val).reshape(1,len(y_val))
            imx = self.sample(sess, z_val, y_val, truncation=TRUNCATION)
            ims.append(imx)

        # Converting to numpy array
        ims = np.array(ims)
        ims = ims.reshape(ims.shape[0] * ims.shape[1], ims.shape[2], ims.shape[3], ims.shape[4])

        return ims
