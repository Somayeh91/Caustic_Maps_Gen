import tensorflow as tf
import numpy as np
from keras import backend as K
import keras
from sf_fft import sf_fft


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_lc_function_sf_max(input):
    y = tf.numpy_function(lc_maker_sf_max, [input], tf.float32)
    return y

@keras.utils.register_keras_serializable()
def custom_loss_func(up=99, low=10):
    def custom_loglikelihood(y, y_hat):
        # batch_size = tf.shape(y)[0]
        # up_lim = tfp.stats.percentile(y, up, axis=[1, 2, 3])
        # low_lim = tfp.stats.percentile(y, low, axis=[1, 2, 3])
        # huber = tf.cast(0, tf.float32)
        # huber1 = tf.cast(0, tf.float32)

        # for i in range(batch_size):
        #   huber += tf.keras.losses.MeanSquaredError()(y[i][tf.math.greater(y[i], up_lim[i])],
        #                                       y_hat[i][tf.math.greater(y[i], up_lim[i])])
        #   huber1 += tf.keras.losses.MeanSquaredError()(y[tf.math.less(y[i], low_lim[i])],
        #                                       y_hat[tf.math.less(y[i], low_lim[i])])

        # huber = tf.keras.losses.MeanSquaredError()(
        #     y[tf.math.logical_and(tf.math.greater(y, up_lim), tf.math.less(y, low_lim))],
        #     y_hat[tf.math.logical_and(tf.math.greater(y, up_lim), tf.math.less(y, low_lim))]
        # )

        # huber = tf.keras.losses.BinaryCrossentropy()(y[tf.math.greater(y, 0.6)],
        #                                         y_hat[tf.math.greater(y, 0.6)])
        # if tf.math.is_nan(huber):
        #     huber = tf.cast(0, tf.float32)
        # # huber1 = tf.keras.losses.BinaryCrossentropy()(y[tf.math.less(y, 0.35)],
        # #                                         y_hat[tf.math.less(y, 0.35)])
        # if tf.math.is_nan(huber1):
        #     huber = tf.cast(0, tf.float32)
        kl_loss = tf.keras.losses.kl_divergence(y, y_hat)
        huber2 = tf.keras.losses.BinaryCrossentropy()(y, y_hat)
        # coeff = tf.Variable(1000)
        return tf.reduce_mean(huber2 + tf.math.abs(kl_loss))  # +\
        # tf.cast(huber/tf.cast(batch_size, tf.float32), tf.float32)+\
        # tf.cast(huber1/tf.cast(batch_size, tf.float32), tf.float32)
        # huber2 + tf.cast(2, tf.float32)*huber1)

    return custom_loglikelihood


def lc_loss_func(metric='mse', lc_loss_coeff=1):
    def lc_loglikelihood(y, y_hat):
        batch_size = tf.shape(y)[0]
        data = tf.concat([tf.reshape(y, [1, batch_size, tf.shape(y)[1], tf.shape(y)[1]]),
                          tf.reshape(y_hat, [1, batch_size, tf.shape(y)[1], tf.shape(y)[1]])],
                         0)
        if metric == 'mse':
            c = K.sum(tf_lc_function_mse(data))
        elif metric == 'sf_median':
            c = K.sum(tf_lc_function_sf_median(data))
        elif metric == 'sf_max':
            c = K.sum(tf_lc_function_sf_max(data))
        tf.cast(y_hat, tf.float32)
        tf.cast(y, tf.float32)
        tf.cast(c, tf.float32)
        lc_loss_coeff_tf = tf.constant(lc_loss_coeff, dtype=tf.float32)
        lc_loss = lc_loss_coeff_tf * c

        return lc_loss + tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits=True)(y, y_hat))

    return lc_loglikelihood


@keras.utils.register_keras_serializable()
def tweedie_loss_func(p):
    def tweedie_loglikelihood(y, y_hat):
        loss = - y * tf.pow(y_hat, 1 - p) / (1 - p) + \
               tf.pow(y_hat, 2 - p) / (2 - p)
        return tf.reduce_mean(loss)

    return tweedie_loglikelihood


from scipy import interpolate


def lc_maker_mse(map, distance=15, n_samp=100):
    mapp, map_hat = map[0], map[1]

    batch_size = mapp.shape[0]

    nxpix = 1000
    nypix = 1000
    factor = 1
    tot_r_e = 25
    xr = [-(tot_r_e / factor), (tot_r_e / factor)]
    yr = [-(tot_r_e / factor), (tot_r_e / factor)]

    x0 = xr[0]
    dx = xr[1] - x0  # map units in R_E

    mappix = 1000 / dx
    nsamp = n_samp
    distance = distance  # in maps units
    dt = (22 * 365) / 1.5  # from Kochanek (2004) for Q2237+0305: The quasar takes 22 years to go through 1.5 R_Ein
    tmax = distance * dt

    t = np.linspace(0, tmax, nsamp, dtype='int')
    metric = []

    for n in range(batch_size):
        c = 0
        # if origin==None:
        mmap = mapp[n]
        mmap_hat = map_hat[n]
        origin = [i // 2 for i in mmap.shape]

        for ang in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            # -------- determine the x and y pixel values
            rad = ang * np.pi / 180.0
            crad = np.cos(rad)
            srad = np.sin(rad)
            pixsz = dx / float(nxpix)  # size of one pixel in R_E
            drpix = distance / dx * float(nxpix) / float(nsamp)  # step's size in pixels
            ix = [i * drpix * crad + origin[0] for i in range(nsamp)]
            iy = [i * drpix * srad + origin[1] for i in range(nsamp)]

            # -------- interpolate onto light curve pixels
            #          (using interp2d!!!)
            x, y = np.arange(float(nxpix)), np.arange(float(nypix))
            mapint = interpolate.interp2d(x, y, mmap, kind='cubic')
            mapint2 = interpolate.interp2d(x, y, mmap_hat, kind='cubic')

            lc = np.array([mapint(i, j)[0] for i, j in zip(*[iy, ix])])
            lc2 = np.array([mapint2(i, j)[0] for i, j in zip(*[iy, ix])])

            c += np.sum((lc - lc2) ** 2)
        metric.append(c / 12.)
    return np.ndarray.astype(np.asarray(metric), np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_lc_function_mse(input):
    y = tf.numpy_function(lc_maker_mse, [input], tf.float32)
    return y


def lc_maker_sf_median(map, distance=15, n_samp=100):
    mapp, map_hat = map[0], map[1]

    batch_size = mapp.shape[0]
    #
    # nxpix = 1000
    # nypix = 1000
    # factor = 1
    # tot_r_e = 25
    # xr = [-(tot_r_e / factor), (tot_r_e / factor)]
    # yr = [-(tot_r_e / factor), (tot_r_e / factor)]
    #
    # x0 = xr[0]
    # dx = xr[1] - x0  # map units in R_E
    #
    # mappix = 1000 / dx
    # nsamp = n_samp
    # distance = distance  # in maps units
    # dt = (22 * 365) / 1.5  # from Kochanek (2004) for Q2237+0305: The quasar takes 22 years to go through 1.5 R_Ein
    # tmax = distance * dt
    #
    # t = np.linspace(0, tmax, nsamp, dtype='int')
    metric = []

    for n in range(batch_size):
        c = 0
        # if origin==None:
        mmap = mapp[n]
        mmap_hat = map_hat[n]
        # origin = [i // 2 for i in mmap.shape]

        # for ang in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        #     # -------- determine the x and y pixel values
        #     rad = ang * np.pi / 180.0
        #     crad = np.cos(rad)
        #     srad = np.sin(rad)
        #     pixsz = dx / float(nxpix)  # size of one pixel in R_E
        #     drpix = distance / dx * float(nxpix) / float(nsamp)  # step's size in pixels
        #     ix = [i * drpix * crad + origin[0] for i in range(nsamp)]
        #     iy = [i * drpix * srad + origin[1] for i in range(nsamp)]
        #
        #     # -------- interpolate onto light curve pixels
        #     #          (using interp2d!!!)
        #     x, y = np.arange(float(nxpix)), np.arange(float(nypix))
        #     mapint = interpolate.interp2d(x, y, mmap, kind='cubic')
        #     mapint2 = interpolate.interp2d(x, y, mmap_hat, kind='cubic')
        #
        #     lc = np.array([mapint(i, j)[0] for i, j in zip(*[iy, ix])])
        #     lc2 = np.array([mapint2(i, j)[0] for i, j in zip(*[iy, ix])])
        #
        #     c += np.median(np.abs(sf_fft(lc) - sf_fft(lc2)))
        c += np.median(np.abs(sf_fft(mmap) - sf_fft(mmap_hat)))

        metric.append(c / batch_size)
    return np.ndarray.astype(np.asarray(metric), np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_lc_function_sf_median(input):
    y = tf.numpy_function(lc_maker_sf_median, [input], tf.float32)
    return y


def lc_maker_sf_max(map, distance=15, n_samp=100):
    mapp, map_hat = map[0], map[1]

    batch_size = mapp.shape[0]

    nxpix = 1000
    nypix = 1000
    factor = 1
    tot_r_e = 25
    xr = [-(tot_r_e / factor), (tot_r_e / factor)]
    yr = [-(tot_r_e / factor), (tot_r_e / factor)]

    x0 = xr[0]
    dx = xr[1] - x0  # map units in R_E

    mappix = 1000 / dx
    nsamp = n_samp
    distance = distance  # in maps units
    dt = (22 * 365) / 1.5  # from Kochanek (2004) for Q2237+0305: The quasar takes 22 years to go through 1.5 R_Ein
    tmax = distance * dt

    t = np.linspace(0, tmax, nsamp, dtype='int')
    metric = []

    for n in range(batch_size):
        c = 0
        # if origin==None:
        mmap = mapp[n]
        mmap_hat = map_hat[n]
        origin = [i // 2 for i in mmap.shape]

        for ang in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            # -------- determine the x and y pixel values
            rad = ang * np.pi / 180.0
            crad = np.cos(rad)
            srad = np.sin(rad)
            pixsz = dx / float(nxpix)  # size of one pixel in R_E
            drpix = distance / dx * float(nxpix) / float(nsamp)  # step's size in pixels
            ix = [i * drpix * crad + origin[0] for i in range(nsamp)]
            iy = [i * drpix * srad + origin[1] for i in range(nsamp)]

            # -------- interpolate onto light curve pixels
            #          (using interp2d!!!)
            x, y = np.arange(float(nxpix)), np.arange(float(nypix))
            mapint = interpolate.interp2d(x, y, mmap, kind='cubic')
            mapint2 = interpolate.interp2d(x, y, mmap_hat, kind='cubic')

            lc = np.array([mapint(i, j)[0] for i, j in zip(*[iy, ix])])
            lc2 = np.array([mapint2(i, j)[0] for i, j in zip(*[iy, ix])])

            c += np.max(np.abs(sf_fft(lc) - sf_fft(lc2)))
            c += np.max(np.abs(sf_fft(mmap) - sf_fft(mmap_hat)))

        metric.append(c / 12.)
    return np.ndarray.astype(np.asarray(metric), np.float32)