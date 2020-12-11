import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# learningRateStep = 0.1

class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, index_non, count_non, index_zero, count_zero):
        self.preds_sub = preds
        self.labels_sub = labels

        # self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.masked_true = tf.multiply(labels, index_non)
        self.masked_training = tf.multiply(preds, index_non)

        # self.masked_true = labels
        # self.masked_training = preds

        # self.cost = norm * tf.losses.mean_squared_error(tf.math.subtract(x=masked_true, y=masked_training))
        # self.cost = (norm) * tf.reduce_mean(tf.square(tf.math.subtract(x=self.masked_true, y=self.masked_training)))
        self.cost = (1/count_non) * (norm) * tf.reduce_sum(tf.square(tf.math.subtract(x=self.masked_true, y=self.masked_training)))

        self.train_mse = self.cost / norm

        self.train_mae = (1/count_non) * tf.reduce_sum(tf.abs(self.masked_true - self.masked_training))

        # FLAGS.learning_rate *= learningRateStep

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = -1 * (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))

        # self.kl = -1 * (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + tf.log(tf.square(model.z_log_std) + 1e-5) - tf.square(model.z_mean) -
        #                   tf.square(model.z_log_std), 1))
        self.cost = self.log_lik + 0.1 * self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
        #                                    tf.cast(labels_sub, tf.int32))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.train_mape = (100/count_non) * tf.reduce_sum(tf.math.divide(tf.abs(self.masked_true - self.masked_training), (self.masked_true + 1e-10)))

        self.masked_true_test = tf.multiply(labels, index_zero)
        self.masked_training_test = tf.multiply(preds, index_zero)
        #

        self.test_mae = (1/count_zero) * tf.reduce_sum(tf.abs(self.masked_true_test - self.masked_training_test))

        self.test_mape = (100/count_zero) * tf.reduce_sum(tf.math.divide(tf.abs(self.masked_true_test - self.masked_training_test), (self.masked_true_test + 1e-10)))

        self.TFMAPE = tf.reduce_mean(tf.keras.losses.mean_absolute_percentage_error(y_pred=self.preds_sub, y_true=self.labels_sub))

        # self.accuracy = (1/count_zero) * norm * tf.reduce_sum(tf.square(tf.math.subtract(x=masked_true, y=masked_training)))
        self.accuracy = (1/(count_zero)) * tf.reduce_sum(tf.square(tf.math.subtract(x=self.masked_true_test, y=self.masked_training_test)))
        # self.accuracy = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(preds,labels),(labels + 1e-10))))

