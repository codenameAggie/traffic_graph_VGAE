from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, GraphConvolutionDec
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.sigmoid,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
        # #
        # self.addedHidden1 = GraphConvolution(input_dim=1024,
        #                                      output_dim=FLAGS.hidden1,
        #                                      adj=self.adj,
        #                                      act=tf.nn.tanh,
        #                                      dropout=self.dropout,
        #                                      logging=self.logging)(self.hidden1)

        # self.addedHidden2 = GraphConvolution(input_dim=128,
        #                                output_dim=64,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.addedHidden1)
        # self.addedHidden3 = GraphConvolution(input_dim=64,
        #                                output_dim=32,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.addedHidden2)
        # self.addedHidden4 = GraphConvolution(input_dim=32,
        #                                output_dim=16,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.addedHidden3)
        # self.addedHidden5 = GraphConvolution(input_dim=16,
        #                                output_dim=FLAGS.hidden1,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.addedHidden4)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)
        #
        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          # act=tf.nn.relu,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        # self.z_log_std = 0.1 * tf.ones_like(self.z_mean)
                                                    # Changing the number of samples
        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        # self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
        #                               act=lambda x: x,
        #                               logging=self.logging)(self.z)

        # self.hidden2 = GraphConvolution(input_dim=FLAGS.hidden2,
        #                                output_dim=FLAGS.hidden1,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.z)
        # #
        # self.hidden3 = GraphConvolution(input_dim=FLAGS.hidden1,
        #                                output_dim=16,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.hidden2)
        #
        # self.hidden4 = GraphConvolution(input_dim=16,
        #                                output_dim=32,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.hidden3)
        # #
        # self.hidden5 = GraphConvolution(input_dim=32,
        #                                output_dim=1024,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.hidden4)

        self.reconstructions = GraphConvolutionDec(input_dim=FLAGS.hidden2,
                                       output_dim=self.input_dim,
                                       adj=self.adj,
                                       # act=tf.nn.relu,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.z)
        # self.reconstructions = GraphConvolutionSparse(input_dim=FLAGS.hidden2,
        #                                       output_dim=self.input_dim,
        #                                       adj=self.adj,
        #                                       features_nonzero=self.features_nonzero,
        #                                       act=tf.nn.relu,
        #                                       dropout=self.dropout,
        #                                       logging=self.logging)(self.z)
