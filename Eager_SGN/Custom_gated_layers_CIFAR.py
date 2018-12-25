#%%
""" # THIS CODE:
# Runs either mnist or cifar depending on which is imported "import_mnist_utils" or "import_cifar_utils"
# It runs them through an SGN model once. It studies alternative beteween learning discriminatory task and generative task
# where the two models share an embedding layer. It displays the flow and the gate averages from the gated layer.

"""

__file__ = r'c:\Users\Ali Hummos\OneDrive\Code\ai\Custom_gated_layers_CIFAR.py'
import sys
import os
sys.path.append(os.path.dirname(__file__))
   
# from import_cifar_utils import *
from import_mnist_utils import *


if not os.path.exists('weights/'):
    os.makedirs('weights/')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This removes the warning of unsued CPU extensions for multiply-and-accumulate (MAC) speed-up.
# import dataset  # download dataset.py file
# dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)
class parameters():
    def __init__(self):
        self.batch_size = 32 # of course 512 lead to OOM
        self.learning_rate = 0.1
        # self.train_epochs = 5
        self.train_steps = 12001
        self.report_every = 1000
        self.embedding_layer_size = 100
        self.output_layer_size = 10
        self.embedding_size = self.embedding_layer_size+ self.output_layer_size


param = parameters()

def random_batch(x_data, y_data, num_of_training_examples, batch_size=32):
    """
    Create a random batch of training-data.

    :param batch_size: Number of images in the batch.
    :return: 3 numpy arrays (x, y, y_cls)
    """
    num_train = num_of_training_examples
    # Create a random index into the training-set.
    idx = np.random.randint(low=0, high=num_train, size=batch_size)
    # Use the index to lookup random training-data.
    x_batch = x_data[idx]
    y_batch = y_data[idx]
    # y_batch_cls = self.y_train_cls[idx]

    return x_batch, y_batch

def disp_grads(grads, variables):
    for grad, vari in zip(grads, variables):
        if grad is not None:
            rg = grad.numpy().reshape(-1)
            nz = np.where(rg==0)[0].shape[0]/rg.shape[0]
            grad_name = vari.name
            print('norm:{:1.1e} \tvar:{:1.3e} \tper 0s:{:.0%}\t{}'.format(np.linalg.norm(rg), np.var(rg), nz, grad_name))
        else:
            grad_name = vari.name
            print('This had None gradient ##### \t{}'.format(grad_name))


def loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))

def test_accuracy(model, x_test, y_test):
    logits = model(x_test)
    predictions = tf.argmax(logits, 1)
    labels = tf.argmax(y_test, 1)

    accu = predictions== labels
    
    return accu*100


def eval_model(images, true_labels, sample_size):
    #Testing one run
    n_test = 100
    accu = 0
    loss_batch = 0
    pred_loss = 0
    for t in range(n_test):
        x_test, y_test = random_batch(images, true_labels, images.shape[0], sample_size)

        outp = model(tf.constant(x_test))
        logits = outp[0]

        predictions = tf.argmax(logits,1)
        labels = tf.argmax(y_test, 1)
        loss_batch += loss(logits = logits, labels=y_test)
        accu += tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
        pred_loss += outp[1]
    return accu, loss_batch, pred_loss

class gated_conv2d(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, gates_input_units, **kwargs):

        super(gated_conv2d, self).__init__(filters, kernel_size, **kwargs)

        self.gates_input_units = gates_input_units
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=2)] 

    def build(self, input_shape):
        super(gated_conv2d, self).build( input_shape[0])

        channel_axis = -1    
        input_dim = input_shape[0][channel_axis].value #makes it an int rather than type(dimension)  
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        #This excludes input_height and width, to keep dim tractable, gating depends not on 
        # location but lcoation can be retrieved from output activaitons.
        self.gates_kernel = self.add_weight(shape=(self.gates_input_units, np.prod(self.kernel_shape)),
                                        initializer=self.kernel_initializer,
                                        name='gates_kernel',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.gates_bias = self.add_weight(shape=(np.prod(self.kernel_shape),),
                                        initializer=self.bias_initializer,
                                        name='gates_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        

    def call(self, inputs):
        input_data      = inputs[0]
        input_embedding = inputs[-1] 
        self.gates = self.update_gates(input_embedding)

        reshaped_kernel = tf.tile( tf.expand_dims(self.kernel, 0), [self.gates.shape[0],1,1,1,1])
        gated_kernel = tf.keras.layers.multiply([reshaped_kernel, self.gates+1.])
# but that would not work with tf.nn.conv2d, the kernels cannot be batched
# run serially?
#         outputs=[]
#         pred_err = []
#         for (inp, gk) in zip(input_data,gated_kernel):
# # will this work? is the zip gonna pick tha batch dim?
#             outputs.append( tf.nn.conv2d(
#                 tf.expand_dims(inp, axis=0),
#                 gk,
#                 strides=[1, *self.strides, 1],  
#                 padding='VALID') # Keras is ok with 'valid' but this is case_sensetive.
#                 ## lots of differences between K.Conv2D arguments and this. 
#                 # it needs 4D strides [1,stride,stride,1] 
#                 # and 4D input [batch, height, w, channels]
#                 # and 4D kernel[kernel height, kernel W, channels_in, channels_out]
#             )
#         outputs = tf.squeeze(outputs, axis=1)

        #F has shape (MB, fh, fw, channels, out_channels)
        # REM: with the notation in the question, we need: channels_img==channels
        MB, fh, fw, channels_in, channels_out = gated_kernel.shape.as_list()

        F = tf.transpose(gated_kernel, [1, 2, 0, 3, 4])
        F = tf.reshape(F, [fh, fw, channels_in*MB, channels_out])
        _ , H, W, _ = input_data.shape.as_list()
        inp_r = tf.transpose(input_data, [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
        inp_r = tf.reshape(inp_r, [1, H, W, MB*channels_in])

        out = tf.nn.depthwise_conv2d(
                inp_r,
                filter=F,
                strides=[1, 1, 1, 1],
                padding='VALID') # here no requirement about padding being 'VALID', use whatever you want. 
        # Now out shape is (1, H, W, MB*channels*out_channels)

        out = tf.reshape(out, [H-fh+1, W-fw+1, MB, channels_in, channels_out]) # careful about the order of depthwise conv out_channels!
        out = tf.transpose(out, [2, 0, 1, 3, 4])
        outputs = tf.reduce_sum(out, axis=3)

        # CALUCULATE PRED_ERR:
        pred_err = tf.zeros(1)
        # inputs_shape = input_data.shape.as_list()
        # inputs_shape[0]= 1
        input_shape = [1, H, W, channels_in]

        for b in range(MB):
            b_out = tf.nn.conv2d_transpose(
                tf.expand_dims(outputs[b], axis=0), self.gates[b], input_shape, strides=[1,1,1,1], padding='VALID')
            # pred_err = tf.reduce_sum( tf.reshape( b_out, input_data[b]  , -1))
            # curr_pred_err = tf.keras.layers.add([input_data[b],-tf.squeeze(b_out,0)])
            # pred_err = tf.keras.backend.sum(pred_err)
            # curr_pred_err = tf.reduce_sum(curr_pred_err)
            # pred_err += (tf.nn.l2_loss(curr_pred_err) )
            # pred_err += tf.square(curr_pred_err)
            pred_err += tf.reduce_sum(tf.pow(input_data[b]- tf.squeeze(b_out,0), 2) )

        # out shape is now (MB, H, W, out_channels)
        # outputs = out
        if self.use_bias:
            outputs = tf.keras.backend.bias_add(outputs, self.bias, data_format='channels_last')


        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs , pred_err

    def update_gates(self, inputs):
        output = tf.matmul(inputs, self.gates_kernel)
        if self.use_bias: 
            output = tf.keras.backend.bias_add(output, self.gates_bias)
        if self.activation is not None:
            output = self.activation(output)
        self.gates = tf.reshape(output, [-1, *self.kernel_shape])
        return self.gates

    def compute_output_shape(self, input_shape):
        input_embed = input_shape[1]
        input_shape = input_shape[0] #routing input_shape so not to rej-write all below.
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            # return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            # [self.filters])
            return tensor_shape.TensorShape([ [input_shape[0]] + new_space + [self.filters], []  ])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            # return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                            # new_space)

            return tensor_shape.TensorShape([ [input_shape[0], self.filters] + new_space, [] ])

class gated_dense(tf.keras.layers.Layer):
    """Using the source code of Keras.layers.Dense as a starting point to define
    a gated dense layer.

    # Input shape
        2 input tensors. Input from previous layer, and input from gate_generation_layer
        tensor with shape: `(batch_size, ..., input_dim)`.
        tensor with shape: `(batch_size, ..., gate_input_dim)`.
        
    """

    def __init__(self, units,
                 gates_input_units,
                 use_bias=True,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_pred_loss= True,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(gated_dense, self).__init__(**kwargs)
        self.units = units
        self.gates_input_units = gates_input_units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.supports_masking = True
        self.use_pred_loss = use_pred_loss

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = int(input_shape[0][-1])
        batch_dim = int(input_shape[0][0]) #batch is not really known at build time it is just '?' which does not work for var init
        
        self.kernel_size = (input_dim, self.units)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.gates_kernel = self.add_weight(shape=(self.gates_input_units, input_dim * self.units),
                                      initializer=self.kernel_initializer,
                                      name='gates_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.gates_bias = self.add_weight(shape=(input_dim*self.units,),
                                    initializer=self.bias_initializer,
                                    name='gates_bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        # self.gates_input = np.zeros([param.batch_size, self.gates_input_units], dtype=np.float32)  # + (1,1)) ?? why was this used?
        # assuming prior knowledge of batch_size
        # self.gates_input = tf.constant(self.gates_input, name='gates_input')   ### HOPE this is note learning anything
        self.gates = np.ones([input_dim, self.units], dtype=np.float32)
        self.flow  = np.ones([input_dim, self.units], dtype=np.float32)
        
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        # self.dense_gates = tf.keras.layers.Dense(input_dim* self.units) # this failed to register as model.variables
        
        # super(gated_dense, self).build(input_shape)  # is this needed?
        self.built = True

    def call(self, inputs):
        inputs1= inputs[0]
        embed = inputs[1]
        # Update the gates
            # if self.gates_input.shape[0] < inputs.shape[0]:
            #     self.gates_input = tf.tile(self.gates_input, [inputs.shape[0], 1])
            # self.gates = self.dense_gates(self.gates_input)
            # self.gates = self.dense_gates(embed)
        self.gates = self.update_gates(embed)
        
        assert(self.gates.shape[0] == inputs1.shape[0])
        
        # Flow gates: Reduced accuracy by almost 1%
        pred_err = 0.
        if self.use_pred_loss:
            reshaped_inputs = tf.tile( tf.expand_dims(inputs1, 2), [1,1, self.kernel.shape[1]])         # To calculate flow, tile inputs to match the output  dim of the kernel
            self.flow   = reshaped_inputs * self.kernel
            norm_flow = tf.keras.backend.l2_normalize(self.flow, axis=[1,2])
            norm_gates = tf.keras.backend.l2_normalize(self.gates, axis=[1,2])
            
            # pred_err = tf.keras.backend.sum(pred_err)
            # # pred_err = tf.nn.l2_loss(pred_err)
            pred_add = tf.keras.layers.add([norm_flow,-norm_gates]) 
            pred_err += tf.reduce_sum(tf.pow(pred_add, 2) )

                
        # self.gates = self.gates * self.flow  #Should this be calculated before or after flow for this time step is calculated.
        # tempting to use flow from most recent iteration to gate feedback, but then use current flow for prediction.     
        reshaped_kernel = tf.tile( tf.expand_dims(self.kernel, 0), [self.gates.shape[0],1,1])
        gated_kernel = tf.keras.layers.multiply([reshaped_kernel, self.gates+1.])
        # gated_kernel = self.kernel * self.gates  ### HOPE this is ok with self.gates having batch size dim
        # if gated_kernel.shape[0] < inputs.shape[0]:
        #     gated_kernel = tf.tile(gated_kernel, [inputs.shape[0], 1,1])

        inputs1 = tf.expand_dims(inputs1, axis = 1)  # inputs = [batch, 1, input], [batch, input, output]
        output = tf.matmul(inputs1, gated_kernel)
        # self.flow   = K.tile(inputs, self.kernel.shape[1]) * self.kernel
        output= tf.squeeze(output, axis=1)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output, pred_err

    def update_gates(self, inputs):
        output = tf.matmul(inputs, self.gates_kernel)
        if self.use_bias: 
            output = tf.keras.backend.bias_add(output, self.gates_bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        self.gates = tf.reshape(output, [-1, *self.kernel_size])
        return self.gates

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape1 = list(input_shape)
        output_shape1[-1] = self.units
        output_shape2 = list(input_shape)   # ouput is [batch, self.units] pred_er [batch, input, units]
        output_shape2.append(self.units)
        return [tuple(output_shape1), tuple(output_shape2)]

class my_model(tf.keras.Model):
    def __init__(self, use_gates = True, use_pred_loss = True):
        super(my_model, self).__init__()
        self.use_gates= use_gates
        self.use_pred_loss = use_pred_loss        
        # self.gated_conv1 = gated_conv2d(32, (3, 3), param.embedding_size, activation='relu', input_shape=(img_size,img_size,num_channels))
        self.a = []
        weight_decay = 1e-4

        # self.conv1 = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(img_size,img_size,num_channels))
        # self.conv2 = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu')
        self.conv1 = tf.keras.layers.Convolution2D(32, (3, 3), activation='elu', input_shape=(img_size,img_size,num_channels), padding='valid', kernel_regularizer=regularizers.l2(weight_decay))
        self.conv2 = tf.keras.layers.Convolution2D(32, (3, 3), activation='elu', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))
        self.conv4 = tf.keras.layers.Convolution2D(64, (3, 3), activation='elu', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))
        self.conv3 = tf.keras.layers.Convolution2D(64, (3, 3), strides=(1,1), activation='elu', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))
        self.conv5 = tf.keras.layers.Convolution2D(128, (3, 3), activation='elu', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))
        self.conv6 = tf.keras.layers.Convolution2D(128, (3, 3), strides=(1,1), activation='elu', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))
        # self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        # self.dropout1 = tf.keras.layers.Dropout(0.25)

        self.flatten1 =tf.keras.layers.Flatten()
        self.gated1 = gated_dense(128, param.embedding_size, activation='elu', use_pred_loss=self.use_pred_loss)
        self.dense1 =  tf.keras.layers.Dense(128, activation = 'relu')
                    
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense( 10, activation='softmax')
        self.gated2 = gated_dense(param.output_layer_size, param.embedding_size, activation='softmax', use_pred_loss=self.use_pred_loss)
        self.embed1 = tf.keras.layers.Dense(param.embedding_layer_size, activation='relu', name='gates_embed')
 
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.batch_norm5 = tf.keras.layers.BatchNormalization()
        self.batch_norm6 = tf.keras.layers.BatchNormalization()

        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dropout3 = tf.keras.layers.Dropout(0.4)

    def _model_run(self, inputs): 
        """Run the model."""
        inputs1 = inputs[0]
        inputs2 = inputs[1]
        predictive_loss = 0.  # init a list to hold the errors between flow and gates.
        pred_err = 0.
        embedding = inputs2  # self.LSTM(activations)
        
        # result, pred_err = self.gated_conv1([inputs1, embedding])
        # predictive_loss += tf.reduce_sum(pred_err)
        
        result = self.conv1(inputs1)
        result = self.batch_norm1(result)
        
        result = self.conv2(result)
        result = self.batch_norm2(result)
        result = self.maxpool1(result)
        result = self.dropout1(result)

        result = self.conv3(inputs1)
        result = self.batch_norm3(result)
        result = self.conv4(result)
        result = self.batch_norm4(result)
        result = self.maxpool2(result)
        result = self.dropout2(result)

        result = self.conv5(inputs1)
        result = self.batch_norm5(result)
        result = self.conv6(result)
        result = self.batch_norm6(result)
        result = self.maxpool3(result)
        result = self.dropout3(result)
        self.a = [result]
        # result = self.maxpool1(result)
        # result = self.dropout1(result)
        """" THIS IS A MAJOR COMPROMISE, FLATTENING ALL CHANNEL DATA 
        I GET OOM WITH OUT IT THOUGH!                               """

        result = tf.reduce_mean(result, axis=[-1])
        result = self.flatten1(result)
        # result = self.batch_norm2(result)

        result, pred_err = self.gated1([result, embedding])
        # result = self.dense1(result)
        # self.a.append(result)  
        # predictive_loss += tf.reduce_sum(pred_err)
        #         # result = self.dropout2(result)
        predictive_loss += tf.reduce_sum(pred_err)
        embed_layer_output = self.embed1(result)
        
        
        # result, pred_err = self.gated2([result, embedding])
        result = self.dense2(embed_layer_output)  
        self.a.append(result)
        embedding_out = tf.concat([result, embed_layer_output], axis=1)
        # embed_layer_output = [] #self.embed1(result)


        #         Tracer()() #this one triggers the debugger
        return [result, embedding_out, predictive_loss]
    def _model_run_no_drop_out(self, inputs): 
        inputs1 = inputs[0]
        inputs2 = inputs[1]
        predictive_loss = 0.  # init a list to hold the errors between flow and gates.
        embedding = inputs2  # self.LSTM(activations)
        result = self.conv1(inputs1)
        result = self.batch_norm1(result)
        result = self.conv2(result)
        result = self.batch_norm2(result)
        result = self.maxpool1(result)
        result = self.conv3(inputs1)
        result = self.batch_norm3(result)
        result = self.conv4(result)
        result = self.batch_norm4(result)
        result = self.maxpool2(result)
        result = self.conv5(inputs1)
        result = self.batch_norm5(result)
        result = self.conv6(result)
        result = self.batch_norm6(result)
        result = self.maxpool3(result)
        result = tf.reduce_mean(result, axis=[-1])
        result = self.flatten1(result)
        result, pred_err = self.gated1([result, embedding])
        predictive_loss += tf.reduce_sum(pred_err)
        embed_layer_output = self.embed1(result)
        result = self.dense2(embed_layer_output)  
        embedding_out = tf.concat([result, embed_layer_output], axis=1)
        return [result, embedding_out, predictive_loss]

    def call(self, inputs):
        input_batch_size = inputs.shape[0]
        init_embed = tf.zeros(shape=[input_batch_size, param.embedding_size])
        embed = init_embed
        [result, embed, pred_loss] = self._model_run([inputs, embed])
        if self.use_gates:
            result, _, pred_loss = self._model_run([inputs, embed])
        
        return result, pred_loss
    
    def confidence(self, inputs):
        input_batch_size = inputs.shape[0]
        init_embed = tf.zeros(shape=[input_batch_size, param.embedding_size])
        embed = init_embed
        final_results = []
        for f in range(3):
            [result, embed, _] = self._model_run([inputs, embed])
            final_results.append(result)
        return final_results
    
    def _generate_run(self, inputs):
        """
            Given an image, this function will run through the model for "iterations"
            and generate an image from the input layer gates.

        """
        input_batch_size = inputs.shape[0]
        init_embed = tf.zeros(shape=[input_batch_size, param.embedding_size])
        embed = init_embed
        [result, embed, _] = self._model_run([inputs, embed])
        
        # result, embed, pred_loss = self._model_run([inputs, embed])
        
        conv_out, _ = self.gated_conv1([inputs, embed])

        conv_gates = self.gated_conv1.gates
        conv_weights = self.gated_conv1.kernel
        input_shape = inputs.shape.as_list()
        input_shape[0] = 1
        images = []
        for b in range(input_batch_size-1):
            b_out = tf.nn.conv2d_transpose(
                tf.expand_dims(conv_out[b], axis=0), conv_gates[b+1], input_shape, strides=[1,1,1,1], padding='VALID')
            # b_out = tf.nn.conv2d_transpose(
            #     tf.expand_dims(conv_out[b], axis=0), conv_weights, input_shape, strides=[1,1,1,1], padding='VALID')
            images.append(b_out)
        
        return images

lr_counter = 0.
learning_rate_schedule = 1.
def lr_schedule():
    lrate = 0.001
    if lr_counter > 20000:
        lrate = 0.0005
    elif lr_counter > 40000:
        lrate = 0.0003        
    return lrate


optimizer = tf.train.AdagradOptimizer(0.1)
# optimizer = tf.train.AdamOptimizer()

x_train = X_train
train_labels = Y_train
test_images = X_test
test_labels = Y_test
steps_per_epoch = int(x_train.shape[0]/param.batch_size)   # 1800 number of epochs  takes 781 batches to finish an epoch


if tfe.num_gpus() == 0: # if CPU then just run a limited debug mode
    param.train_steps = 1 
    param.report_every = 1 
    model.load_weights('weights/' + weights_folder + '/model_weights')

def train(disc_loss, pred_loss, vars, steps):
    start_time = time.time()
    accuracy_values = []
    for i in range(steps):
        with tf.GradientTape() as tape:
            x_batch, y_batch = random_batch(x_train, train_labels, x_train.shape[0], batch_size=param.batch_size)
            logits, pred_loss_value = model(tf.constant(x_batch))
            disc_loss_value = loss(logits = logits, labels=y_batch)
        
        losses= []
        if disc_loss:
            losses.append(disc_loss_value)
        if pred_loss:
            losses.append(pred_loss_value)

        grads = tape.gradient(losses, vars)
        optimizer.apply_gradients( zip(grads, vars) , global_step=tf.train.get_or_create_global_step()   )

        if i%param.report_every == 1 :
    #training set accuracy
            test_start_time = time.time()
            train_accu,train_loss, _ = eval_model(x_train, train_labels, 100)
            test_accu,_, test_pred_loss = eval_model(test_images, test_labels, 100)
            accuracy_values.append(test_accu)
        
            print('Step:{} train acc:{:2.3f} loss:{:2.3f}  test acc:{:2.3f} pred_loss:{:1.2e}, in seconds: {:2.0f},t {:2.0f}'.format(i, train_accu, train_loss, test_accu, test_pred_loss, test_start_time - start_time ,time.time()-test_start_time))
            start_time = time.time()
            # disp_grads(model.a, forward_variables)   

model= my_model(use_gates=True, use_pred_loss = True)
# model= my_model(use_gates=False, use_pred_loss = False)

x_batch, y_batch = random_batch(x_train, train_labels, x_train.shape[0], batch_size=param.batch_size)
x =tf.constant(x_batch)
_,_ = model(x) ## just to build the model. (to be able to reference model.variables below.)

gen_vars = [v for v in model.variables if 'gates' in v.name] 
forward_variables = [v for v in model.variables if 'gates' not in v.name] 
embed_vars = [v for v in model.variables if 'embed' in v.name]
forward_variables = forward_variables+ embed_vars
#%%
train(disc_loss=True, pred_loss=False, vars=forward_variables, steps=4*steps_per_epoch)
#%%
train(disc_loss=True, pred_loss=False, vars=forward_variables, steps=2*steps_per_epoch)
train(disc_loss=False, pred_loss=True, vars=gen_vars, steps=2*steps_per_epoch)
train(disc_loss=True, pred_loss=False, vars=forward_variables, steps=2*steps_per_epoch)
train(disc_loss=False, pred_loss=True, vars=gen_vars, steps=2*steps_per_epoch)
train(disc_loss=True, pred_loss=False, vars=forward_variables, steps=2*steps_per_epoch)
train(disc_loss=False, pred_loss=True, vars=gen_vars, steps=2*steps_per_epoch)
train(disc_loss=True, pred_loss=False, vars=forward_variables, steps=2*steps_per_epoch)
train(disc_loss=False, pred_loss=True, vars=gen_vars, steps=2*steps_per_epoch)
train(disc_loss=True, pred_loss=False, vars=forward_variables, steps=2*steps_per_epoch)

from SGN_utils import *


#%%

gates_god(model)
