

'''
***************************************************
Part I import the model
'''
#3rd library
import tensorflow as tf 
import numpy as np
import cv2 # Open Source Computer Vision Library from Opencv
import scipy.io

#standard library
import argparse
import struct #be used in handling binary data
import errno
import time
import os

'''
********************************************************************************************************
********************************************************************************************************
Part II parsing the command line arguments,configration
********************************************************************************************************
********************************************************************************************************
'''
def parse_args():
        parser=argparse.ArgumentParser(description="tensorflow Neural-Algorithm for Artistic Style ") 
        # to create a Instance of class ArgumentParser
        
        # option for single image
        parser.add_argument('--verbose',action='store_true', 
                            # if the command-line shows "--verbose", 
                            #then argument "varbose" will be assigned to be 1(which means true) ！！！！！
                           help='Boolean flag indicating if statements should be printed to the console,')
                            # Call the methond of class ArgumentParser
            
        parser.add_argument('--img_name',type=str,default='result',#！the name can be changed!
                           help='Filename of the output image.')
        parser.add_argument('--style_imgs',nargs='+',type=str,default='woman.jpg',
                           help='Filename of the style images (example:starry-nigth.jpg)')
        parser.add_argument('--style_imgs_weights',nargs='+',type=float,default=[1.0],
                           help='Interpolation weight of each of the style images.(example:0.5 0.5)')
        parser.add_argument('--content_img',type=str, default='lion.jpg',
                           help='Filename of the content image (example:line.jpg)')
        parser.add_argument('--style_imgs_dir',type=str, default="C:\\Users\\Mason\\desktop\\styles",
                           help='Directory path to the style images.(default: %(default)s)')
                           #the directory can be changed!
            
        parser.add_argument('--content_img_dir',type=str,default='C:\\Users\\Mason\\desktop\\contents',
                           help='Directory path to the content image.(default:%(default)s)')
        parser.add_argument('--init_img_type',type=str,default='content',choices=['random','content','style'],
                           help='Image used to initialize the network.(default:%(default)s)')
        parser.add_argument('--max_size',type=int,default=512,
                           help='Maximum width or height of the input images.(default:%(default)s)')
        parser.add_argument('--content_weight',type=float,default=5e0,
                           help='Weight for the content loss function.(defaule:(%default)s)')
        parser.add_argument('--style_weight',type=float,default=1e4,
                           help='Weight for the style loss function.(default:%(default)s)')
        parser.add_argument('--tv_weight',type=float,default=1e-3,
            help='Weight for the total variational loss function. Set small(e.g. 1e-3).(default:%(default)s)')
        parser.add_argument('--temporal_weight',type=float,default=2e2,
                           help='Weight for the temporal loss function.(default:%(default)s)')
        parser.add_argument('--content_loss_function',type=int,default=1,choices=[1,2,3],
                           help='Different constants for the content layer loss function.(default:%(default)s)')
        parser.add_argument('--content_layers',nargs='+',type=str,default=['conv4_2'],
                            #the VGG19 layers  from the file?
                           help='VGG19 layers used for the content image.(default:%(default)s)') 
        parser.add_argument('--style_layers',nargs='+',type=str,
                            default=['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'],
                           help='VGG19 layers used for the style image.(default:%(default)s)')
        parser.add_argument('--style_layer_weights',nargs='+',type=float,default=[0.2,0.2,0.2,0.2,0.2],
                           help='Contributions (weight) of each content layer to loss.(defaule:%(default)s)')
        parser.add_argument('--content_layer_weights',nargs='+',type=float,default=[1.0],
                           help='Contributions (weight) of each content layer to loss.(defaule:%(default)s)')
        #parser.add_argument('--style_layer_weights',nargs='+',type=float, default=[0.2,0.2,0.2,0.2,0.2],
        #                   help='Contribution (weights) of each style layer to loss.(default:%(default)s)')
        parser.add_argument('--origional_colors',action='store_true',
                           help='Transfer the style but not the colors.')
        parser.add_argument('--color_convert_time',type=str,default='after',choices=['after','before'],
                           help='Time (before or after) to convert to origional colors (default:%(default)s)')
        parser.add_argument('--color_convert_type',type=str,default='yuv',choices=['yuv','ycrcb','luv','lab'],
                           help='Color space for conversion to origional colors (default:%(default)s)')
        parser.add_argument('--style_mask',action='store_true',
                            # if true then use mask, the directory path of mask is same as centent image.
                           help='Transfer the style to masked region.')
        parser.add_argument('--style_mask_imgs',nargs='+',type=str,default=None,
                           help='Filename of the style mask image(example:face_mask.png) (default:%(default)s)')
        parser.add_argument('--noise_ratio',type=float,default=1,
                    help="Tnterpolation value between content and noise image if network initialized with 'random'.")
        parser.add_argument('--seed',type=int, default=0,
                           help='Seed for the random number generator.(default:%(default)s)')
        parser.add_argument('--model_weights',type=str,default='imagenet-vgg-verydeep-19.mat',
                            #imagenet-vgg-verydeep-19.mat need download from web.
                           help='Weights and biases of the VGG-19 network')
        parser.add_argument('--pooling_type',type=str,default='avg',
                           help='type of pooling in convolutional neural network.(default:%(default)s)')
        parser.add_argument('--device',type=str,default='/cpu:0',choices=['/gpu:0','/cpu:0'],
                            #the :0  means the first cpu/gpu, since you may have more than one cpu/gpu.
                           help='GPU or CPU mode.GPU mode requires NVIDIA CUDA.(default|recommended:GPU)')
        parser.add_argument('--img_output_dir',type=str,default='C:\\Users\\Mason\\Desktop\\output',
                           help='Relative or absolute directory path to output image and data.')
        
        
        #optimizations
        parser.add_argument('--optimizer',type=str,default='lbfgs',choices=['lbfgs','adam'],
                    help='Loss minimization optimizer. lbfgs gives better results. adam uses less memory.')
        parser.add_argument('--learning_rate',type=float,default=1e0,
                           help='Learning rate parameter for the adam optimizer.(default:%(default)s)')
        parser.add_argument('--print_iterations',type=int,default=10,
                           help='Number of iterations between optimizer print statements.(default:%(default)s)')
        parser.add_argument('--max_iterations',type=int,default=100,
                           help='Max number of iterations for the adam or lbfgs optimizer.(default:%(default)s)')
        
        args=parser.parse_args()#args is a  namespace object

        args.style_layer_weights = normalize(args.style_layer_weights)
        args.content_layer_weights = normalize(args.content_layer_weights)
        args.style_imgs_weights = normalize(args.style_imgs_weights)
        
        #create directories for output
        maybe_make_directory(args.img_output_dir)

        return args


            
            
            
'''
********************************************************************************************************
********************************************************************************************************
Part III pre-trianed vgg 19 CNN, remark: layers are manually initialized for clarity
********************************************************************************************************
********************************************************************************************************

'''
# building the self net, extract layers from VGG19 model*******************************
def build_model(input_img):  
    
    if args.verbose: print('\nBUILDING VGG-19 NETWORK')# if "verbose" is true then print all if and do statements
    net={}
    _,h,w,d =input_img.shape
        
    if args.verbose: print('loading model weights...')
    vgg_rawnet=scipy.io.loadmat(args.model_weights)
    # scipy.io.loadmat(), used to load a matlab file, 
    #and args.model_weights is point to file"imagenet-vgg-verydeep-19.mat" 
    #and the layer_data is 5D，  4D for image data, and 1D for layer number 0-42. know from the .mat file 
    vgg_layers=vgg_rawnet['layers'][0]# this additional [0] is weights (0 is Weight W; 1 is bias)
    #, that makes vgg_layers to be 6D,[][][][][][]
    
    if args.verbose: print('constructing layers...')
    net['input']=tf.Variable(np.zeros((1,h,w,d),dtype=np.float32))
    #tf.variable():to create a tensor variable ,
    #and numpy.zeros():Return a new array of given shape and type, filled with zeros. 
    
    if args.verbose: print('LAYER GROUP 1')
    # the net of built model, is actually a slight change from VGG19 
    # the conv_layers and relu_layers is compluted by tensorflow, 
    #and the arguments conv{}_{} are initialized by tf.variable(np.zeros())
    # so, as a whole, the net are get by tensorflow, the weight are from VGG19 matlab file which download from web.
    

    net['conv1_1'] =conv_layer('conv1_1',net['input'],W=get_weights(vgg_layers,0)) 
    # VGG19 has 43 layers, and begin from 0(0~42)
    net['relu1_1'] =relu_layer('relu1_1',net['conv1_1'],b=get_biases(vgg_layers,0))
    net['conv1_2'] =conv_layer('conv1_2',net['relu1_1'],W=get_weights(vgg_layers,2))
    net['relu1_2'] =relu_layer('relu1_2',net['conv1_2'],b=get_biases(vgg_layers,2))
    net['pool1'] = pool_layer('pool1',net['relu1_2'])
        
    if args.verbose: print('LAYER GROUP 2')
    net['conv2_1'] =conv_layer('conv2_1',net['pool1'],W=get_weights(vgg_layers,5))
    net['relu2_1'] =relu_layer('relu2_1',net['conv2_1'],b=get_biases(vgg_layers,5))
    net['conv2_2'] =conv_layer('conv2_2',net['relu2_1'],W=get_weights(vgg_layers,7))
    net['relu2_2'] =relu_layer('relu2_2',net['conv2_2'],b=get_biases(vgg_layers,7))
    net['pool2'] = pool_layer('pool2',net['relu2_2'])
        
    if args.verbose: print('LAYER GROUP 3')
    net['conv3_1'] =conv_layer('conv3_1',net['pool2'],W=get_weights(vgg_layers,10))
    net['relu3_1'] =relu_layer('relu3_1',net['conv3_1'],b=get_biases(vgg_layers,10))
    net['conv3_2'] =conv_layer('conv3_2',net['relu3_1'],W=get_weights(vgg_layers,12))
    net['relu3_2'] =relu_layer('relu3_2',net['conv3_2'],b=get_biases(vgg_layers,12))
    net['conv3_3'] =conv_layer('conv3_3',net['relu3_2'],W=get_weights(vgg_layers,14))
    net['relu3_3'] =relu_layer('relu3_3',net['conv3_3'],b=get_biases(vgg_layers,14))
    net['conv3_4'] =conv_layer('conv3_4',net['relu3_3'],W=get_weights(vgg_layers,16))
    net['relu3_4'] =relu_layer('relu3_4',net['conv3_4'],b=get_biases(vgg_layers,16))
    net['pool3'] = pool_layer('pool3',net['relu3_4'])
    if args.verbose: print('LAYER GROUP 4')
    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_biases(vgg_layers, 19))
    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_biases(vgg_layers, 21))
    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_biases(vgg_layers, 23))
    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_biases(vgg_layers, 25))
    net['pool4']   = pool_layer('pool4', net['relu4_4'])

    if args.verbose: print('LAYER GROUP 5') # the layer of built model total contains 5 layers.
    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_biases(vgg_layers, 28))
    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_biases(vgg_layers, 30))
    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_biases(vgg_layers, 32))
    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_biases(vgg_layers, 34))
    net['pool5']   = pool_layer('pool5', net['relu5_4'])
        
    return net

#*builing model end ****************************************



# tendorflow computes*************************************************
def conv_layer(layer_name,layer_input,W):
    conv = tf.nn.conv2d(layer_input,W,strides=[1,1,1,1],padding='SAME') # W weights  is filter
    #Computes a 2-D convolution (neural network)given 4-D input and filter tensors,return a 4D tensor type as input
    # strides:A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. 
    # padding:A string from: "SAME", "VALID". The type of padding algorithm to use.
    if args.verbose: 
        print('--{} | shape={} | weights_shape={}'.format(layer_name,conv.get_shape(),W.get_shape())) 
        # the symbol "|" can mean "or"   
        # and the print('{}'.format(?,?,?))， this is print with format and {} can be replace by %
        # eg:   print("my name is {}| age {}".format("Mason", 20))  my name is Mason|age 20
    return conv
    
def relu_layer(layer_name,layer_input,b):
    relu =tf.nn.relu(layer_input + b)
    #relu :Rectified Linear Units, a active function. Computes rectified linear: max(features, 0)
    # return a tensor
    if args.verbose: print('--{}|shape{}|bias_shape={}'.format(layer_name,relu.get_shape(), b.get_shape()))
    return relu

def pool_layer(layer_name,layer_input):
    if args.pooling_type =='avg':
        pool =tf.nn.avg_pool(layer_input,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        # tf.nn.vag_pool:Performs the average pooling on the input.
        # ksize: A 1-D int Tensor of 4 elements. The size of the window for each dimension of the input tensor.
    elif args.pooling_type=='max':
        pool=tf.nn.max_pool(layer_input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    if args.verbose:
        print('--{}|shape={}'.format(layer_name,pool.get_shape()))
    return pool
#tf end****************************************************


def get_weights(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0] 
    #vgg_layers consist of 6-D? using 6 dimensions list.
    # first [i] means ith layer,  but what is the meaning of data [0][0][2][0]? maybe its doesnt matter.
    W=tf.constant(weights)#maping the 'layer' of .mat file
    return W #return a filter data, 4D-list (* 3x3 matrixes, and *=d multiply _). 
            #the vgg19 met file: layer1_i.weight{1,1} 

def get_biases(vgg_layers,i):
    bias=vgg_layers[i][0][0][2][0][1]
    b=tf.constant(np.reshape(bias,(bias.size)))    
    return b#return a 1D array (1x_), layer-data is (_,h,w,d)



#*Loss function****************************************************

def content_layer_loss(p,x):
    _,h,w,d=p.get_shape()
    M=h.value*w.value
    N=d.value
    if args.content_loss_function ==1:
        K=1./(2. * N**0.5 * M**0.5) #the ** means exponential operation
    elif args.content_loss_function ==2:
        K=1./(N*M)
    elif args.content_loss_function ==3:
        K=1./2.
    loss=K*tf.reduce_sum(tf.pow((x - p),2))
    return loss

def style_layer_loss(a,x):
    _,h,w,d=a.get_shape()
    M=h.value*w.value
    N=d.value
    A=gram_matrix(a,M,N)
    G=gram_matrix(x,M,N)
    loss=(1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
    return loss

def gram_matrix(x,area,depth):
    F=tf.reshape(x,(area,depth))
    G=tf.matmul(tf.transpose(F),F)
    return G

def mask_style_layer(a,x,mask_img):
    _,h,w,d=a.get_shape()
    mask=get_mask_image(mask_img,w.value,h.value)
    mask=tf.convert_to_tensor(mask)
    tensors=[]
    for _ in range(d,value):
        tensors.append(mask)
        mask=tf.stack(tensors,axis=2)
        mask=tf.stack(mask,axis=0)
        mask=tf.expand_dims(mask,0)
        a=tf.multiply(a,mask)
        x=tf.multiply(x.mask)
    return a,x

def sum_masked_style_losses(sess,net,style_imgs):
    total_style_loss = 0.
    weights = args.style_imgs_weights
    masks = args.style_mask_imgs
    for img, img_weight, img_mask in zip(style_imgs, weights, masks):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(args.style_layers, args.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            a, x = mask_style_layer(a, x, img_mask)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(args.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss

def sum_style_losses(sess,net,style_imgs):
    total_style_loss=0.
    weights=args.style_imgs_weights
    for img,img_weight in zip(style_imgs,weights):
        sess.run(net['input'].assign(img))
        style_loss=0.
        for layer,weight in zip(args.style_layers,args.style_layer_weights):
            a=sess.run(net[layer])
            x=net[layer]
            a=tf.convert_to_tensor(a)            
            style_loss += style_layer_loss(a,x) * weight
        style_loss /= float(len(args.style_layers))
        total_style_loss += (style_loss*img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss

def sum_content_losses(sess,net,content_img):
    sess.run(net['input'].assign(content_img))
    content_loss=0.
    for layer, weight in zip(args.content_layers, args.content_layer_weights):
        p=sess.run(net[layer])
        x=net[layer]
        p=tf.convert_to_tensor(p)
        content_loss += content_layer_loss(p,x)* weight
    content_loss /= float(len(args.content_layers))
    return content_loss

'''
#artistic style transfer for videos' loss functions
def temporal_loss(x,w,c):
    c = c[np.newaxis,:,:,:]
    D = float(x.size)
    loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
    loss = tf.cast(loss, tf.float32)
    return loss

def get_longterm_weight(i,j):
    c_sum = 0.
    for k in range(args.prev_frame_indices):
        if i - k > i - j:
            c_sum += get_content_weights(i, i - k)
    c = get_content_weights(i, i - j)
    c_max = tf.maximum(c - c_sum, 0.)
    return c_max

def sum_longterm_temporal_losses(sess,net,frame,input_img):
    x = sess.run(net['input'].assign(input_img))
    loss = 0.
    for j in range(args.prev_frame_indices):
        prev_frame = frame - j
        w = get_prev_warped_frame(frame)
        c = get_longterm_weights(frame, prev_frame)
        loss += temporal_loss(x, w, c)
    return loss

def sum_shortterm_termporal_loss(sess,net,frame,input_img):
    x = sess.run(net['input'].assign(input_img))
    prev_frame = frame - 1
    w = get_prev_warped_frame(frame)
    c = get_content_weights(frame, prev_frame)
    loss = temporal_loss(x, w, c)
    return loss
'''
#****loss function end*************************************


'''
utilities and I/O
'''
def read_image(path):
    #bgr image
    img=cv2.imread(path,cv2.IMREAD_COLOR) 
    # cv2 model, imread:input, imshow:show, imwrite:save;
    # IMREAD_COLOR: input as colorful model
    check_image(img,path) # a def function
    img=img.astype(np.float32)
    img=preprocess(img)    #preprocess is a def function 
    return img

def write_image(path,img):
    img=postprocess(img)
    cv2.imwrite(path,img)
    
def preprocess(img):
    img=img[...,::-1]# python list slice[start:end:step] to create a new sequence, -1 mean revers direction. 
    img=img[np.newaxis,:,:,:] 
    # shape(h,w,d) to (1,h,w,d)
    # add a new axis in list, np.newaxis equal to None,ima[None,:,:,:]
    img -= np.array([123.68,116.779,103.939]).reshape((1,1,1,3))
    #numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0), to create a array.
    # np.arrar([1.0,2.0,3.0]).reshape((1,1,1,3)) results transfer 
    #array to 4-Dimensional 1x1x1x3 matrix [[[[1.0,2.0,3.0]]]]
    return img

def postprocess(img):
    img += np.array([123.68,116.779,103.939]).reshape((1,1,1,3))
    # shape(h,w,d) to (1,h,w,d)
    img=img[0]
    img=np.clip(img,0,255).astype(np.float32)# to uint8
    img=img.astye(np.uint8)
    # np.clip(array_name,Low limit,Up limit),
    #if the element is less than LL then will be force to equal to LL value, and large -> UL
    #astype: transfer the data type
    
    img=img[...,::-1]
    return img

'''
def read_flow_file(path):
    with open(path,'rb') as f: # open with binary read model
        header=struct.unpack('4s',f.read(4))[0] 
        #struct.unpack(format,string):4s is for strings each, .read(4) means open file and read 4 strings
        # [][], means that header is a element of a 2d array
        w=struct.unpack('i',f.read(4))[0] # i means unsigned int, size 4 bit
        h=struct.unpack('i',f.read(4))[0]
        flow=np.ndarray((2,h,w),dtype=np.float32)
        for y in range(h):
            for x in range(w):
                flow[0,y,x]=struct.unpack('f',f.read(4))[0]
                flwo[1,y,x]=struct.unpack('f',f.read(4))[0]
    return flow
'''

def read_weights_file(path):
    lines=open(path).readlines()
    header=list(map(int,lines[0].split(''))) 
    # list() transfer tuple to list. and map(fuction,arguments): each argument call function,return a list
    
    w=header[0]
    h=header[1]
    vals=np.zeros((h,w),dtype=np.float32)
    for i in range(1,len(lines)):
        line=lines[i].rstrip().split('')
        vals[i-1]=np.array(list(map(np.float32,line)))
        vals[i-1]=list(map(lambda x:0. if x< 255. else 1.,vals[i-1]))
        # lambda is used to create a anonymous function consistng of single express. eg, log2=lambda x:log(x)/log(2)
    weights=np.dstack([vals.astype(np.float32)]*3)
    # np.dstack:Takes a sequence of arrays and stack them along the third axis to make a single array
    return weights

def normalize(weights):
    denom=sum(weights)
    if denom>0:
        return [float(i)/denom for i in weights]
    else:
        return [0.]*len(weights)
    

def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
       os.makedirs(dir_path)
    
def check_image(img,path):
    if img is None:
       raise OSError(errno.ENOENT,"No such file", path)
    




'''
*****************************************************************
Part IV rendering--where the magic happen !!!!
'''
def stylize(content_img,style_imgs,init_img,frame=None):
    with tf.device(args.device),tf.Session() as sess:
            
    #setup network
            net =build_model(content_img)
    
    #style loss
            if args.style_mask:
                L_style=sum_masked_style_losses(sess, net, style_imgs)
            else:
                L_style=sum_style_losses(sess,net,style_imgs)
        
    #content loss 
            L_content=sum_content_losses(sess,net,content_img)
    
    #denoising loss
            L_tv=tf.image.total_variation(net['input'])
    
    #loss weights
            alpha=args.content_weight
            beta=args.style_weight
            theta=args.tv_weight
    
    #totle loss
            L_total =alpha * L_content
            L_total +=beta*L_style
            L_total +=theta*L_tv
    

    #optimization algorithm
            optimizer=get_optimizer(L_total)
            if args.optimizer == 'adam':
                   minimize_with_adam(sess,net,optimizer,init_img,L_total)
            elif args.optimizer =='lbfgs':
                   minimize_with_lbfgs(sess,net,optimizer,init_img)
            output_img=sess.run(net['input'])
    
            if args.original_colors:
                output_img=convert_to_original_colors(np.copy(content_img),output_img)
            #if args.video:
            #    write_video_output(frame,output_img)
            else:
                write_image_output(output_img,content_img,style_imgs,init_img)
        

def minimize_with_lbfgs(sess,net,optimizer,init_img):
    if args.verbose:print('\nMINIMIZING LOSS USING:L-BFGS OPTIMIZER')
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)
        
    
def minimize_with_adam(sess,net,optimizer, init_img,loss):
    if args.verbose:print('/nMINIMIZING LOSS USING:ADAM OPTIMIZER')
    train_op=optimizer.minimize(loss)
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations =0
    while (iterations<args.max_iterations):
        sess.run(train_op)
        if iterations % args.print_iterations==0 and args.verbose:
            curr_loss=loss.eval()
            #equal to' curr_loss=eval(loss)'  the data loss has the attribution .eval()
            
            print("At iterate {}\tf=  {:.5E}".format(iterations, curr_loss))
            # the c# print style, first {} ralated to iterations, and second {} to curr_loss
            #\t is escape character.  f= curr_loss   with exponential notaton with 5 significant figures.
            #:. means right justify.    for example f=1.00210D+09    D means Double float, 
        iterations +=1
        
    
def get_optimizer(loss):
    print_iterations=args.print_iterations if args.verbose else 0
    if args.optimizer == 'lbfgs':
        optimizer=tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                                                        options={'maxter':args.max_iterations,
                                                                'disp':print_iterations})
    elif args.optimizer=='adam':
        optimizer=tf.train.AdamOptimizer(args.learning_rate)
    
    return optimizer

'''
def write_videi_output(frame,out_img):
    fn=args.content_frame_frmt.format(str(frame).zfill(4))
    #str.zfill(width), zero fill and right alignment,returns width's number strings
    path=os.path.join(args.video_output_dir, fn)
    write_image(path,output_image)
'''    
    
def write_image_output(output_img,content_img,style_imgs,init_img):
    out_dir=os.path.join(args.img_output_dir,args.img_name)
    maybe_make_directory(out_dir)
    img_path=os.path.join(oout_dir,args.img_name+'.png')
    content_path=os.path.join(out_dir,'content.png')
    init_path=os.path.join(out_dir,'init.png')
    
    write_image(img_path,output_img)
    write_image(content_path,content_img)
    write_image(init_path,init_img)
    index=0
    for style_img in style_imgs:
        path=os.path.join(out_dir,'style_'+str(index)+'.png')
        write_image(path,style_img)
        index +=1
    
    #save the configuration settings
    out_file=os.path.join(out_dir,'mate_data.txt')
    f=open(out_file,'w')
    f.write('image_name:{}\n'.format(args.img_name))
    f.write('content:{}\n'.format(args.content_img))
    index=0
    for style_img,weight in zip(args.style_imgs,args.style_imgs_weights):
        f.write('style['+str(index)+']:{}\n'.format(weight,style_img))
        index +=1
    index=0
    if args.style_mask_imgs is not None:
        for mask in args.style_mask_imgs:
            f.write ('style_masks['+str(index)+']:{}\n'.format(mask))
            index+=1
    f.write('init_type:{}\n'.format(args.init_img_type))
    f.write('content_weight:{}\n'.format(args.content_weight))
    f.write('style_weight:{}\n'.format(args.style_weight))
    f.write('tv_weight:{}\n'.format(args.tv_weight))
    f.write('content_layers:{}\n'.format(args.content_layers))
    f.write('style_layers:{}\n'.format(args.style_layers))
    f.write('optimizer_type:{}\n'.format(args.optimizer))
    f.write('max_iterations:{}\n'.format(args.max_iterations))
    f.write('max_image_size:{}\n'.format(args.max_size))
    
    
    

'''
image loading and processing
'''
def get_init_image(init_type,content_img,style_imgs,frame=None):
    if init_type=='content':
        return content_img
    elif init_type=='style':
        return style_img[0]
    elif init_type=='random':
        init_img=get_noise_image(args.noise_rate,content_img)
        return init_img
    
    
    elif init_type =='prev': #only for video frames*****
        init_img=get_prev_warped_frame(frame)
        return init_img
'''
def get_content_frame(frame):
    fn=args.content_frame_frmt.format(str(frame).zfill(4))
    path=os.path.join(args.video_input_dir,fn)
    img=read_image(path)
    return img
'''

def get_content_image(content_img):
    path=os.path.join(args.content_img_dir,content_img)
    print(path)
    img=cv2.imread(path,cv2.IMREAD_COLOR)
    check_image(img,path)
    img=img.astype(np.float32)
    h,w,d=img.shape
    mx=args.max_size
    if h > w and h > mx:
        w=(float(mx)/float(h))*w
        img=cv2.resize(img,dsize=(int(w),mx),interpolation=cv2.INTER_AREA)#
    if w > mx:
        h=(float(mx)/float(h))*w
        img=cv2.resize(img,dsize=(int(w),mx),interpolation=cv2.INTER_AREA)
    img=preprocess(img)
    
    return img


def get_style_images(content_img):
    _,ch,cw,cd=content_img.shape
    style_imgs=[]
    for _ in args.style_imgs:
        
        path=os.path.join(args.style_imgs_dir,args.style_imgs)# os.path.join(A,B)combining path"A\\B"
        
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        check_image(img,path)
        img=img.astype(np.float32)
        # in numpy, the float is deefault float 64 size 4, float32 is size 8
        img=cv2.resize(img,dsize=(cw,ch),interpolation=cv2.INTER_AREA)
        img=preprocess(img)
        style_imgs.append(img)

    return style_imgs

def get_noise_image(noise_ratio,content_img):
    np.random.seed(args.seed)
    noise_img=np.random.unifom(-20.,20.,content_img.shape).astype(np.float32)
    img=noise_ratio*noise_img+(1.-noise_ratio)*content_img
    return img

def get_mask_image(mask_img,width,height):
    path =os.path.join(args.content_img_dir,mask_img)
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    check_image(img,path)
    img=cv2.resize(img,dsize=(width,height),interpolation=cv2.INTER_AREA)
    img=img.astype(np.float32)
    mx=np.amax(img)
    img /=mx
    
    return img

'''
def get_prev_frame(frame):
    prev_frame=frame-1
    fn=args.content_frame_frmt.format(str(prev_frame).zfill(4))
    path=os.path.join(args.video_output_dir,fn)
    img=cv2.imread(path,cv2.IMREAD_COLOR)
    check_image(img,path)
    return img

def get_prev_warped_frame(frame):
    prev_img=get_prev_frame(frame)
    prev_frame=frame-1
    fn=args.backward_optical_flow_frmt.format(str(frame),str(prev_frame))
    path=os.path.join(args.video_input_dir,fn)
    flow=read_flow_file(path)
    warped_img=warp_image(prev_img,flow).astype(np.float32)
    img=preprocess(warped_img)
    return img

def get_content_weights(frame,prev_frame):
    forward_fn = args.content_weights_frmt.format(str(prev_frame), str(frame))
    backward_fn = args.content_weights_frmt.format(str(frame), str(prev_frame))
    forward_path = os.path.join(args.video_input_dir, forward_fn)
    backward_path = os.path.join(args.video_input_dir, backward_fn)
    forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    
    return forward_weights


def warp_image(scr,flow):
    _, h, w = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[1,y,:] = float(y) + flow[1,y,:]
    for x in range(w):
        flow_map[0,:,x] = float(x) + flow[0,:,x]
    # remap pixels to optical flow
    dst = cv2.remap(
                   src, flow_map[0], flow_map[1], 
                   interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    
    return dst
'''   

def convert_to_original_colors(content_img,stylized_img):
    content_img  = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if args.color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif args.color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif args.color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif args.color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = preprocess(dst)
    return dst

def render_single_image():  # do the image render
    content_img = get_content_image(args.content_img)
    style_imgs = get_style_images(content_img)
    with tf.Graph().as_default():  
        # # tf.Graph().as_default() Define the in 'with-body' Graph should be put into session to be run.
        # and with sentance is used to process the exception
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        init_img = get_init_image(args.init_img_type, content_img, style_imgs)
        tick = time.time()
        stylize(content_img, style_imgs, init_img)
        #the big function defined containing the session.run
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))
'''    
def render_video():         
       for frame in range(args.start_frame, args.end_frame+1):
            with tf.Graph().as_default():
                print('\n---- RENDERING VIDEO FRAME: {}/{} ----\n'.format(frame, args.end_frame))
                if frame == 1:
                    content_frame = get_content_frame(frame)
                    style_imgs = get_style_images(content_frame)
                    init_img = get_init_image(args.first_frame_type, content_frame, style_imgs, frame)
                    args.max_iterations = args.first_frame_iterations
                    tick = time.time()
                    stylize(content_frame, style_imgs, init_img, frame)
                    tock = time.time()
                    print('Frame {} elapsed time: {}'.format(frame, tock - tick))
                else:
                    content_frame=get_content_frame(frame)
                    init_img=get_init_image(args.init_frame_type,content_frame,style_imgs,frame)
                    args.max_iterations=args.frame_iteration
                    tick=time.tiem() # time????
                    stylize(content_frame,style_imgs,init_img,frame)
                    tock=time.time()
                    print('Frame{} elapsed time:{}'.format(frame,tock-tick))
'''
    

'''
*************************************************
Part V Main function
'''    
def main():   # the main function ,and the place where the code begin to run
    global args        # define a global variable
    args=parse_args()  # call method/function 'parse_args' 
    render_single_image()
        
if __name__=='__main__':
    main()




