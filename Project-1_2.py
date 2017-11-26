
# coding: utf-8

# In[ ]:



# 6 parts: 1解析/定义参数；2输入数据；3网络搭建；4损耗及优化；5数据输出；6模块入口及目标函数


#python standard library
import argparse
import struct #be used in handling binary data
import errno
#import time
import os
#other library
import tensorflow as tf 
import numpy as ny
import cv2 # Open Source Computer Vision Library from Opencv
#the scipy.misc/scipy.io also can do some I/O work. eg:imread....
import scipy.io # this is a model contained in scipy, not a class.


# I 解析命令行/ 参数定义
def parse_args():
        parser=argparse.ArgumentParser(description="tensorflow Neural-Algorithm for Artistic Style ") 
        
        parser.add_argument('--verbose',action='store_true', 
                            # if the command-line shows "--verbose", 
                            #then argument "varbose" will be assigned to be 1(which means true) ！！！！！
                           help='Boolean flag indicating if statements should be printed to the console,')
                            # Call the methond of class ArgumentParser
            
        # input/loading & preprocessing  
        parser.add_argument('--img_name',type=str,default='result',#！the name can be changed!
                           help='Filename of the output image.')
        
        parser.add_argument('--style_imgs',nargs='+',type=str,default='woman.jpg',
                           help='Filename of the style images (example:starry-nigth.jpg)')
        
        parser.add_argument('--style_imgs_weights',nargs='+',type=float,default=[1.0],
                           help='Interpolation weight of each of the style images.(example:0.5 0.5)')
        
        parser.add_argument('--content_img',type=str, default='lion.jpg',
                           help='Filename of the content image (example:line.jpg)')
        
        parser.add_argument('--style_imgs_dir',type=str, 
                            default="C:\\Users\\Mason\\desktop\\styles",
                           help='Directory path to the style images.(default: %(default)s)')
            
        parser.add_argument('--content_img_dir',type=str,
                            default='C:\\Users\\Mason\\desktop\\contents',
                           help='Directory path to the content image.(default:%(default)s)')
        
        parser.add_argument('--init_img_type',type=str,default='content',
                            choices=['random','content','style'],
                           help='Image used to initialize the network.(default:%(default)s)')
        
        parser.add_argument('--max_size',type=int,default=512,
                           help='Maximum width or height of the input images.(default:%(default)s)')
        
        parser.add_argument('--noise_ratio',type=float,default=1,
                           help="Tnterpolation value, if network initialized with 'random'.")
                           # Tnterpolation value between content and noise image        
 
        parser.add_argument('--seed',type=int, default=0,
                           help='Seed for the random number generator.(default:%(default)s)')


        # network build 
        parser.add_argument('--content_layers',nargs='+',type=str,default=['conv4_2'],
                           help='VGG19 layers used for the content image.(default:%(default)s)') 
        
        parser.add_argument('--style_layers',nargs='+',type=str,
                            default=['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'],
                           help='VGG19 layers used for the style image.(default:%(default)s)')

        parser.add_argument('--model_weights',type=str,default='imagenet-vgg-verydeep-19.mat',
                            #imagenet-vgg-verydeep-19.mat need download from web.
                           help='Weights and biases of the VGG-19 network')
        
        parser.add_argument('--pooling_type',type=str,default='avg',
                           help='type of pooling in convolutional neural network.(default:%(default)s)')
 
        
        # Loss function 
        parser.add_argument('--style_weight',type=float,default=1e4,
                           help='Weight for the style loss function.(default:%(default)s)')

        parser.add_argument('--content_weight',type=float,default=5e0,
                           help='Weight for the content loss function.(defaule:(%default)s)')
 
        parser.add_argument('--tv_weight',type=float,default=1e-3,
                           help='Weight for the total variational loss function.(default:%(default)s)')
        
        parser.add_argument('--content_loss_function',type=int,default=1,choices=[1,2,3],
                           help='type of the content layer loss function.(default:%(default)s)')
        
        parser.add_argument('--style_layer_weights',nargs='+',type=float,default=[0.2,0.2,0.2,0.2,0.2],
                           help='Contributions of each content layer to loss.(defaule:%(default)s)')
        
        parser.add_argument('--content_layer_weights',nargs='+',type=float,default=[1.0],
                           help='Contributions of each content layer to loss.(defaule:%(default)s)')
        
        # output 
        parser.add_argument('--origional_colors',action='store_true',
                           help='if true, then transfer only the style，not the colors.')
        
        parser.add_argument('--color_convert_time',type=str,default='after',
                            choices=['after','before'],
                           help='Time (before or after) to convert to origional colors.')
        
        parser.add_argument('--color_convert_type',type=str,default='yuv',
                            choices=['yuv','ycrcb','luv','lab'],
                           help='Color space for conversion to origional colors.')
               
        parser.add_argument('--device',type=str,default='/cpu:0',
                            choices=['/gpu:0','/cpu:0'],
                            #the :0  means the first cpu/gpu, since you may have more than one cpu/gpu.
                           help='GPU or CPU mode.GPU mode requires NVIDIA CUDA.')
        
        parser.add_argument('--img_output_dir',type=str,default='C:\\Users\\Mason\\Desktop\\output',
                           help='Relative or absolute directory path to output image and data.')
        
        
        #optimizations
        parser.add_argument('--optimizer',type=str,default='lbfgs',
                            choices=['lbfgs','adam'],
                           help='Loss minimization optimizer.' )
                           # lbfgs gives better results. adam uses less memory.
            
        parser.add_argument('--learning_rate',type=float,default=1e0,
                           help='Learning rate for the adam optimizer.(default:%(default)s)')
        
        parser.add_argument('--print_iterations',type=int,default=10,
                           help='Number of iterations between optimizer print statements.')
        
        parser.add_argument('--max_iterations',type=int,default=100,
                           help='Max number of iterations for the adam/lbfgs optimizer.')
        
        #args is a  namespace object
        args=parser.parse_args()
        
        #normolization for loss function
        args.style_layer_weights = normalize(args.style_layer_weights)
        args.content_layer_weights = normalize(args.content_layer_weights)
        args.style_imgs_weights = normalize(args.style_imgs_weights) 
        
        def normalize(weigths):
            denom=sum(weights)
            if denom>0:
                return [float(i)/denom for i in weigths]
            else:
                return [0.]*len(weights)
            
        # parse_args ‘s return
        return args


## II 加载和处理image input/loading
def get_init_image(init_type,content_img,style_imgs,frame=None):
    if init_type=='content':
        return content_img
    elif init_type=='style':
        return style_img[0]
    elif init_type=='random':
        init_img=get_noise_image(args.noise_rate,content_img)
        return init_img
    
def get_noise_image(noise_ratio,content_img):
    np.random.seed(args.seed)
    noise_img=np.random.unifom(-20.,20.,content_img.shape).astype(np.float32)
    img=noise_ratio*noise_img+(1.-noise_ratio)*content_img
    return img


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

def check_image(img,path):
    if img is None:
       raise OSError(errno.ENOENT,"No such file", path)

def get_style_images(content_img):
    _,ch,cw,cd=content_img.shape
    style_imgs=[]
    for _ in args.style_imgs:       
        path=os.path.join(args.style_imgs_dir,args.style_imgs)
        # os.path.join(A,B)combining path"A\\B"        
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        check_image(img,path)
        img=img.astype(np.float32)
        # in numpy, the float is deefault float 64 size 4, float32 is size 8
        img=cv2.resize(img,dsize=(cw,ch),interpolation=cv2.INTER_AREA)
        img=preprocess(img)
        style_imgs.append(img)
    return style_imgs

def preprocess(img):
    img=img[...,::-1]
    # python list slice[start:end:step] to create a new sequence, -1 mean revers direction. 
    img=img[np.newaxis,:,:,:] 
    # shape(h,w,d) to (1,h,w,d)
    # add a new axis in list, np.newaxis equal to None,ima[None,:,:,:]
    img -= np.array([123.68,116.779,103.939]).reshape((1,1,1,3))
    #numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0).
    # np.arrar([1.0,2.0,3.0]).reshape((1,1,1,3)) results transfer 
    #array to 4-Dimensional 1x1x1x3 matrix [[[[1.0,2.0,3.0]]]]
    return img





### III 搭建神经网络
def build_model(input_img):  
    if args.verbose: print('\nBUILDING VGG-19 NETWORK')
    # if "verbose" is true then print all if and do statements
    net={}
     _,h,w,d =input_img.shape        
    if args.verbose: print('loading model weights...')
    vgg_rawnet=scipy.io.loadmat(args.model_weights)
    # scipy.io.loadmat(), used to load a matlab file, 
    #and args.model_weights is point to file"imagenet-vgg-verydeep-19.mat" 
    #and the layer_data is 5D，4D for image data, and 1D for layer number 0-42. 
    # that is known from the .mat file    
    vgg_layers=vgg_rawnet['layers'][0]
    # this additional [0] is weights (0 is Weight W; 1 is bias)
    #, that makes vgg_layers to be 6D,[][][][][][]   
    
    if arg.verbose: print('constructing layers...')
    net['input']=tf.Variable(np.zeros((1,h,w,d),dtypye=np.float32))
    #tf.variable():to create a tensor variable ,
    #and numpy.zeros():Return a new array of given shape and type, filled with zeros.    
    
    if args.verbose: print('LAYER GROUP 1')
    # the net of built model, is actually a slight change from VGG19 
    # the conv_layers and relu_layers is compluted by tensorflow, 
    #and the arguments conv{}_{} are initialized by tf.variable(np.zeros())
   

    net['conv1_1'] =conv_layer('conv1_1',net['input'],W=get_weights(vgg_layers,0)) 
    # VGG19 has 43 layers, and begin from 0(0~42)
    net['conv1_1'] =relu_layer('relu1_1',net['conv1_1'],b=get_biases(vgg_layers,0))
    net['conv1_2'] =conv_layer('conv1_2',net['relu1_1'],W=get_weights(vgg_layers,2))
    net['conv1_2'] =relu_layer('relu1_2',net['conv2_1'],b=get_biases(vgg_layers,2))
    net['pool1'] = pool_layer('pool1',net['relu1_2'])
        
    if args.verbose: print('LAYER GROUP 2')
    net['conv2_1'] =conv_layer('conv2_1',net['pool1'],W=get_weights(vgg_layers,5))
    net['conv2_1'] =relu_layer('relu2_1',net['conv2_1'],b=get_biases(vgg_layers,5))
    net['conv2_2'] =conv_layer('conv2_2',net['relu2_1'],W=get_weights(vgg_layers,7))
    net['conv2_2'] =relu_layer('relu2_2',net['conv2_2'],b=get_biases(vgg_layers,7))
    net['pool2'] = pool_layer('pool2',net['relu2_2'])
        
    if args.verbose: print('LAYER GROUP 3')
    net['conv3_1'] =conv_layer('conv3_1',net['pool2'],W=get_weights(vgg_layers,10))
    net['conv3_1'] =relu_layer('relu3_1',net['conv3_1'],b=get_biases(vgg_layers,10))
    net['conv3_2'] =conv_layer('conv3_2',net['relu3_1'],W=get_weights(vgg_layers,12))
    net['conv3_2'] =relu_layer('relu3_2',net['conv3_2'],b=get_biases(vgg_layers,12))
    net['conv3_3'] =conv_layer('conv3_3',net['relu3_2'],W=get_weights(vgg_layers,14))
    net['conv3_3'] =relu_layer('relu3_3',net['conv3_3'],b=get_biases(vgg_layers,14))
    net['conv3_4'] =conv_layer('conv3_4',net['relu3_3'],W=get_weights(vgg_layers,16))
    net['conv3_4'] =relu_layer('relu3_4',net['conv3_4'],b=get_biases(vgg_layers,16))
    net['pool3'] = pool_layer('pool3',net['relu3_4'])
    
    if args.verbose: print('LAYER GROUP 4')
    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))
    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))
    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))
    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))
    net['pool4']   = pool_layer('pool4', net['relu4_4'])

    if args.verbose: print('LAYER GROUP 5') # the layer of built model total contains 5 layers.
    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))
    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))
    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))
    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))
    net['pool5']   = pool_layer('pool5', net['relu5_4'])
        
    return net

def conv_layer(layer_name,layer_input,W):
    conv = tf.nn.conv2d(layer_input,W,strides=[1,1,1,1],padding='SAME') # W weights  is filter
    # strides:A list of ints. 1-D tensor of length 4.  
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
        pool =tf.nn.vag_pool(layer_input,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        # tf.nn.vag_pool:Performs the average pooling on the input.
        # ksize: A 1-D int Tensor of 4 elements. The size of the window for each dimension of the input tensor.
    elif args.pooling_type=='max':
        pool=tf.nn.max.pool(layer_input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    if args.verbose:
        print('--{}|shape={}'.format(layer_name,pool.get_shape()))
    return pool

def get_weight(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0] 
    #vgg_layers consist of 6-D? using 6 dimensions list.
    # first [i] means ith layer,  but what is the meaning of data [0][0][2][0]? maybe its doesnt matter.
    W=tf.constant(weights)#maping the 'layer' of .mat file
    return W #return a filter data, 4D-list (* 3x3 matrixes, and *=d multiply _). 
            #the vgg19 met file: layer1_i.weight{1,1} 

def get_bias(vgg_layers,i)
    bias=vegg_layers[i][0][0][2][0][1]
    b=tf.constant(np.reshape(bias,(bias.size)))    
    return b#return a 1D array (1x_), layer-data is (_,h,w,d)



#### IV 损耗及优化

#loss
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

def style_layer_loss(a,x):
    _,h,w,d=a.get_shape()
    M=h.value*W.value
    N=d.value
    A=gram_matrix(a,M,N)
    G=gram_matrix(x,M,N)
    loss=(1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
    return loss

def gram_matrix(x,area,depth):
    F=tf.reshape(x,(area,depth))
    G=tf.matmul(tf.transpose(F),F)
    return G

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

def content_layer_loss(p,x):
    _,h,w,d=p.get_shape()
    M=h.value*w.value
    N=d.value
    if agrs.content_loss_function ==1:
        K=1./(2. * N**0.5 * M**0.5) #the ** means exponential operation
    elif args.content_loss_function ==2:
        K=1./(N*M)
    elif args.content_loss_function ==3:
        K=1./2.
    loss=K*tf.reduce_sum(tf.pow((x - p),2))
    return loss

#Optimization
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
        optimizer=tf.train.AdamOptimizer(args.learning_rate    return optimizer



##### V 输出 image output
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
                                           
def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
       os.makedirs(dir_path)

def write_image(path,img):
    img=postprocess(img)
    cv2.imwrite(path,img)
        
###### VI 模块入口，main（）& Target Function
#target function
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
    #output
            write_image_output(output_img,content_img,style_imgs,init_img)
 
def target()
    stylize()
    
#main（），place where the code begin to run
def main():   
    global args        # define a global variable
    args=parse_args()  # call method/function 'parse_args' 
    target()
        
if __name__=='__main__':
    main()

