function test_example_CNN
% 1 test_example_CNN 设置CNN的基本参数规格，如卷积、降采样层的数量，卷积核的大小、降采样的降幅
% 2 cnnsetup函数 初始化卷积核、偏置等
% 3 cnntrain函数 训练cnn，把训练数据分成batch，然后调用
%   3.1 cnnff 完成训练的前向过程
%   3.2 cnnbp计算并传递神经网络的error，并计算梯度（权重的修改量）
%   3.3 cnnapplygrads 把计算出来的梯度加到原始模型上去
% 4 cnntest 函数，测试当前模型的准确率

clear all; close all; clc; 
addpath('../data');  
addpath('../util'); 


% 加载数据
node1 = load('1_node.txt');
node1 = node1';
[row, column] = size(node1);
edge = [];
class = 1;
dataY = [];
for i = 1 : class
    oneHot = zeros(1,class);
    oneHot(i) = 1;
    filename = [num2str(i) 'edge.txt'];
    edge1 = load(filename);
    for j = 1: size(edge1,1)
        dataY = [dataY;oneHot];
    end
    edge = [edge; edge1];
end
totalNum = size(edge,1);
for i = 1: totalNum
    node(:,:,i)= node1;
end
% reshape数据集，train_x中图像的存放方式是三维的reshape(train_x',28,28,60000)，
% 前面两维表示图像的行与列，第三维就表示有多少个图像,注意这里必须先转置,因为reshape是按列操作的
edge = reshape(edge',row,column,totalNum);   % edge需要归一化吗？？？？？？？？？？？？？？？？？
dataY = dataY';

%% 这样得到数据集：node/edge:N*N*M， dataY:class*M


% 产生随机序列
shuffled = randperm(totalNum);
trainNum = int32(totalNum * 0.9);
for i = 1 : trainNum
    trainNode(:,:,i) = node(:,:,shuffled(i));
    trainEdge(:,:,i) = edge(:,:,shuffled(i));
    trainY(:,i) = dataY(:,shuffled(i));
end

for i = trainNum+1 : totalNum
    testNode(:,:,i - trainNum) = node(:,:,shuffled(i));
    testEdge(:,:,i - trainNum) = edge(:,:,shuffled(i));
    testY(:,i - trainNum) = dataY(:,shuffled(i));
end

        
train_x{1} = trainNode;
train_x{2} = trainEdge;
test_x{1} = testNode;
test_x{2} = testEdge;
train_y = trainY;
test_y = testY;


%% 设置网络结构及训练参数
%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

% 定义CNN层的情况（cnn是一个结构体，这里定义cnn.layers有五个struct类型的元素）
% type = s 表示这是一个子采样层，= i 表示是输入层，= c 表示是卷积层
% outputmaps是卷积层输出的maps个数
% kernelsize是卷积层的卷积核大小
% scale定义下采样层的缩放尺寸n（scale*scale -> 1*1）
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

%% 训练选项，alpha学习效率，batchsiaze批训练总样本的数量，numepoches迭代次数（训练次数）
opts.alpha = 1;
% 每次挑出一个batchsize的batch来训练，也就是每用batchsize个样本就调整一次权值，而不是  
% 把所有样本都输入了，计算所有样本的误差了才调整一次权值
opts.batchsize = 50;
opts.numepochs = 1;

%% 初始化网络，对数据进行批训练，验证模型准确率
% 这里把cnn的设置给cnnsetup，它会据此构建一个完整的CNN网络，并返回
cnn = cnnsetup(cnn, train_x, train_y);
% 然后开始把训练样本给它，开始训练这个CNN网络
cnn = cnntrain(cnn, train_x, train_y, opts);
% 然后就用测试样本来测试
[er, bad] = cnntest(cnn, test_x, test_y);
disp(er);
%plot mean squared error绘制均方误差曲线
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
