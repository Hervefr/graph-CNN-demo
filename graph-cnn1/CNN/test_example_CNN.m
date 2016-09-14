function test_example_CNN
% 1 test_example_CNN ����CNN�Ļ�����������������������������������˵Ĵ�С���������Ľ���
% 2 cnnsetup���� ��ʼ������ˡ�ƫ�õ�
% 3 cnntrain���� ѵ��cnn����ѵ�����ݷֳ�batch��Ȼ�����
%   3.1 cnnff ���ѵ����ǰ�����
%   3.2 cnnbp���㲢�����������error���������ݶȣ�Ȩ�ص��޸�����
%   3.3 cnnapplygrads �Ѽ���������ݶȼӵ�ԭʼģ����ȥ
% 4 cnntest ���������Ե�ǰģ�͵�׼ȷ��

clear all; close all; clc; 
addpath('../data');  
addpath('../util'); 


% ��������
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
% reshape���ݼ���train_x��ͼ��Ĵ�ŷ�ʽ����ά��reshape(train_x',28,28,60000)��
% ǰ����ά��ʾͼ��������У�����ά�ͱ�ʾ�ж��ٸ�ͼ��,ע�����������ת��,��Ϊreshape�ǰ��в�����
edge = reshape(edge',row,column,totalNum);   % edge��Ҫ��һ���𣿣�������������������������������
dataY = dataY';

%% �����õ����ݼ���node/edge:N*N*M�� dataY:class*M


% �����������
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


%% ��������ṹ��ѵ������
%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

% ����CNN��������cnn��һ���ṹ�壬���ﶨ��cnn.layers�����struct���͵�Ԫ�أ�
% type = s ��ʾ����һ���Ӳ����㣬= i ��ʾ������㣬= c ��ʾ�Ǿ����
% outputmaps�Ǿ���������maps����
% kernelsize�Ǿ����ľ���˴�С
% scale�����²���������ųߴ�n��scale*scale -> 1*1��
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

%% ѵ��ѡ�alphaѧϰЧ�ʣ�batchsiaze��ѵ����������������numepoches����������ѵ��������
opts.alpha = 1;
% ÿ������һ��batchsize��batch��ѵ����Ҳ����ÿ��batchsize�������͵���һ��Ȩֵ��������  
% �����������������ˣ�������������������˲ŵ���һ��Ȩֵ
opts.batchsize = 50;
opts.numepochs = 1;

%% ��ʼ�����磬�����ݽ�����ѵ������֤ģ��׼ȷ��
% �����cnn�����ø�cnnsetup������ݴ˹���һ��������CNN���磬������
cnn = cnnsetup(cnn, train_x, train_y);
% Ȼ��ʼ��ѵ��������������ʼѵ�����CNN����
cnn = cnntrain(cnn, train_x, train_y, opts);
% Ȼ����ò�������������
[er, bad] = cnntest(cnn, test_x, test_y);
disp(er);
%plot mean squared error���ƾ����������
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
