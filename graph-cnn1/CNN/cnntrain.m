function net = cnntrain(net, x, y, opts)
% 该函数用于训练CNN。
% 生成随机序列，每次选取一个batch（50）个样本进行训练。
% 批训练：计算50个随机样本的梯度，求和之后一次性更新到模型权重中。
% 在批训练过程中调用：
% cnnff.m 完成前向过程
% cnnbp.m 完成误差传导和梯度计算过程
% cnnapplygrads.m 把计算出来的梯度加到原始模型上去
% net为网络,x是数据，y为训练目标，opts (options)为训练参數

    node = x{1};
    edge = x{2};
    
    
    % m为训练祥本的數量，size(x) = 28*28*60000
    m = size(node, 3);
    % batchsize为批训练时，一批所含图片样本数，numbatches是批量次数
    numbatches = m / opts.batchsize;
    % rem: Remainder after division. rem(x,y) is x - n.*y 相当于求余  
    % rem(numbatches, 1) 就相当于取其小数部分，如果为0，就是整数
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    % rL是最小均方误差的平滑序列，绘图要用
    net.rL = [];
    % 迭代训练
    for i = 1 : opts.numepochs
        % 显示训练到第几个epoch，一共多少个epoch
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        % tic 和 toc 是用来计时的，计算这两条语句之间所耗的时间
        tic;
        % P = randperm(N) 返回[1, N]之间所有整数的一个随机的序列，例如  
        % randperm(6) 可能会返回 [2 4 5 6 1 3]  
        % 这样就相当于把原来的样本排列打乱，再挑出一些样本来训练
        kk = randperm(m);
        for l = 1 : numbatches
            % 取出打乱顺序后的batchsize个样本和对应的标签
            batchNode = node(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batchEdge = edge(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)); 
%            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            batch_x{1} = batchNode;
            batch_x{2} = batchEdge;

            % 前向传播计算在当前的网络权值和网络输入下计算网络的输出
            net = cnnff(net, batch_x);
            % 得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值（也就是那些卷积核的元素）的导数
            net = cnnbp(net, batch_y);
            % 更新权值
            net = cnnapplygrads(net, opts);
            % net.L为模型的costFunction,即最小均方误差mse
            % net.rL是最小均方误差的平滑序列
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            % 保存历史的误差值，以便画图分析
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end
