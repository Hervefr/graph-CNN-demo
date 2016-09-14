 function net = cnnbp(net, y)
    n = numel(net.layers);

    %   error
    net.e = net.o - y;
    %  loss function 代价函数是均方误差MSE
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas
    % 计算输出层的残差（关于输入）
    net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
    % 残差 反向传播回 前一层（关于输出）
    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1}); % size(a{1}) = [4*4*50]，一共有a{1}~a{12}
    fvnum = sa(1) * sa(2); % fvnum一个图所含有的特征向量的数量，4*4
    for j = 1 : numel(net.layers{n}.a)
        % 在fvd里面保存的是所有样本的特征向量（在cnnff.m函数中用特征map拉成的），所以这里需要重新  
        % 变换回来特征map的形式。d 保存的是 delta，也就是 灵敏度 或者 残差
        % size(net.layers{numLayers}.d{j}) = [4 * 4 * 50]（j从1到12）
        % size(net.fvd) = [192 * 50]
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    % 对于 输出层前面的层（与输出层计算残差的方式不同）
    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            % net.layers{l}.d{j} 保存的是 第l层 的 第j个 map 的 灵敏度map。 也就是每个神经元节点的delta的值  
            % expand的操作相当于对l+1层的灵敏度map进行上采样。然后前面的操作相当于对该层的输入a进行sigmoid求导
            for j = 1 : numel(net.layers{l}.a)
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    % 计算特征抽取层(卷积+降采样)的梯度
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a) % featureMap的数量
                for i = 1 : numel(net.layers{l - 1}.a)
                    % 卷积核的修改量 = 输入图像 * 输出图像的delta
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                % net.layers.d{j}(:)是一个24*24*50的矩阵，db仅除于50
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    % 计算机尾部单层感知机的梯度
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
