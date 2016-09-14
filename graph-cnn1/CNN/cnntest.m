function [er, bad] = cnntest(net, x, y)
% 验证测试样本的准确率

    %  feedforward
    % 前向传播得到输出
    net = cnnff(net, x);
    % [Y,I] = max(X) returns the indices of the maximum values in vector I
    % 找到最大的输出对应的标签
    [~, h] = max(net.o);
    % 找到最大的期望输出对应的索引
    [~, a] = max(y);
    % 找到他们不相同的个数，也就是错误的次数
    bad = find(h ~= a);

    % 计算错误率 
    er = numel(bad) / size(y, 2);
    
    tmp = net.o;
    save('../data/test.mat','tmp');
    save('../data/error.mat','er');
end
