function [er, bad] = cnntest(net, x, y)
% ��֤����������׼ȷ��

    %  feedforward
    % ǰ�򴫲��õ����
    net = cnnff(net, x);
    % [Y,I] = max(X) returns the indices of the maximum values in vector I
    % �ҵ����������Ӧ�ı�ǩ
    [~, h] = max(net.o);
    % �ҵ��������������Ӧ������
    [~, a] = max(y);
    % �ҵ����ǲ���ͬ�ĸ�����Ҳ���Ǵ���Ĵ���
    bad = find(h ~= a);

    % ��������� 
    er = numel(bad) / size(y, 2);
    
    tmp = net.o;
    save('../data/test.mat','tmp');
    save('../data/error.mat','er');
end
