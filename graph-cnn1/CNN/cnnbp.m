 function net = cnnbp(net, y)
    n = numel(net.layers);

    %   error
    net.e = net.o - y;
    %  loss function ���ۺ����Ǿ������MSE
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas
    % ���������Ĳв�������룩
    net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
    % �в� ���򴫲��� ǰһ�㣨���������
    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1}); % size(a{1}) = [4*4*50]��һ����a{1}~a{12}
    fvnum = sa(1) * sa(2); % fvnumһ��ͼ�����е�����������������4*4
    for j = 1 : numel(net.layers{n}.a)
        % ��fvd���汣���������������������������cnnff.m������������map���ɵģ�������������Ҫ����  
        % �任��������map����ʽ��d ������� delta��Ҳ���� ������ ���� �в�
        % size(net.layers{numLayers}.d{j}) = [4 * 4 * 50]��j��1��12��
        % size(net.fvd) = [192 * 50]
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    % ���� �����ǰ��Ĳ㣨����������в�ķ�ʽ��ͬ��
    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            % net.layers{l}.d{j} ������� ��l�� �� ��j�� map �� ������map�� Ҳ����ÿ����Ԫ�ڵ��delta��ֵ  
            % expand�Ĳ����൱�ڶ�l+1���������map�����ϲ�����Ȼ��ǰ��Ĳ����൱�ڶԸò������a����sigmoid��
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
    % ����������ȡ��(���+������)���ݶ�
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a) % featureMap������
                for i = 1 : numel(net.layers{l - 1}.a)
                    % ����˵��޸��� = ����ͼ�� * ���ͼ���delta
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                % net.layers.d{j}(:)��һ��24*24*50�ľ���db������50
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    % �����β�������֪�����ݶ�
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
