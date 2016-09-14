function net = cnntrain(net, x, y, opts)
% �ú�������ѵ��CNN��
% ����������У�ÿ��ѡȡһ��batch��50������������ѵ����
% ��ѵ��������50������������ݶȣ����֮��һ���Ը��µ�ģ��Ȩ���С�
% ����ѵ�������е��ã�
% cnnff.m ���ǰ�����
% cnnbp.m ����������ݶȼ������
% cnnapplygrads.m �Ѽ���������ݶȼӵ�ԭʼģ����ȥ
% netΪ����,x�����ݣ�yΪѵ��Ŀ�꣬opts (options)Ϊѵ���Δ�

    node = x{1};
    edge = x{2};
    
    
    % mΪѵ���鱾�Ĕ�����size(x) = 28*28*60000
    m = size(node, 3);
    % batchsizeΪ��ѵ��ʱ��һ������ͼƬ��������numbatches����������
    numbatches = m / opts.batchsize;
    % rem: Remainder after division. rem(x,y) is x - n.*y �൱������  
    % rem(numbatches, 1) ���൱��ȡ��С�����֣����Ϊ0����������
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    % rL����С��������ƽ�����У���ͼҪ��
    net.rL = [];
    % ����ѵ��
    for i = 1 : opts.numepochs
        % ��ʾѵ�����ڼ���epoch��һ�����ٸ�epoch
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        % tic �� toc ��������ʱ�ģ��������������֮�����ĵ�ʱ��
        tic;
        % P = randperm(N) ����[1, N]֮������������һ����������У�����  
        % randperm(6) ���ܻ᷵�� [2 4 5 6 1 3]  
        % �������൱�ڰ�ԭ�����������д��ң�������һЩ������ѵ��
        kk = randperm(m);
        for l = 1 : numbatches
            % ȡ������˳����batchsize�������Ͷ�Ӧ�ı�ǩ
            batchNode = node(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batchEdge = edge(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)); 
%            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            batch_x{1} = batchNode;
            batch_x{2} = batchEdge;

            % ǰ�򴫲������ڵ�ǰ������Ȩֵ�����������¼�����������
            net = cnnff(net, batch_x);
            % �õ���������������ͨ����Ӧ��������ǩ��bp�㷨���õ���������Ȩֵ��Ҳ������Щ����˵�Ԫ�أ��ĵ���
            net = cnnbp(net, batch_y);
            % ����Ȩֵ
            net = cnnapplygrads(net, opts);
            % net.LΪģ�͵�costFunction,����С�������mse
            % net.rL����С��������ƽ������
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            % ������ʷ�����ֵ���Ա㻭ͼ����
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end
