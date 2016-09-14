function net = cnnsetup(net, x, y)
% �ú��������ڳ�ʼ��CNN�Ĳ�����
% ���ø����mapsize��С����ʼ��������ľ����ˡ�bias��β�������֪���Ĳ�������
% biasͳһ��ʼ��Ϊ0
% Ȩ������Ϊ��random(-1,1)/sqrt��6/��������Ԫ����+�����Ԫ������)
% ���ھ�����Ȩ�أ��������Ϊfan_in, fan_out
% fan_in�Ƕ�ÿһ�����map,������maps�����Ĳ�������
% fan_out�Ƕ�ÿһ������map,�����maps�����Ĳ�������
% net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));

    assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    % �����feature maps����
    inputmaps = 2; 
    % B=squeeze(A) ���غ;���A��ͬԪ�ص����е�һά���Ƴ��ľ���B����һά������size(A,dim)=1��ά��  
    % train_x��ͼ��Ĵ�ŷ�ʽ����ά��reshape(train_x',28,28,60000)��ǰ����ά��ʾͼ��������У�  
    % ����ά�ͱ�ʾ�ж��ٸ�ͼ������squeeze(x(:, :, 1))���൱��ȡ��һ��ͼ���������ٰѵ���ά  
    % �Ƴ����ͱ����28x28�ľ���Ҳ���ǵõ�һ��ͼ����sizeһ�¾͵õ���ѵ������ͼ��������������� 
    node = x{1};
    edge = x{2};
    mapsize = size(squeeze(edge(:, :, 1)));

    % ����ͨ������net����ṹ������㹹��CNN����  
    % n = numel(A)��������A��Ԫ�ظ���  
    % net.layers�������struct���͵�Ԫ�أ�ʵ���Ͼͱ�ʾCNN������㣬���ﷶΧ����5
    % net.layers{i}.type = s ��ʾ����һ���Ӳ����㣬= i ��ʾ������㣬= c ��ʾ�Ǿ�����
    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's') % ���������Ӳ�����
            % subsampling���mapsize���ʼmapsize��ÿ��ͼ�Ĵ�С28*28  
            % �������scale=2������pooling֮��ͼ�Ĵ�С��pooling��֮��û���ص�������pooling���ͼ��Ϊ14*14  
            % ע��������ұߵ�mapsize����Ķ�����һ��ÿ������map�Ĵ�С����������ѭ�����в��ϸ���
            mapsize = mapsize / net.layers{l}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            % inputmap������һ���ж���������ͼ��pooling���ı����ֵ
            for j = 1 : inputmaps 
                net.layers{l}.b{j} = 0; % ��ƫ�ó�ʼ��Ϊ0
                % �²�����Ϊʲôֻ��ƫ��b��û��ȨֵW�أ���������������������������������������������
            end
        end
        if strcmp(net.layers{l}.type, 'c') % �������Ǿ�����
            % �ɵ�mapsize���������һ�������map�Ĵ�С����ô��������˵��ƶ�������1������  
            % kernelsize*kernelsize��С�ľ����˾�����һ�������map�󣬵õ����µ�map�Ĵ�С��������������
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            % �ò���Ҫѧϰ�Ĳ���������ÿ������map��һ��(�������ͼ����)*(����������patchͼ�Ĵ�С)  
            % ��Ϊ��ͨ����һ���˴�������һ������map�����ƶ����˴���ÿ���ƶ�1�����أ���������һ������map  
            % ���ÿ����Ԫ���˴�����kernelsize*kernelsize��Ԫ����ɣ�ÿ��Ԫ����һ��������Ȩֵ������  
            % ����kernelsize*kernelsize����Ҫѧϰ��Ȩֵ���ټ�һ��ƫ��ֵ�����⣬������Ȩֵ������Ҳ����  
            % ˵ͬһ������map������ͬһ��������ͬȨֵԪ�ص�kernelsize*kernelsize�ĺ˴���ȥ����������һ  
            % ������map���ÿ����Ԫ�õ��ģ�����ͬһ������map������Ȩֵ��һ���ģ������ģ�Ȩֵֻȡ����  
            % �˴��ڡ�Ȼ�󣬲�ͬ������map��ȡ������һ������map�㲻ͬ�����������Բ��õĺ˴��ڲ�һ����Ҳ  
            % ����Ȩֵ��һ��������outputmaps������map���У�kernelsize*kernelsize+1��* outputmaps��ô���Ȩֵ��  
            % ������fan_outֻ��������˵�ȨֵW��ƫ��b������������档
            % fan_out�Ƕ�����һ���һ������map������Ӧ����Ҫѧϰ�Ĳ���������
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                % fan_in�Ƕ���ÿһ����ǰ������ͼ���ж��ٸ���������ǰ��
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  input map
                    % �����ʼ��Ȩֵ��Ҳ���ǹ���outputmaps�������ˣ����ϲ��ÿ������map������Ҫ����ô���������ȥ������ȡ������  
                    % rand(n)�ǲ���n��n�� 0-1֮�����ȡֵ����ֵ�ľ����ټ�ȥ0.5���൱�ڲ���-0.5��0.5֮��������  
                    % �� *2 �ͷŴ� [-1, 1]��Ȼ���ٳ��Ժ�����һ����why��  
                    % �������ǽ�������ÿ��Ԫ�س�ʼ��Ϊ[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]  
                    % ֮������������Ϊ������Ȩֵ�����ģ�Ҳ���Ƕ���һ������map�����и���Ұλ�õľ����˶���һ����  
                    % ����ֻ��Ҫ������� inputmaps * outputmaps �������ˡ�
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                % ƫ�ú;����˲�ͬ��ֻ��Ӧoutputmaps
                net.layers{l}.b{j} = 0; 
            end
            % ֻ���ھ������ʱ��Ż�ı�����map�ĸ�����pooling��ʱ�򲻻�ı������������������map��������  
            % ���뵽��һ�������map����
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    
    %% β�������֪����ȫ���Ӳ㣩�Ĳ�����Ȩ�غ�ƫ��������
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    % fvnum ��������ǰ��һ�����Ԫ������  
    % ��һ�����һ���Ǿ���pooling��Ĳ㣬������inputmaps������map��ÿ������map�Ĵ�С��mapsize��  
    % ���ԣ��ò����Ԫ������ inputmaps * ��ÿ������map�Ĵ�С��  
    % prod: Product of elements.  
    % For vectors, prod(X) is the product of the elements of X  
    % ������ mapsize = [����map������ ����map������]������prod����� ����map����*��  
    fvnum = prod(mapsize) * inputmaps;
    % onum �Ǳ�ǩ�ĸ�����Ҳ�����������Ԫ�ĸ�������Ҫ�ֶ��ٸ��࣬��Ȼ���ж��ٸ������Ԫ
    onum = size(y, 1);

    % ���������һ����������趨  
    % ffb �������ÿ����Ԫ��Ӧ�Ļ�biases 
    net.ffb = zeros(onum, 1);
    % ffW �����ǰһ�� �� ����� ���ӵ�Ȩֵ��������֮����ȫ���ӵģ�ע�����У�
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end