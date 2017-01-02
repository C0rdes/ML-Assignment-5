function [z, a] = hiddenNeuron(i, w, z, minInputIndex, maxInputIndex)
    % i = The index of this neuron
    % w = Weight matrix
    % z: Output of neurons
    % minInputIndex: The first neuron this should get input from.
    % maxInputIndex: The last neuron this should get input from
    
    % First, get bias
    N = size(z, 1);
    
    a = w(i, 1)*z(:, 1);
    for j = minInputIndex : maxInputIndex
        a = a + w(i, j)*z(:, j);
    end
    
    for j = 1 : N
    z(j, i) = alternativeSigmoid(a(j, 1));
    end
end