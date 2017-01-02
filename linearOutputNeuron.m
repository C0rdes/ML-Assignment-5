function [z] = linearOutputNeuron(i, w, z, minInputIndex, maxInputIndex)
        
    a = w(i, 1)*z(:, 1);
    for j = minInputIndex : maxInputIndex
        a = a + w(i, j)*z(:, j);
    end
    
    
    z(:, i) = a;
end