function [a] = alternativeSigmoid(a)
    
        a = a / (1+abs(a));
    
end