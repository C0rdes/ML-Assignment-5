function [h] = alternativeSigmoidDerivative(a)
    h = 1/ (1 + abs(a))^2;
end