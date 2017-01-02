function [E] = MeanSquaredError(yhat, yReal)
    N = size(yhat,1);
    K = size(yhat, 2);
    E = 0;
    for n = 1 : N
        for k = 1 : K
            E = E + (yReal(n, k) - yhat(n, k))^2;
        end
    end
    E = 1/2 * E;
end