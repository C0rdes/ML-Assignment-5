function [w, yHattest, TrainingElist, ValidationEList] = neuralNetwork(TrainingData, rate, d, K, M, MaxIterations, minError, minDeltaE, ValidationData)
    % get number of data points
    N = size(TrainingData, 1);
    NVal = size(ValidationData, 1);
    % Calculate index of where the
    dStart = 2;
    dEnd = d+1;
    MStart = d+2;
    MEnd = d+M+1;
    KStart = d+M+2;
    KEnd = d+M+K+1;
    % Get target values
    Ystrain = TrainingData(:, d+1);
    Ysval = ValidationData(:, d+1);
    % Prepare error values
    TrainingElist = [];
    ValidationEList = [];
    Etrain = 9999999999999;
    Eval = 99999999999999; 
    deltaE = Inf;
    % Get total number of neurons
    numNeurons = d + M + K;
    % initialize weights
    w = randn(numNeurons+1);
    % Initialize iteration counter
    Iterations = 0;
    % initialize z values
    z = zeros(N, numNeurons+1);
    zvalidation = zeros(NVal, numNeurons+1); 
    % save bias to z
    z(:, 1) = neuronBias;
    zvalidation(:, 1) = neuronBias;
    % Initialize error values
    
    
    % Fire input neurons on training set
    for i = 1 : d
       z(:, i+1) = inputNeuron(i, TrainingData);
       zvalidation(:, i+1) = inputNeuron(i, ValidationData);
    end
    

    
    while (Iterations < MaxIterations) && (Etrain > minError) && (deltaE > minDeltaE)
        a = zeros(N, M);
        Iterations = Iterations + 1;
        % Fire hidden neuron layer for training and validation set
        for i = MStart : MEnd
            [zvalidation, ~] = hiddenNeuron(i, w, zvalidation, 2, d+1);
            [z, aout] = hiddenNeuron(i, w, z, 2, d+1);
            a(:, i-d-1) = aout;
        end
       
        
        % Fire output neurons
        for i = KStart : KEnd
            z = linearOutputNeuron(i, w, z, d+2, d+M+1); 
            zvalidation = linearOutputNeuron(i, w, zvalidation, d+2, d+M+1);
        end
        
        yHattest = z(:, KStart:KEnd);
        yHatvalidation = zvalidation(:, KStart:KEnd);
        Eold = Etrain;
        Etrain = MeanSquaredError(yHattest, Ystrain);
        Eval = MeanSquaredError(yHatvalidation, Ysval);

        deltaE = abs(Etrain - Eold);
        TrainingElist = [TrainingElist; Etrain];
        ValidationEList = [ValidationEList; Eval];
        
        % Do backpropagation
        delta = zeros(N, M+K);
        % Start with output units
        for i = KEnd :-1: KStart
            delta(:, i-d-1) = z(:, i) - Ystrain(:, i-M-d-1);
        end
        
        % Next, we calculate for hidden ones
        for i = MEnd :-1: MStart
            Sum = 0;
            for k = i+1 : KEnd
                Sum = Sum + w(k, i)*delta(:, k-d-1);
            end
            hprime = arrayfun(@alternativeSigmoidDerivative, a(:, i-d-1));
           delta(:, i-d-1) = hprime.*Sum;
        end
        
        % Lastly, we update the weights
        for i = KStart : KEnd
            for j = MStart : MEnd
                w(i, j) = w(i, j) - rate * dot(delta(:, i-d-1), z(:, j));
            end
            w(i, 1) = w(i, 1) - rate * dot(delta(:, i-d-1), z(:, j));
        end
        
        for i = MStart : MEnd
            for j = dStart : dEnd
                w(i, j) = w(i, j) - rate * dot(delta(:, i-d-1), z(:,j)); 
            end
            w(i, 1) = w(i, 1) - rate * dot(delta(:, i-d-1), z(:, j));
        end
        
    end
    yHattest = z(:, d+M+2:d+M+K+1);
    Etrain = MeanSquaredError(yHattest, Ystrain);
    TrainingElist = [TrainingElist; Etrain];
end