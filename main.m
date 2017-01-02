traindata = 'datasets\sinctrain25.dt';
validatedata = 'datasets\sincvalidate10.dt';

TrainM = dlmread(traindata);
ValidateM = dlmread(validatedata);

%%% 2 Hidden Neurons
[w1, yHat1, EList1, EValList1] = neuralNetwork(TrainM, 0.001, 1, 1, 2, 50000, 0.01, 0.000001, ValidateM);
[w3, yHat3, EList3, EValList3] = neuralNetwork(TrainM, 0.0001, 1, 1, 2, 50000, 0.01, 0.000001, ValidateM);
[w4, yHat4, EList4, EValList4] = neuralNetwork(TrainM, 0.1, 1, 1, 2, 50000, 0.01, 0.000001, ValidateM);

figure;
loglog(EList1);
hold on;
loglog(EList3);
loglog(EList4);
loglog(EValList1);
loglog(EValList3);
loglog(EValList4);
axis([0 inf 0 10000]);
title('Squared-mean loss of neural network with two hidden neurons');
legend('Rate = 0.001, Training', 'Rate = 0.0001, Training', 'Rate = 0.1, Training',  'Rate = 0.001, Validation', 'Rate = 0.0001, Validation', 'Rate = 0.1, Validation');
xlabel('Number of iterations');
ylabel('Squared-mean error');
hold off;


%%% 20 Hidden Neurons
[w1, yHat1, EList7, EValList7] = neuralNetwork(TrainM, 0.001, 1, 1, 20, 50000, 0.01, 0.000001, ValidateM);
[w3, yHat3, EList9, EValList9] = neuralNetwork(TrainM, 0.0001, 1, 1, 20, 50000, 0.01, 0.000001, ValidateM);
[w4, yHat4, EList10, EValList10] = neuralNetwork(TrainM, 0.1, 1, 1, 20, 50000, 0.01, 0.000001, ValidateM);

figure;
loglog(EList7);
hold on;
loglog(EList9);
loglog(EList10);
loglog(EValList7);
loglog(EValList9);
loglog(EValList10);
axis([0 inf 0 10000]);
title('Squared-mean loss of neural network with twenty hidden neurons');
legend('Rate = 0.001, Training', 'Rate = 0.0001, Training', 'Rate = 0.1, Training', 'Rate = 0.001, Validation', 'Rate = 0.0001, Validation', 'Rate = 0.1, Validation');
xlabel('Number of iterations');
ylabel('Squared-mean error');
hold off;
