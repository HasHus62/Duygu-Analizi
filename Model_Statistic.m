function [ output ] = Model_Statistic(XNew_Labels,labelsNew)
    
output.XNew_Labels = XNew_Labels;
output.labelsNew=labelsNew;
    %actual(Y) class değişkenine atandı.
    Class = categories(XNew_Labels);
    NumberOfInstanceAsClass = countcats(XNew_Labels);
    NumberOfClass = length(Class);
    ROC = zeros(NumberOfClass, 1);
    
    %örnek sayısı NumberofInstance değişkenine atandı.
    NumberOfInstance = length(XNew_Labels);
    output.NOI = NumberOfInstance;
    
    %Confusion Matrix oluşturuldu.
    ConfusionMatrix = confusionmat(XNew_Labels, labelsNew);
    
    %True Positive değeri hesaplandı.
    TP = diag(ConfusionMatrix) ./ sum(ConfusionMatrix,2);
    
    %False Positive değeri hesaplandı.
    FP = (sum(ConfusionMatrix,1)' - diag(ConfusionMatrix)) ./ (length(XNew_Labels) - sum(ConfusionMatrix,2));
    
    %Precision değeri hesaplandı.
    Precision = diag(ConfusionMatrix) ./ sum(ConfusionMatrix,1)';
    
    %Recall değeri hesaplandı.
    Recall = TP;
    
    %F-Measure değeri hesaplandı.
    F_Measure = (2*Precision.*Recall)./(Precision+Recall);
    
    %Kappa değeri hesaplandı.
    pA = trace(ConfusionMatrix) / NumberOfInstance;
    pE = sum((sum(ConfusionMatrix,2).*sum(ConfusionMatrix,1)') ./ (NumberOfInstance*NumberOfInstance));
    output.Kappa = (pA-pE)/(1-pE);
    predict = (XNew_Labels == labelsNew);
    meanPredict = mean(predict);
    sumPredict = sum(predict);
    
    %Mean Absolute Error değeri hesaplandı.
    output.MAE = 1-meanPredict;
    
    %Root Mean Squared Error değeri hesaplandı.
    output.RMSE = sqrt(1-meanPredict);
 
    %Relative Absolute Error değeri hesaplandı.
    output.RAE = sumPredict / sum(abs(meanPredict-double(predict)));
    
    %Root Relative Squared Error değeri hesaplandı.
    output.RRSE = sqrt(sumPredict / sum((meanPredict-double(predict)).^2));
    
    %Her bir classın 'TPR' değeri hesaplandı.
    last = 0;
    TPR = zeros(length(XNew_Labels), NumberOfClass);
    for i = 1 : NumberOfClass
        first = last + 1;
        last = last + NumberOfInstanceAsClass(i);
        row = 1;
        for j = first : last
            TPR(row, i) = sum(labelsNew(first:j) == Class(i)) / NumberOfInstanceAsClass(i);
            row = row + 1;
        end
        TPR(row:end, i) = TPR(row-1, i);
    end
    
    %ROC değeri hesaplandı.
    for i = 1 : NumberOfClass
        ROC(i) = TP(i)*(1-FP(i));
    end
    
    %Detailed accuracy table adında tablo oluşturuldu ve hesapladığımız 
    %TP, FP, Precision, Recall, FMeasure, ROC değerleri yazdırıldı.
    Class{NumberOfClass+1} = '';
    RowNames = cell(NumberOfClass+1,1);
    for i = 1 : NumberOfClass
        RowNames(i) = cellstr(int2str(i));
    end
    RowNames(end) = cellstr('Weighted Avg.');
    output.DetailedAccuracyByClass = table([TP; mean(TP)], [FP; mean(FP)], [Precision; mean(Precision)], ...
                                    [Recall; mean(Recall)], [F_Measure; mean(F_Measure)], [ROC; mean(ROC)], ...
                                    cellstr(Class), ...
                                    'VariableNames', {'TP', 'FP', 'Precision', 'Recall', 'FMeasure', 'ROC', ... 
                                    'Class'}, 'RowNames', RowNames);
    
    %ePrediction(tahmini) değerleri output.Prediction değişkenine atandı.
    output.Prediction = labelsNew;
    
    %Doğru sınıflandırılan örnekler CCI değişkenine atandı.
    output.CCI = trace(ConfusionMatrix);
        
    %Hatalı sınıflandırılan örnekler ICI değişkenine atandı.
    output.ICI = length(XNew_Labels) - trace(ConfusionMatrix);
    
    %Accuracy Rate hesaplandı.
    output.AccuracyRate = output.CCI * 100 / output.NOI;
    
    %Confusion Matrix oluşturuldu.
    CM = table;
    for i = 1 : NumberOfClass
        CM(:,i) = num2cell(ConfusionMatrix(:,i));
    end
    CM.Properties.VariableNames = Class(1:end-1);
    CM.Properties.RowNames = Class(1:end-1);
    output.ConfusionMatrix = CM;
end