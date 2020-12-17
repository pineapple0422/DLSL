function [ H ] = get_H( trainClassLabels, train_labels )
% transform the train_labels to one-hot vectors
H=zeros(length(trainClassLabels),length(train_labels));

for i=1:length(trainClassLabels)
    ind=train_labels==trainClassLabels(i);
    H(i,ind)=1;
end

end

